use super::*;
use crate::node::fixed_arity::NodeWithLevelCons;
use crate::terminal_manager::StaticTerminalManagerCons;
use oxidd_core::Manager as _;
use oxidd_core::ManagerRef as _;
use oxidd_core::function::Function as _;
use oxidd_rules_bdd::simple::{BDDRules, BDDTerminal};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

type Nodes = NodeWithLevelCons<2>;
type Terminals = StaticTerminalManagerCons<BDDTerminal>;
type TestRef = ManagerRef<Nodes, (), Terminals, Rules, Data, 2>;
type TestFunc = Function<Nodes, (), Terminals, Rules, Data, 2>;
type TestManager<'id> = M<'id, Nodes, (), Terminals, Rules, Data, 2>;
type TestEdge<'id> = Edge<'id, <Nodes as InnerNodeCons<()>>::T<'id>, ()>;

struct Rules;
impl DiagramRulesCons<Nodes, (), Terminals, Data, 2> for Rules {
    type T<'id> = BDDRules;
}

struct Data;
impl ManagerDataCons<Nodes, (), Terminals, Rules, 2> for Data {
    type T<'id> = TestData;
}

struct TestData {
    hook: Option<Arc<Hook>>,
}

impl<'id> DropWith<TestEdge<'id>> for TestData {
    fn drop_with(self, _drop_edge: impl Fn(TestEdge<'id>)) {}
}

struct Hook {
    entered: Sender<()>,
    resume: Mutex<Receiver<()>>,
    revived: Option<Sender<TestRef>>,
    revived_once: std::sync::atomic::AtomicBool,
    drop_on_worker: bool,
}

impl<'id> ManagerEventSubscriber<TestManager<'id>> for TestData {
    fn pre_gc(&self, manager: &TestManager<'id>) {
        if let Some(hook) = &self.hook {
            let _ = hook.entered.send(());
            let _ = hook.resume.lock().recv();
            if let Some(revived) = &hook.revived
                && !hook.revived_once.swap(true, Relaxed)
            {
                let _ = revived.send(TestRef::from(manager));
            }
            if hook.drop_on_worker {
                drop(TestRef::from(manager));
            }
        }
    }
}

fn new_test_manager(hook: Option<Arc<Hook>>) -> TestRef {
    new_manager::<Nodes, (), Terminals, Rules, Data, 2>(1024, 2, 1, TestData { hook })
}

fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    condition()
}

fn wait_for_worker_gc(manager: &TestRef, entered: &Receiver<()>) {
    assert!(
        wait_until(|| {
            manager.0.gc_signal.1.notify_one();
            entered.try_recv().is_ok()
        }),
        "this manager's GC worker did not enter the callback"
    );
}

#[test]
fn repeated_manager_lifetimes_on_one_thread() {
    for lifetime in 0..100 {
        let manager = new_test_manager(None);
        let worker = manager.0.gc_thread.lock().as_ref().unwrap().thread().id();
        assert_ne!(worker, std::thread::current().id());
        let weak = Arc::downgrade(&manager.0);
        let function = manager.with_manager_shared(|m| {
            TestFunc::from_edge(m, m.get_terminal(BDDTerminal::True).unwrap())
        });
        drop(manager);
        assert!(weak.upgrade().is_some(), "function lost store {lifetime}");
        drop(function);
        assert!(weak.upgrade().is_none(), "worker survived {lifetime}");
    }
}

#[test]
fn concurrent_final_refs_retire_gc() {
    for lifetime in 0..100 {
        let manager = new_test_manager(None);
        let weak = Arc::downgrade(&manager.0);
        let other = manager.clone();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let first_barrier = barrier.clone();
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            drop(manager);
        });
        let second_barrier = barrier.clone();
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            drop(other);
        });
        barrier.wait();
        first.join().unwrap();
        second.join().unwrap();
        assert!(weak.upgrade().is_none(), "worker survived {lifetime}");
    }
}

#[test]
fn final_drop_during_background_gc() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let manager = new_test_manager(Some(Arc::new(Hook {
        entered: entered_tx,
        resume: Mutex::new(resume_rx),
        revived: None,
        revived_once: std::sync::atomic::AtomicBool::new(false),
        drop_on_worker: false,
    })));
    let weak = Arc::downgrade(&manager.0);
    wait_for_worker_gc(&manager, &entered_rx);

    let (done_tx, done_rx) = mpsc::channel();
    let dropping = std::thread::spawn(move || {
        drop(manager);
        done_tx.send(()).unwrap();
    });
    let reached_zero = wait_until(|| {
        weak.upgrade()
            .is_some_and(|store| store.external_refs.load(Acquire) == 0)
    });
    let still_waiting = done_rx.try_recv().is_err();
    resume_tx.send(()).unwrap();
    assert!(reached_zero, "final drop did not reach zero");
    assert!(still_waiting, "final drop returned during collection");
    done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    dropping.join().unwrap();
    assert!(weak.upgrade().is_none(), "this manager's worker survived");
}

#[test]
fn gc_callback_revives_zero_external_count() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (revived_tx, revived_rx) = mpsc::channel();
    let manager = new_test_manager(Some(Arc::new(Hook {
        entered: entered_tx,
        resume: Mutex::new(resume_rx),
        revived: Some(revived_tx),
        revived_once: std::sync::atomic::AtomicBool::new(false),
        drop_on_worker: false,
    })));
    let weak = Arc::downgrade(&manager.0);
    wait_for_worker_gc(&manager, &entered_rx);

    let (done_tx, done_rx) = mpsc::channel();
    let dropping = std::thread::spawn(move || {
        drop(manager);
        done_tx.send(()).unwrap();
    });
    let reached_zero = wait_until(|| {
        weak.upgrade()
            .is_some_and(|store| store.external_refs.load(Acquire) == 0)
    });
    resume_tx.send(()).unwrap();
    assert!(reached_zero, "final drop did not reach zero");
    let revived = revived_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    dropping.join().unwrap();
    assert!(weak.upgrade().is_some(), "revived handle lost its store");
    wait_for_worker_gc(&revived, &entered_rx);
    let (final_done_tx, final_done_rx) = mpsc::channel();
    let final_drop = std::thread::spawn(move || {
        drop(revived);
        final_done_tx.send(()).unwrap();
    });
    let reached_zero_again = wait_until(|| {
        weak.upgrade()
            .is_some_and(|store| store.external_refs.load(Acquire) == 0)
    });
    resume_tx.send(()).unwrap();
    assert!(reached_zero_again, "revived handle did not reach zero");
    final_done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    final_drop.join().unwrap();
    assert!(
        weak.upgrade().is_none(),
        "revived handle did not retire worker"
    );
}

#[test]
fn gc_worker_can_drop_the_last_transient_handle() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let manager = new_test_manager(Some(Arc::new(Hook {
        entered: entered_tx,
        resume: Mutex::new(resume_rx),
        revived: None,
        revived_once: std::sync::atomic::AtomicBool::new(false),
        drop_on_worker: true,
    })));
    let weak = Arc::downgrade(&manager.0);
    wait_for_worker_gc(&manager, &entered_rx);

    let (done_tx, done_rx) = mpsc::channel();
    let dropping = std::thread::spawn(move || {
        drop(manager);
        done_tx.send(()).unwrap();
    });
    let reached_zero = wait_until(|| {
        weak.upgrade()
            .is_some_and(|store| store.external_refs.load(Acquire) == 0)
    });
    resume_tx.send(()).unwrap();
    assert!(reached_zero, "final drop did not reach zero");
    done_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    dropping.join().unwrap();
    assert!(weak.upgrade().is_none(), "this manager's worker survived");
}
