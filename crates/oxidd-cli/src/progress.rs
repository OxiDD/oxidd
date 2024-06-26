use std::hash::Hash;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oxidd::{Manager, ManagerRef};
use oxidd_core::HasApplyCache;
use parking_lot::{Mutex, MutexGuard};

// spell-checker:ignore mref

pub struct Progress {
    task: Mutex<String>,
    operations_total: AtomicUsize,
    operations_started: AtomicUsize,
    operations_done: AtomicUsize,
    print_lock: Mutex<()>,
}

#[allow(dead_code)] // we need the inner `MutexGuard`
pub struct PauseProgressHandle<'a>(MutexGuard<'a, ()>);

impl Progress {
    pub fn set_task(&self, name: impl ToString, operations: usize) {
        let name = name.to_string();
        *self.task.lock() = name;
        self.operations_total.store(operations, Relaxed);
        self.operations_done.store(0, Relaxed);
        self.operations_started.store(0, Relaxed);
    }

    pub fn start_op(&self) {
        self.operations_started.fetch_add(1, Relaxed);
    }

    pub fn finish_op(&self) {
        self.operations_done.fetch_add(1, Relaxed);
    }

    pub fn pause_progress_report(&self) -> PauseProgressHandle {
        PauseProgressHandle(self.print_lock.lock())
    }
}

pub static PROGRESS: Progress = Progress {
    task: Mutex::new(String::new()),
    operations_total: AtomicUsize::new(0),
    operations_started: AtomicUsize::new(0),
    operations_done: AtomicUsize::new(0),
    print_lock: Mutex::new(()),
};

pub struct ProgressJoinHandle {
    finish_lock: MutexGuard<'static, ()>,
    join_handle: JoinHandle<()>,
}

impl ProgressJoinHandle {
    pub fn join(self) {
        drop(self.finish_lock);
        self.join_handle.join().unwrap();
    }
}

pub fn start_progress_report<MR, O>(mref: MR, interval: Duration) -> ProgressJoinHandle
where
    MR: ManagerRef + Send + 'static,
    for<'id> MR::Manager<'id>: HasApplyCache<MR::Manager<'id>, O>,
    O: Copy + Ord + Hash,
{
    static FINISH_LOCK: Mutex<()> = Mutex::new(());

    let finish_lock = FINISH_LOCK
        .try_lock()
        .expect("background statistics have already been started");

    let join_handle = std::thread::spawn(move || {
        let start = Instant::now();
        let mut finished = false;
        while !finished {
            if FINISH_LOCK.try_lock_for(interval).is_some() {
                finished = true;
            }

            let print_lock = PROGRESS.print_lock.lock();
            let inner_nodes = mref.with_manager_shared(|manager| {
                //manager.apply_cache().print_stats();
                manager.approx_num_inner_nodes()
            });

            let task = PROGRESS.task.lock();
            let ops_total = PROGRESS.operations_total.load(Relaxed);
            let ops_done = PROGRESS.operations_done.load(Relaxed);
            let ops_started = PROGRESS.operations_started.load(Relaxed);
            println!("[{:08.2}] {task}, finished operations: {ops_done}, started {ops_started}, total: {ops_total}, inner nodes in manager: ~{inner_nodes}", start.elapsed().as_secs_f32());
            drop(task);
            drop(print_lock);
        }
    });

    ProgressJoinHandle {
        finish_lock,
        join_handle,
    }
}
