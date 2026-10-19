use std::cell::UnsafeCell;
use std::hash::{Hash, Hasher};
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, Release};

use arcslab::AtomicRefCounted;

use oxidd_core::util::{DropWith, EdgeRefIter, Own, Ref};
use oxidd_core::{AtomicLevelNo, HasLevel, InnerNode, LevelNo};

use crate::DiagramInfo;
use crate::manager::Edge;
use crate::node::{LevelInfo, NodeBase};

pub struct Node<'id, DI: DiagramInfo, L, const ARITY: usize> {
    rc: AtomicUsize,
    level: L,
    children: UnsafeCell<[Own<Edge<'id, DI>>; ARITY]>,
    value: UnsafeCell<DI::InnerNodeValue>,
}

unsafe impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> AtomicRefCounted
    for Node<'_, DI, L, ARITY>
{
    #[inline(always)]
    fn retain(&self) {
        if self.rc.fetch_add(1, Relaxed) > (usize::MAX >> 1) {
            std::process::abort();
        }
    }

    #[inline(always)]
    unsafe fn release(&self) -> usize {
        self.rc.fetch_sub(1, Release)
    }

    #[inline(always)]
    fn current(&self) -> usize {
        self.rc.load(Relaxed)
    }
}

impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> PartialEq for Node<'_, DI, L, ARITY> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: by default, we have shared access to the node. If a user
        // wants to modify the node, it is their responsibility to ensure
        // exclusive access.
        unsafe {
            *self.children.get() == *other.children.get() && *self.value.get() == *other.value.get()
        }
    }
}
impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> Eq for Node<'_, DI, L, ARITY> {}

impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> Hash for Node<'_, DI, L, ARITY> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: we have shared access to the node (as detailed above)
        crate::util::hash_children(unsafe { &*self.children.get() }, state);
        // SAFETY: as above
        unsafe { &*self.value.get() }.hash(state);
    }
}

// SAFETY: The reference counter is initialized to 2, `load_rc` uses the given
// ordering.
unsafe impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> NodeBase for Node<'_, DI, L, ARITY> {
    #[inline(always)]
    fn needs_drop() -> bool {
        std::mem::needs_drop::<DI::InnerNodeValue>()
    }

    #[inline(always)]
    fn load_rc(&self, order: atomic::Ordering) -> usize {
        self.rc.load(order)
    }
}

impl<'id, DI: DiagramInfo, L: LevelInfo, const ARITY: usize> DropWith<Edge<'id, DI>>
    for Node<'id, DI, L, ARITY>
{
    #[inline]
    fn drop_with(self, drop_edge: impl Fn(Own<Edge<'id, DI>>)) {
        for c in self.children.into_inner() {
            drop_edge(c);
        }
    }
}

impl<'id, DI: DiagramInfo, L: LevelInfo, const ARITY: usize> InnerNode<Edge<'id, DI>>
    for Node<'id, DI, L, ARITY>
{
    const ARITY: usize = ARITY;

    type Value = DI::InnerNodeValue;

    type ChildrenIter<'a>
        = EdgeRefIter<'a, Edge<'id, DI>, std::slice::Iter<'a, Own<Edge<'id, DI>>>>
    where
        Self: 'a;

    #[inline(always)]
    fn new(
        level: LevelNo,
        children: impl IntoIterator<Item = Own<Edge<'id, DI>>>,
        value: DI::InnerNodeValue,
    ) -> Self {
        let drop_fn = crate::manager::Manager::<'id, DI>::drop_edge_common;
        Self {
            rc: AtomicUsize::new(2),
            level: L::from(level),
            children: UnsafeCell::new(oxidd_core::util::collect_exact(children, drop_fn)),
            value: UnsafeCell::new(value),
        }
    }

    #[inline(always)]
    fn check_level(&self, check: impl FnOnce(LevelNo) -> bool) -> bool {
        self.level.check(check)
    }
    #[inline(always)]
    #[track_caller]
    fn assert_level_matches(&self, level: LevelNo) {
        self.level.assert_matches(level);
    }

    #[inline(always)]
    fn children(&self) -> Self::ChildrenIter<'_> {
        // SAFETY: we have shared access to the node
        EdgeRefIter::from(unsafe { &*self.children.get() }.iter())
    }

    #[inline(always)]
    fn child(&self, n: usize) -> Ref<'_, Edge<'id, DI>> {
        // SAFETY: we have shared access to the node
        let children = unsafe { &*self.children.get() };
        children[n].borrowed()
    }

    #[inline(always)]
    unsafe fn set_child(&self, n: usize, child: Own<Edge<'id, DI>>) -> Own<Edge<'id, DI>> {
        // SAFETY: we have exclusive access to the node and no child is
        // referenced
        let children = unsafe { &mut *self.children.get() };
        std::mem::replace(&mut children[n], child)
    }

    #[inline(always)]
    fn ref_count(&self) -> usize {
        // Subtract 1 for the reference in the unique table
        self.rc.load(Relaxed) - 1
    }

    #[inline(always)]
    fn value(&self) -> &Self::Value {
        // SAFETY: we have shared access to the node
        unsafe { &*self.value.get() }
    }

    #[inline(always)]
    unsafe fn value_mut(&self) -> &mut Self::Value {
        // SAFETY: we have shared access to the node
        unsafe { &mut *self.value.get() }
    }
}

unsafe impl<DI: DiagramInfo, const ARITY: usize> HasLevel for Node<'_, DI, AtomicLevelNo, ARITY> {
    #[inline(always)]
    fn level(&self) -> LevelNo {
        self.level.load(Relaxed)
    }

    #[inline(always)]
    unsafe fn set_level(&self, level: LevelNo) {
        self.level.store(level, Relaxed);
    }
}

unsafe impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> Send for Node<'_, DI, L, ARITY> {}
unsafe impl<DI: DiagramInfo, L: LevelInfo, const ARITY: usize> Sync for Node<'_, DI, L, ARITY> {}
