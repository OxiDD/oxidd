use std::cell::UnsafeCell;
use std::hash::{Hash, Hasher};
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, Release};

use arcslab::AtomicRefCounted;

use oxidd_core::util::{DropWith, EdgeRefIter, Own, Ref};
use oxidd_core::{AtomicLevelNo, HasLevel, InnerNode, LevelNo};

use crate::{DiagramInfo, manager::Edge};

use super::NodeBase;

pub struct NodeWithLevel<'id, DI: DiagramInfo, const ARITY: usize> {
    rc: AtomicUsize,
    level: AtomicLevelNo,
    children: UnsafeCell<[Own<Edge<'id, DI>>; ARITY]>,
}

unsafe impl<DI: DiagramInfo, const ARITY: usize> AtomicRefCounted for NodeWithLevel<'_, DI, ARITY> {
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

impl<DI: DiagramInfo, const ARITY: usize> PartialEq for NodeWithLevel<'_, DI, ARITY> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: we have shared access to the node
        unsafe { *self.children.get() == *other.children.get() }
    }
}
impl<DI: DiagramInfo, const ARITY: usize> Eq for NodeWithLevel<'_, DI, ARITY> {}

impl<DI: DiagramInfo, const ARITY: usize> Hash for NodeWithLevel<'_, DI, ARITY> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: we have shared access to the node
        crate::util::hash_children(unsafe { &*self.children.get() }, state)
    }
}

// SAFETY: The reference counter is initialized to 2, `load_rc` uses the given
// ordering.
unsafe impl<DI: DiagramInfo, const ARITY: usize> NodeBase for NodeWithLevel<'_, DI, ARITY> {
    #[inline(always)]
    fn needs_drop() -> bool {
        false
    }

    #[inline(always)]
    fn load_rc(&self, order: atomic::Ordering) -> usize {
        self.rc.load(order)
    }
}

impl<'id, DI: DiagramInfo, const ARITY: usize> DropWith<Edge<'id, DI>>
    for NodeWithLevel<'id, DI, ARITY>
{
    #[inline]
    fn drop_with(self, drop_edge: impl Fn(Own<Edge<'id, DI>>)) {
        for c in self.children.into_inner() {
            drop_edge(c);
        }
    }
}

impl<'id, DI: DiagramInfo, const ARITY: usize> InnerNode<Edge<'id, DI>>
    for NodeWithLevel<'id, DI, ARITY>
{
    const ARITY: usize = ARITY;

    type ChildrenIter<'a>
        = EdgeRefIter<'a, Edge<'id, DI>, std::slice::Iter<'a, Own<Edge<'id, DI>>>>
    where
        Self: 'a;

    #[inline(always)]
    fn new(level: LevelNo, children: impl IntoIterator<Item = Own<Edge<'id, DI>>>) -> Self {
        let drop_fn = crate::manager::Manager::<'id, DI>::drop_edge_common;
        Self {
            rc: AtomicUsize::new(2),
            level: AtomicLevelNo::new(level),
            children: UnsafeCell::new(oxidd_core::util::collect_exact(children, drop_fn)),
        }
    }

    #[inline(always)]
    fn check_level(&self, check: impl FnOnce(LevelNo) -> bool) -> bool {
        check(self.level.load(Relaxed))
    }
    #[inline(always)]
    #[track_caller]
    fn assert_level_matches(&self, level: LevelNo) {
        assert_eq!(
            self.level.load(Relaxed),
            level,
            "the level number does not match"
        );
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
}

unsafe impl<DI: DiagramInfo, const ARITY: usize> HasLevel for NodeWithLevel<'_, DI, ARITY> {
    #[inline(always)]
    fn level(&self) -> LevelNo {
        self.level.load(Relaxed)
    }

    #[inline(always)]
    unsafe fn set_level(&self, level: LevelNo) {
        self.level.store(level, Relaxed);
    }
}

unsafe impl<DI: DiagramInfo, const ARITY: usize> Send for NodeWithLevel<'_, DI, ARITY> {}
unsafe impl<DI: DiagramInfo, const ARITY: usize> Sync for NodeWithLevel<'_, DI, ARITY> {}
