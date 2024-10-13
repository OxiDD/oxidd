use std::cell::UnsafeCell;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, Release};

use arcslab::AtomicRefCounted;

use oxidd_core::util::Borrowed;
use oxidd_core::util::BorrowedEdgeIter;
use oxidd_core::util::DropWith;
use oxidd_core::AtomicLevelNo;
use oxidd_core::Edge;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Tag;

use crate::manager;
use crate::manager::InnerNodeCons;

use super::NodeBase;

pub struct NodeWithLevel<'id, ET, const TAG_BITS: u32, const ARITY: usize> {
    rc: AtomicUsize,
    level: AtomicLevelNo,
    children: UnsafeCell<[manager::Edge<'id, Self, ET, TAG_BITS>; ARITY]>,
}

impl<'id, ET: Tag, const TAG_BITS: u32, const ARITY: usize>
    NodeWithLevel<'id, ET, TAG_BITS, ARITY>
{
    const UNINIT_EDGE: MaybeUninit<manager::Edge<'id, Self, ET, TAG_BITS>> = MaybeUninit::uninit();
}

unsafe impl<ET, const TAG_BITS: u32, const ARITY: usize> AtomicRefCounted
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
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

impl<ET: Tag, const TAG_BITS: u32, const ARITY: usize> PartialEq
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: we have shared access to the node
        unsafe { *self.children.get() == *other.children.get() }
    }
}
impl<ET: Tag, const TAG_BITS: u32, const ARITY: usize> Eq
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
}

impl<ET: Tag, const TAG_BITS: u32, const ARITY: usize> Hash
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: we have shared access to the node
        unsafe { &*self.children.get() }.hash(state);
    }
}

// SAFETY: The reference counter is initialized to 2.
unsafe impl<ET: Tag, const TAG_BITS: u32, const ARITY: usize> NodeBase
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn needs_drop() -> bool {
        false
    }
}

impl<'id, ET: Tag, const TAG_BITS: u32, const ARITY: usize>
    DropWith<manager::Edge<'id, Self, ET, TAG_BITS>> for NodeWithLevel<'id, ET, TAG_BITS, ARITY>
{
    #[inline]
    fn drop_with(self, drop_edge: impl Fn(manager::Edge<'id, Self, ET, TAG_BITS>)) {
        for c in self.children.into_inner() {
            drop_edge(c);
        }
    }
}

impl<'id, ET: Tag, const TAG_BITS: u32, const ARITY: usize>
    InnerNode<manager::Edge<'id, Self, ET, TAG_BITS>> for NodeWithLevel<'id, ET, TAG_BITS, ARITY>
{
    const ARITY: usize = 2;

    type ChildrenIter<'a>
        = BorrowedEdgeIter<
        'a,
        manager::Edge<'id, Self, ET, TAG_BITS>,
        std::slice::Iter<'a, manager::Edge<'id, Self, ET, TAG_BITS>>,
    >
    where
        Self: 'a;

    #[inline(always)]
    fn new(
        level: LevelNo,
        children: impl IntoIterator<Item = manager::Edge<'id, Self, ET, TAG_BITS>>,
    ) -> Self {
        let mut it = children.into_iter();
        let mut children = [Self::UNINIT_EDGE; ARITY];

        for slot in &mut children {
            slot.write(it.next().unwrap());
        }
        debug_assert!(it.next().is_none());

        // SAFETY:
        // - all elements are initialized
        // - we effectively move out of `children`; the old `children` are not dropped
        //   since they are `MaybeUninit`
        //
        // TODO: replace this by `MaybeUninit::transpose()` /
        // `MaybeUninit::array_assume_init()` once stable
        let children = unsafe {
            std::ptr::read(std::ptr::addr_of!(children)
                as *const [manager::Edge<'id, Self, ET, TAG_BITS>; ARITY])
        };

        Self {
            rc: AtomicUsize::new(2),
            level: AtomicLevelNo::new(level),
            children: UnsafeCell::new(children),
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
        BorrowedEdgeIter::from(unsafe { &*self.children.get() }.iter())
    }

    #[inline(always)]
    fn child(&self, n: usize) -> Borrowed<manager::Edge<'id, Self, ET, TAG_BITS>> {
        // SAFETY: we have shared access to the node
        let children = unsafe { &*self.children.get() };
        children[n].borrowed()
    }

    #[inline(always)]
    unsafe fn set_child(
        &self,
        n: usize,
        child: manager::Edge<'id, Self, ET, TAG_BITS>,
    ) -> manager::Edge<'id, Self, ET, TAG_BITS> {
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

unsafe impl<ET, const TAG_BITS: u32, const ARITY: usize> HasLevel
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn level(&self) -> LevelNo {
        self.level.load(Relaxed)
    }

    #[inline(always)]
    unsafe fn set_level(&self, level: LevelNo) {
        self.level.store(level, Relaxed);
    }
}

unsafe impl<ET: Send + Sync, const TAG_BITS: u32, const ARITY: usize> Send
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
}
unsafe impl<ET: Send + Sync, const TAG_BITS: u32, const ARITY: usize> Sync
    for NodeWithLevel<'_, ET, TAG_BITS, ARITY>
{
}

pub struct NodeWithLevelCons<const ARITY: usize>;
impl<ET: Tag, const TAG_BITS: u32, const ARITY: usize> InnerNodeCons<ET, TAG_BITS>
    for NodeWithLevelCons<ARITY>
{
    type T<'id> = NodeWithLevel<'id, ET, TAG_BITS, ARITY>;
}
