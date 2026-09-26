use std::cell::UnsafeCell;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::sync::atomic;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{Relaxed, Release};

use oxidd_core::util::{DropWith, EdgeRefIter, Own, Ref};
use oxidd_core::{AtomicLevelNo, HasLevel, InnerNode, LevelNo, Tag};

use crate::manager;
use crate::manager::InnerNodeCons;

use super::NodeBase;

pub struct NodeWithLevel<'id, ET, const ARITY: usize> {
    rc: AtomicU32,
    level: AtomicLevelNo,
    children: UnsafeCell<[Own<manager::Edge<'id, Self, ET>>; ARITY]>,
}

impl<'id, ET: Tag, const ARITY: usize> NodeWithLevel<'id, ET, ARITY> {
    const UNINIT_EDGE: MaybeUninit<Own<manager::Edge<'id, Self, ET>>> = MaybeUninit::uninit();
}

impl<ET: Tag, const ARITY: usize> PartialEq for NodeWithLevel<'_, ET, ARITY> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: we have shared access to the node
        unsafe { *self.children.get() == *other.children.get() }
    }
}
impl<ET: Tag, const ARITY: usize> Eq for NodeWithLevel<'_, ET, ARITY> {}

impl<ET: Tag, const ARITY: usize> Hash for NodeWithLevel<'_, ET, ARITY> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: we have shared access to the node
        crate::util::hash_children(unsafe { &*self.children.get() }, state);
    }
}

// SAFETY:
// - `InnerNode::new()` initializes the reference counter to 2
// - `Self::retain()` increments the reference counter by 1 with `Relaxed` order
// - `Self::release()` decrements the reference counter by 1 with `Release`
//   order
// - No other functions modify the reference counter.
// - `Self::load_rc()` loads the reference counter with the given `order`
unsafe impl<ET: Tag, const ARITY: usize> NodeBase for NodeWithLevel<'_, ET, ARITY> {
    #[inline(always)]
    fn retain(&self) {
        if self.rc.fetch_add(1, Relaxed) > (u32::MAX >> 1) {
            std::process::abort();
        }
    }

    #[inline(always)]
    unsafe fn release(&self) -> usize {
        self.rc.fetch_sub(1, Release) as usize
    }

    #[inline(always)]
    fn load_rc(&self, order: atomic::Ordering) -> usize {
        self.rc.load(order) as usize
    }

    #[inline(always)]
    fn needs_drop() -> bool {
        false
    }
}

impl<'id, ET: Tag, const ARITY: usize> DropWith<manager::Edge<'id, Self, ET>>
    for NodeWithLevel<'id, ET, ARITY>
{
    #[inline]
    fn drop_with(self, drop_edge: impl Fn(Own<manager::Edge<'id, Self, ET>>)) {
        for c in self.children.into_inner() {
            drop_edge(c);
        }
    }
}

impl<'id, ET: Tag, const ARITY: usize> InnerNode<manager::Edge<'id, Self, ET>>
    for NodeWithLevel<'id, ET, ARITY>
{
    const ARITY: usize = ARITY;

    type ChildrenIter<'a>
        = EdgeRefIter<
        'a,
        manager::Edge<'id, Self, ET>,
        std::slice::Iter<'a, Own<manager::Edge<'id, Self, ET>>>,
    >
    where
        Self: 'a;

    #[inline(always)]
    fn new(
        level: LevelNo,
        children: impl IntoIterator<Item = Own<manager::Edge<'id, Self, ET>>>,
    ) -> Self {
        let mut it = children.into_iter();
        let mut children = [Self::UNINIT_EDGE; ARITY];

        for slot in &mut children {
            slot.write(it.next().unwrap());
        }
        debug_assert!(it.next().is_none());

        // SAFETY:
        // - all elements are initialized
        // - we effectively move out of `children`; the old `children` are not
        //   dropped since they are `MaybeUninit`
        //
        // TODO: replace this by `MaybeUninit::transpose()` /
        // `MaybeUninit::array_assume_init()` once stable
        let children = unsafe {
            std::ptr::read(
                (&raw const children).cast::<[Own<manager::Edge<'id, Self, ET>>; ARITY]>(),
            )
        };

        Self {
            rc: AtomicU32::new(2),
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
        EdgeRefIter::from(unsafe { &*self.children.get() }.iter())
    }

    #[inline(always)]
    fn child(&self, n: usize) -> Ref<'_, manager::Edge<'id, Self, ET>> {
        // SAFETY: we have shared access to the node
        let children = unsafe { &*self.children.get() };
        children[n].borrowed()
    }

    #[inline(always)]
    unsafe fn set_child(
        &self,
        n: usize,
        child: Own<manager::Edge<'id, Self, ET>>,
    ) -> Own<manager::Edge<'id, Self, ET>> {
        // SAFETY: we have exclusive access to the node and no child is
        // referenced
        let children = unsafe { &mut *self.children.get() };
        std::mem::replace(&mut children[n], child)
    }

    #[inline(always)]
    fn ref_count(&self) -> usize {
        // Subtract 1 for the reference in the unique table
        (self.rc.load(Relaxed) - 1) as usize
    }
}

unsafe impl<ET, const ARITY: usize> HasLevel for NodeWithLevel<'_, ET, ARITY> {
    #[inline(always)]
    fn level(&self) -> LevelNo {
        self.level.load(Relaxed)
    }

    #[inline(always)]
    unsafe fn set_level(&self, level: LevelNo) {
        self.level.store(level, Relaxed);
    }
}

unsafe impl<ET: Send + Sync, const ARITY: usize> Send for NodeWithLevel<'_, ET, ARITY> {}
unsafe impl<ET: Send + Sync, const ARITY: usize> Sync for NodeWithLevel<'_, ET, ARITY> {}

pub struct NodeWithLevelCons<const ARITY: usize>;
impl<ET: Tag + Send + Sync, const ARITY: usize> InnerNodeCons<ET> for NodeWithLevelCons<ARITY> {
    type T<'id> = NodeWithLevel<'id, ET, ARITY>;
}
