use std::cell::UnsafeCell;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, Release};

use arcslab::AtomicRefCounted;

use oxidd_core::util::{DropWith, EdgeRefIter, Own, Ref};
use oxidd_core::{AtomicLevelNo, HasLevel, InnerNode, LevelNo, Tag};

use crate::manager;
use crate::manager::InnerNodeCons;

use super::NodeBase;

pub struct NodeWithLevel<'id, ET, V, const TAG_BITS: u32, const ARITY: usize> {
    rc: AtomicUsize,
    level: AtomicLevelNo,
    children: UnsafeCell<[Own<manager::Edge<'id, Self, ET, TAG_BITS>>; ARITY]>,
    value: V,
}

impl<'id, ET: Tag, V, const TAG_BITS: u32, const ARITY: usize>
    NodeWithLevel<'id, ET, V, TAG_BITS, ARITY>
{
    const UNINIT_EDGE: MaybeUninit<Own<manager::Edge<'id, Self, ET, TAG_BITS>>> =
        MaybeUninit::uninit();
}

unsafe impl<ET, V, const TAG_BITS: u32, const ARITY: usize> AtomicRefCounted
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
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

impl<ET: Tag, V: PartialEq, const TAG_BITS: u32, const ARITY: usize> PartialEq
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: we have shared access to the node
        (unsafe { *self.children.get() == *other.children.get() }) && self.value == other.value
    }
}
impl<ET: Tag, V: Eq, const TAG_BITS: u32, const ARITY: usize> Eq
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
{
}

impl<ET: Tag, V: Hash, const TAG_BITS: u32, const ARITY: usize> Hash
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // SAFETY: we have shared access to the node
    crate::util::hash_children(unsafe { &*self.children.get() }, state);
        self.value.hash(state);
    }
}

// SAFETY: The reference counter is initialized to 2, `load_rc` uses the given
// ordering.
unsafe impl<ET: Tag, V: Eq + Hash, const TAG_BITS: u32, const ARITY: usize> NodeBase
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
{
    #[inline(always)]
    fn needs_drop() -> bool {
        false
    }

    #[inline(always)]
    fn load_rc(&self, order: atomic::Ordering) -> usize {
        self.rc.load(order)
    }
}

impl<'id, ET: Tag, V: Eq + Hash, const TAG_BITS: u32, const ARITY: usize>
    DropWith<manager::Edge<'id, Self, ET, TAG_BITS>>
    for NodeWithLevel<'id, ET, V, TAG_BITS, ARITY>
{
    #[inline]
    fn drop_with(self, drop_edge: impl Fn(Own<manager::Edge<'id, Self, ET, TAG_BITS>>)) {
        for c in self.children.into_inner() {
            drop_edge(c);
        }
    }
}

impl<'id, ET: Tag, V: Eq + Hash, const TAG_BITS: u32, const ARITY: usize>
    InnerNode<manager::Edge<'id, Self, ET, TAG_BITS>>
    for NodeWithLevel<'id, ET, V, TAG_BITS, ARITY>
{
    const ARITY: usize = 2;

    type Value = V;

    type ChildrenIter<'a>
        = EdgeRefIter<
        'a,
        manager::Edge<'id, Self, ET, TAG_BITS>,
        std::slice::Iter<'a, Own<manager::Edge<'id, Self, ET, TAG_BITS>>>,
    >
    where
        Self: 'a;

    #[inline(always)]
    fn new(
        level: LevelNo,
    children: impl IntoIterator<Item = Own<manager::Edge<'id, Self, ET, TAG_BITS>>>,
        value: V,
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
                (&raw const children)
                    .cast::<[Own<manager::Edge<'id, Self, ET, TAG_BITS>>; ARITY]>(),
            )
        };

        Self {
            rc: AtomicUsize::new(2),
            level: AtomicLevelNo::new(level),
            children: UnsafeCell::new(children),
            value,
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
    fn child(&self, n: usize) -> Ref<'_, manager::Edge<'id, Self, ET, TAG_BITS>> {
        // SAFETY: we have shared access to the node
        let children = unsafe { &*self.children.get() };
        children[n].borrowed()
    }

    #[inline(always)]
    unsafe fn set_child(
        &self,
        n: usize,
        child: Own<manager::Edge<'id, Self, ET, TAG_BITS>>,
    ) -> Own<manager::Edge<'id, Self, ET, TAG_BITS>> {
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
    fn get_value(&self) -> &V {
        &self.value
    }
}

unsafe impl<ET, V, const TAG_BITS: u32, const ARITY: usize> HasLevel
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
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

unsafe impl<ET: Send + Sync, V: Send + Sync, const TAG_BITS: u32, const ARITY: usize> Send
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
{
}
unsafe impl<ET: Send + Sync, V: Send + Sync, const TAG_BITS: u32, const ARITY: usize> Sync
    for NodeWithLevel<'_, ET, V, TAG_BITS, ARITY>
{
}

pub struct NodeWithLevelCons<V, const ARITY: usize>(PhantomData<V>);
impl<ET: Tag, V: Eq + Hash + Send + Sync, const TAG_BITS: u32, const ARITY: usize>
    InnerNodeCons<ET, TAG_BITS> for NodeWithLevelCons<V, ARITY>
{
    type T<'id> = NodeWithLevel<'id, ET, V, TAG_BITS, ARITY>;
}
