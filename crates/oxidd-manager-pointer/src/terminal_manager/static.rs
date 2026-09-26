use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::align_of;
use std::ptr::NonNull;

use oxidd_core::util::{AllocResult, Own, Ref};
use oxidd_core::{Countable, Tag};

use crate::manager::{DiagramRulesCons, Edge, InnerNodeCons, ManagerDataCons, TerminalManagerCons};
use crate::node::NodeBase;

use super::TerminalManager;

#[repr(align(128))]
pub struct StaticTerminalManager<'id, T, N, ET, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>(
    PhantomData<(&'id (), T, N, ET, MD)>,
);

impl<T: Countable, N, ET: Tag, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    StaticTerminalManager<'_, T, N, ET, MD, PAGE_SIZE, TAG_BITS>
{
    /// All "info" bits of edges: `TAG_BITS` for the `EdgeTag`, one bit for
    /// inner/terminal node, and `bit_width(Terminal::MAX_VALUE)` bits for the
    /// terminal value.
    const ALL_BITS: u32 = TAG_BITS + 1 + (usize::BITS - T::MAX_VALUE.leading_zeros());

    /// Bit mask corresponding to `Self::ALL_BITS`
    const ALL_BITS_MASK: usize = (1 << Self::ALL_BITS) - 1;

    /// Bit indicating whether an edge points to a terminal or an inner node
    const TERMINAL_BIT: u32 = TAG_BITS;

    /// Least significant bit of the value
    const VAL_LSB: u32 = TAG_BITS + 1;

    const ASSERT_SUFFICIENT_ALIGN: () = {
        assert!(
            align_of::<Self>() >= 1 << Self::ALL_BITS,
            "Too many `TAG_BITS` / too large `Terminal::MAX_VALUE`"
        );
    };
}

unsafe impl<'id, T, N, ET, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>
    for StaticTerminalManager<'id, T, N, ET, MD, PAGE_SIZE, TAG_BITS>
where
    T: Countable + Eq + Hash,
    N: NodeBase,
    ET: Tag,
{
    type TerminalNode = T;
    type TerminalNodeRef<'a>
        = T
    where
        Self: 'a;

    type Iterator<'a>
        = StaticTerminalIterator<'id, N, ET, TAG_BITS>
    where
        Self: 'a,
        'id: 'a;

    #[inline(always)]
    unsafe fn new_in(_slot: *mut Self) {
        let () = Self::ASSERT_SUFFICIENT_ALIGN;
    }

    #[inline]
    fn terminal_manager(edge: Ref<'_, Edge<'id, N, ET, TAG_BITS>>) -> NonNull<Self> {
        assert!(!edge.raw().is_inner());
        let edge_ptr = edge.raw().as_ptr().as_ptr();
        let ptr = edge_ptr.map_addr(|p| p & !Self::ALL_BITS_MASK) as *mut Self;
        unsafe { NonNull::new_unchecked(ptr) }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        T::MAX_VALUE + 1
    }

    #[inline]
    fn deref_edge(&self, edge: Ref<'_, Edge<'id, N, ET, TAG_BITS>>) -> T {
        T::from_usize((edge.raw().addr() & Self::ALL_BITS_MASK) >> Self::VAL_LSB)
    }

    #[inline]
    fn clone_edge(edge: Ref<'_, Edge<'id, N, ET, TAG_BITS>>) -> Own<Edge<'id, N, ET, TAG_BITS>> {
        let raw = edge.raw();
        assert!(!raw.is_inner());
        let ptr = raw.as_ptr();
        unsafe { Edge::from_ptr(ptr) }
    }

    #[inline(always)]
    fn drop_edge(edge: Own<Edge<'id, N, ET, TAG_BITS>>) {
        let edge = edge.into_raw();
        debug_assert!(!edge.is_inner());
    }

    #[inline]
    unsafe fn get(this: *const Self, terminal: T) -> AllocResult<Own<Edge<'id, N, ET, TAG_BITS>>> {
        let ptr = (this as *mut ())
            .map_addr(|p| p | (1 << Self::TERMINAL_BIT) | (terminal.as_usize() << Self::VAL_LSB));
        Ok(unsafe { Edge::from_ptr(NonNull::new_unchecked(ptr)) })
    }

    #[inline]
    unsafe fn iter<'a>(this: *const Self) -> Self::Iterator<'a>
    where
        Self: 'a,
    {
        let first = (this as *mut ()).map_addr(|p| p | (1 << Self::TERMINAL_BIT));
        StaticTerminalIterator::new(NonNull::new(first).unwrap(), T::MAX_VALUE + 1)
    }

    #[inline(always)]
    fn gc(&self) -> usize {
        0 // Nothing to collect
    }
}

pub struct StaticTerminalManagerCons<Terminal>(PhantomData<Terminal>);

impl<
    T: Countable + Hash + Eq,
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    MDC: ManagerDataCons<NC, ET, Self, RC, PAGE_SIZE, TAG_BITS>,
    RC: DiagramRulesCons<NC, ET, Self, MDC, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
> TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS> for StaticTerminalManagerCons<T>
{
    type TerminalNode = T;
    type T<'id> = StaticTerminalManager<'id, T, NC::T<'id>, ET, MDC::T<'id>, PAGE_SIZE, TAG_BITS>;
}

pub struct StaticTerminalIterator<'id, N, ET, const TAG_BITS: u32> {
    ptr: NonNull<()>,
    count: usize,
    phantom: PhantomData<Edge<'id, N, ET, TAG_BITS>>,
}

impl<N, ET, const TAG_BITS: u32> StaticTerminalIterator<'_, N, ET, TAG_BITS> {
    const TERMINAL_BIT: u32 = TAG_BITS;

    const VAL_LSB: u32 = TAG_BITS + 1;

    pub fn new(first_ptr: NonNull<()>, count: usize) -> Self {
        assert!(first_ptr.as_ptr().addr() & (1 << Self::TERMINAL_BIT) != 0);
        Self {
            ptr: first_ptr,
            count,
            phantom: PhantomData,
        }
    }
}

impl<'id, N: NodeBase, ET: Tag, const TAG_BITS: u32> Iterator
    for StaticTerminalIterator<'id, N, ET, TAG_BITS>
{
    type Item = Own<Edge<'id, N, ET, TAG_BITS>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count != 0 {
            let current = self.ptr;
            self.ptr = {
                let p =
                    (self.ptr.as_ptr() as *mut u8).wrapping_offset(1 << Self::VAL_LSB) as *mut ();
                // SAFETY: cannot be null as the `TERMINAL_BIT` is set
                unsafe { NonNull::new_unchecked(p) }
            };
            self.count -= 1;

            Some(unsafe { Edge::from_ptr(current) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<N: NodeBase, ET: Tag, const TAG_BITS: u32> FusedIterator
    for StaticTerminalIterator<'_, N, ET, TAG_BITS>
{
}

impl<N: NodeBase, ET: Tag, const TAG_BITS: u32> ExactSizeIterator
    for StaticTerminalIterator<'_, N, ET, TAG_BITS>
{
    fn len(&self) -> usize {
        self.count
    }
}
