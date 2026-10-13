use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::align_of;
use std::ptr::NonNull;

use oxidd_core::Countable;
use oxidd_core::util::{AllocResult, Own, Ref};

use crate::{DiagramInfo, manager::Edge};

use super::TerminalManager;

#[repr(align(128))]
pub struct StaticTerminalManager<'id, DI, const PAGE_SIZE: usize, const TAG_BITS: u32>(
    PhantomData<(&'id (), DI)>,
);

impl<DI, const PAGE_SIZE: usize, const TAG_BITS: u32>
    StaticTerminalManager<'_, DI, PAGE_SIZE, TAG_BITS>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
    DI::TerminalNode: Countable,
{
    /// All "info" bits of edges: `TAG_BITS` for the `EdgeTag`, one bit for
    /// inner/terminal node, and `bit_width(Terminal::MAX_VALUE)` bits for the
    /// terminal value.
    const ALL_BITS: u32 =
        TAG_BITS + 1 + (usize::BITS - DI::TerminalNode::MAX_VALUE.leading_zeros());

    /// Bit mask corresponding to `Self::ALL_BITS`
    const ALL_BITS_MASK: usize = (1 << Self::ALL_BITS) - 1;

    /// Bit indicating whether an edge points to a terminal or an inner node
    const TERMINAL_BIT: u32 = TAG_BITS;

    /// Least significant bit of the value
    const VAL_LSB: u32 = TAG_BITS + 1;
}

unsafe impl<'id, DI, const PAGE_SIZE: usize, const TAG_BITS: u32>
    TerminalManager<'id, DI, PAGE_SIZE, TAG_BITS>
    for StaticTerminalManager<'id, DI, PAGE_SIZE, TAG_BITS>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
    DI::TerminalNode: Countable,
{
    type TerminalNodeRef<'a>
        = DI::TerminalNode
    where
        Self: 'a;

    type Iterator<'a>
        = StaticTerminalIterator<'id, DI, PAGE_SIZE, TAG_BITS>
    where
        Self: 'a,
        'id: 'a;

    #[inline(always)]
    unsafe fn new_in(_slot: *mut Self) {
        const {
            assert!(
                align_of::<Self>() >= 1 << Self::ALL_BITS,
                "Too many `TAG_BITS` / too large `Terminal::MAX_VALUE`"
            )
        };
    }

    #[inline]
    fn terminal_manager(edge: Ref<'_, Edge<'id, DI, PAGE_SIZE, TAG_BITS>>) -> NonNull<Self> {
        assert!(!edge.raw().is_inner());
        let edge_ptr = edge.raw().as_ptr().as_ptr();
        let ptr = edge_ptr.map_addr(|p| p & !Self::ALL_BITS_MASK) as *mut Self;
        unsafe { NonNull::new_unchecked(ptr) }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        DI::TerminalNode::MAX_VALUE + 1
    }

    #[inline]
    fn deref_edge(&self, edge: Ref<'_, Edge<'id, DI, PAGE_SIZE, TAG_BITS>>) -> DI::TerminalNode {
        DI::TerminalNode::from_usize((edge.raw().addr() & Self::ALL_BITS_MASK) >> Self::VAL_LSB)
    }

    #[inline]
    fn clone_edge(
        edge: Ref<'_, Edge<'id, DI, PAGE_SIZE, TAG_BITS>>,
    ) -> Own<Edge<'id, DI, PAGE_SIZE, TAG_BITS>> {
        let raw = edge.raw();
        assert!(!raw.is_inner());
        let ptr = raw.as_ptr();
        unsafe { Edge::from_ptr(ptr) }
    }

    #[inline(always)]
    fn drop_edge(edge: Own<Edge<'id, DI, PAGE_SIZE, TAG_BITS>>) {
        let edge = edge.into_raw();
        debug_assert!(!edge.is_inner());
    }

    #[inline]
    unsafe fn get(
        this: *const Self,
        terminal: DI::TerminalNode,
    ) -> AllocResult<Own<Edge<'id, DI, PAGE_SIZE, TAG_BITS>>> {
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
        StaticTerminalIterator::new(
            NonNull::new(first).unwrap(),
            DI::TerminalNode::MAX_VALUE + 1,
        )
    }

    #[inline(always)]
    fn gc(&self) -> usize {
        0 // Nothing to collect
    }
}

pub struct StaticTerminalIterator<'id, DI, const PAGE_SIZE: usize, const TAG_BITS: u32>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
{
    ptr: NonNull<()>,
    count: usize,
    phantom: PhantomData<Edge<'id, DI, PAGE_SIZE, TAG_BITS>>,
}

impl<DI, const PAGE_SIZE: usize, const TAG_BITS: u32>
    StaticTerminalIterator<'_, DI, PAGE_SIZE, TAG_BITS>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
{
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

impl<'id, DI, const PAGE_SIZE: usize, const TAG_BITS: u32> Iterator
    for StaticTerminalIterator<'id, DI, PAGE_SIZE, TAG_BITS>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
{
    type Item = Own<Edge<'id, DI, PAGE_SIZE, TAG_BITS>>;

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

impl<DI, const PAGE_SIZE: usize, const TAG_BITS: u32> FusedIterator
    for StaticTerminalIterator<'_, DI, PAGE_SIZE, TAG_BITS>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
{
}

impl<DI, const PAGE_SIZE: usize, const TAG_BITS: u32> ExactSizeIterator
    for StaticTerminalIterator<'_, DI, PAGE_SIZE, TAG_BITS>
where
    DI: DiagramInfo<PAGE_SIZE, TAG_BITS>,
{
    fn len(&self) -> usize {
        self.count
    }
}
