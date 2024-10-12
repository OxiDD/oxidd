use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::align_of;
use std::ptr::NonNull;

use oxidd_core::util::AllocResult;
use oxidd_core::Countable;
use oxidd_core::Tag;

use crate::manager::DiagramRulesCons;
use crate::manager::Edge;
use crate::manager::InnerNodeCons;
use crate::manager::ManagerDataCons;
use crate::manager::TerminalManagerCons;
use crate::node::NodeBase;

use super::TerminalManager;

#[repr(align(128))]
pub struct StaticTerminalManager<
    'id,
    Terminal,
    InnerNode,
    EdgeTag,
    ManagerData,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>(PhantomData<(&'id (), Terminal, InnerNode, EdgeTag, ManagerData)>);

impl<
        Terminal: Countable,
        InnerNode,
        EdgeTag: Tag,
        ManagerData,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > StaticTerminalManager<'_, Terminal, InnerNode, EdgeTag, ManagerData, PAGE_SIZE, TAG_BITS>
{
    /// All "info" bits of edges: `TAG_BITS` for the `EdgeTag`, one bit for
    /// inner/terminal node, and `bit_width(Terminal::MAX_VALUE)` bits for the
    /// terminal value.
    const ALL_BITS: u32 = TAG_BITS + 1 + (usize::BITS - Terminal::MAX_VALUE.leading_zeros());

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

unsafe impl<
        'id,
        Terminal,
        InnerNode,
        EdgeTag,
        ManagerData,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > TerminalManager<'id, InnerNode, EdgeTag, ManagerData, PAGE_SIZE, TAG_BITS>
    for StaticTerminalManager<'id, Terminal, InnerNode, EdgeTag, ManagerData, PAGE_SIZE, TAG_BITS>
where
    Terminal: Countable + Eq + Hash,
    InnerNode: NodeBase,
    EdgeTag: Tag,
{
    type TerminalNode = Terminal;
    type TerminalNodeRef<'a>
        = Terminal
    where
        Self: 'a;

    type Iterator<'a>
        = StaticTerminalIterator<'id, InnerNode, EdgeTag, TAG_BITS>
    where
        Self: 'a,
        'id: 'a;

    #[inline(always)]
    unsafe fn new_in(_slot: *mut Self) {
        let _ = Self::ASSERT_SUFFICIENT_ALIGN;
    }

    #[inline]
    fn terminal_manager(edge: &Edge<'id, InnerNode, EdgeTag, TAG_BITS>) -> NonNull<Self> {
        assert!(!edge.is_inner());
        let ptr = sptr::Strict::map_addr(edge.as_ptr().as_ptr(), |p| p & !Self::ALL_BITS_MASK)
            as *mut Self;
        unsafe { NonNull::new_unchecked(ptr) }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        Terminal::MAX_VALUE + 1
    }

    #[inline]
    fn deref_edge(&self, edge: &Edge<'id, InnerNode, EdgeTag, TAG_BITS>) -> Terminal {
        Terminal::from_usize((edge.addr() & Self::ALL_BITS_MASK) >> Self::VAL_LSB)
    }

    #[inline]
    fn clone_edge(
        edge: &Edge<'id, InnerNode, EdgeTag, TAG_BITS>,
    ) -> Edge<'id, InnerNode, EdgeTag, TAG_BITS> {
        assert!(!edge.is_inner());
        let ptr = edge.as_ptr();
        unsafe { Edge::from_ptr(ptr) }
    }

    #[inline(always)]
    fn drop_edge(edge: Edge<'id, InnerNode, EdgeTag, TAG_BITS>) {
        debug_assert!(!edge.is_inner());
        std::mem::forget(edge)
    }

    #[inline]
    unsafe fn get(
        this: *const Self,
        terminal: Terminal,
    ) -> AllocResult<Edge<'id, InnerNode, EdgeTag, TAG_BITS>> {
        let ptr = sptr::Strict::map_addr(this as *mut (), |p| {
            p | 1 << Self::TERMINAL_BIT | terminal.as_usize() << Self::VAL_LSB
        });
        Ok(unsafe { Edge::from_ptr(NonNull::new_unchecked(ptr)) })
    }

    #[inline]
    unsafe fn iter<'a>(this: *const Self) -> Self::Iterator<'a>
    where
        Self: 'a,
    {
        let first = sptr::Strict::map_addr(this as *mut (), |p| p | 1 << Self::TERMINAL_BIT);
        StaticTerminalIterator::new(NonNull::new(first).unwrap(), Terminal::MAX_VALUE + 1)
    }

    #[inline(always)]
    fn gc(&self) -> usize {
        0 // Nothing to collect
    }
}

pub struct StaticTerminalManagerCons<Terminal>(PhantomData<Terminal>);

impl<
        Terminal: Countable + Hash + Eq,
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        MDC: ManagerDataCons<NC, ET, Self, RC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, Self, MDC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>
    for StaticTerminalManagerCons<Terminal>
{
    type TerminalNode = Terminal;
    type T<'id> =
        StaticTerminalManager<'id, Terminal, NC::T<'id>, ET, MDC::T<'id>, PAGE_SIZE, TAG_BITS>;
}

pub struct StaticTerminalIterator<'id, InnerNode, EdgeTag, const TAG_BITS: u32> {
    ptr: NonNull<()>,
    count: usize,
    phantom: PhantomData<Edge<'id, InnerNode, EdgeTag, TAG_BITS>>,
}

impl<InnerNode, EdgeTag, const TAG_BITS: u32>
    StaticTerminalIterator<'_, InnerNode, EdgeTag, TAG_BITS>
{
    const TERMINAL_BIT: u32 = TAG_BITS;

    const VAL_LSB: u32 = TAG_BITS + 1;

    pub fn new(first_ptr: NonNull<()>, count: usize) -> Self {
        assert!(sptr::Strict::addr(first_ptr.as_ptr()) & (1 << Self::TERMINAL_BIT) != 0);
        Self {
            ptr: first_ptr,
            count,
            phantom: PhantomData,
        }
    }
}

impl<'id, InnerNode: NodeBase, EdgeTag: Tag, const TAG_BITS: u32> Iterator
    for StaticTerminalIterator<'id, InnerNode, EdgeTag, TAG_BITS>
{
    type Item = Edge<'id, InnerNode, EdgeTag, TAG_BITS>;

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

impl<InnerNode: NodeBase, EdgeTag: Tag, const TAG_BITS: u32> FusedIterator
    for StaticTerminalIterator<'_, InnerNode, EdgeTag, TAG_BITS>
{
}

impl<InnerNode: NodeBase, EdgeTag: Tag, const TAG_BITS: u32> ExactSizeIterator
    for StaticTerminalIterator<'_, InnerNode, EdgeTag, TAG_BITS>
{
    fn len(&self) -> usize {
        self.count
    }
}
