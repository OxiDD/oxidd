use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;

use oxidd_core::util::AllocResult;
use oxidd_core::Countable;
use oxidd_core::Tag;

use crate::manager::Edge;
use crate::manager::InnerNodeCons;
use crate::manager::TerminalManagerCons;
use crate::node::NodeBase;

use super::TerminalManager;

pub struct StaticTerminalManager<'id, T, N, ET, const TERMINALS: usize>(
    PhantomData<(&'id (), T, N, ET)>,
);

impl<T: Countable, N, ET: Tag, const TERMINALS: usize>
    StaticTerminalManager<'_, T, N, ET, TERMINALS>
{
    const CHECK_TERMINALS: () = assert!(TERMINALS > T::MAX_VALUE);
}

impl<'id, Terminal, InnerNode, EdgeTag, const TERMINALS: usize>
    TerminalManager<'id, InnerNode, EdgeTag, TERMINALS>
    for StaticTerminalManager<'id, Terminal, InnerNode, EdgeTag, TERMINALS>
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
        = StaticTerminalIterator<'id, InnerNode, EdgeTag>
    where
        Self: 'a,
        'id: 'a;

    fn with_capacity(_capacity: u32) -> Self {
        let _ = Self::CHECK_TERMINALS;
        Self(PhantomData)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        Terminal::MAX_VALUE + 1
    }

    #[inline]
    unsafe fn get_terminal(&self, id: usize) -> Terminal {
        Terminal::from_usize(id)
    }

    #[inline(always)]
    unsafe fn retain(&self, _id: usize) {
        // Nothing to do
    }

    #[inline(always)]
    unsafe fn release(&self, _id: usize) {
        // Nothing to do
    }

    #[inline]
    fn get_edge(&self, terminal: Terminal) -> AllocResult<Edge<'id, InnerNode, EdgeTag>> {
        let _ = Self::CHECK_TERMINALS;
        // SAFETY: `terminal.as_usize() <= Terminal::MAX_VALUE` is guaranteed
        // and we checked `TERMINALS > Terminal::MAX_VALUE`. There are no
        // reference counters to update.
        Ok(unsafe { Edge::from_terminal_id(terminal.as_usize() as u32) })
    }

    #[inline]
    fn iter<'a>(&'a self) -> Self::Iterator<'a>
    where
        Self: 'a,
    {
        StaticTerminalIterator::new((Terminal::MAX_VALUE + 1) as u32)
    }

    #[inline(always)]
    fn gc(&self) -> u32 {
        0 // Nothing to collect
    }
}

pub struct StaticTerminalManagerCons<Terminal>(PhantomData<Terminal>);

impl<T, NC, ET, const TERMINALS: usize> TerminalManagerCons<NC, ET, TERMINALS>
    for StaticTerminalManagerCons<T>
where
    T: Countable + Hash + Eq + Send + Sync,
    NC: InnerNodeCons<ET>,
    ET: Tag + Send + Sync,
{
    type TerminalNode = T;
    type T<'id> = StaticTerminalManager<'id, T, NC::T<'id>, ET, TERMINALS>;
}

pub struct StaticTerminalIterator<'id, InnerNode, EdgeTag> {
    id: u32,
    count: u32,
    phantom: PhantomData<Edge<'id, InnerNode, EdgeTag>>,
}

impl<InnerNode, EdgeTag> StaticTerminalIterator<'_, InnerNode, EdgeTag> {
    #[inline(always)]
    pub fn new(count: u32) -> Self {
        Self {
            id: 0,
            count,
            phantom: PhantomData,
        }
    }
}

impl<'id, InnerNode: NodeBase, EdgeTag: Tag> Iterator
    for StaticTerminalIterator<'id, InnerNode, EdgeTag>
{
    type Item = Edge<'id, InnerNode, EdgeTag>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.id != self.count {
            let current = self.id;
            self.id += 1;
            // SAFETY: `terminal.as_usize() <= Terminal::MAX_VALUE` is
            // guaranteed and we checked `TERMINALS > Terminal::MAX_VALUE`.
            // There are no reference counters to update.
            Some(unsafe { Edge::from_terminal_id(current) })
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<InnerNode: NodeBase, EdgeTag: Tag> FusedIterator
    for StaticTerminalIterator<'_, InnerNode, EdgeTag>
{
}

impl<InnerNode: NodeBase, EdgeTag: Tag> ExactSizeIterator
    for StaticTerminalIterator<'_, InnerNode, EdgeTag>
{
    #[inline(always)]
    fn len(&self) -> usize {
        (self.count - self.id) as usize
    }
}
