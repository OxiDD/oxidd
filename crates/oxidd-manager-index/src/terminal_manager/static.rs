use std::iter::FusedIterator;
use std::marker::PhantomData;

use oxidd_core::Countable;
use oxidd_core::util::{AllocResult, Own};

use crate::{DiagramInfo, manager::Edge, terminal_manager::TerminalManager};

pub struct StaticTerminalManager<'id, DI, const TERMINALS: usize>(PhantomData<(&'id (), DI)>);

impl<'id, DI: DiagramInfo<TERMINALS>, const TERMINALS: usize> TerminalManager<'id, DI, TERMINALS>
    for StaticTerminalManager<'id, DI, TERMINALS>
where
    DI::TerminalNode: Countable,
{
    type TerminalNodeRef<'a>
        = DI::TerminalNode
    where
        Self: 'a;

    type Iterator<'a>
        = StaticTerminalIterator<'id, DI, TERMINALS>
    where
        Self: 'a,
        'id: 'a;

    fn with_capacity(_capacity: u32) -> Self {
        Self(PhantomData)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        DI::TerminalNode::MAX_VALUE + 1
    }

    #[inline]
    unsafe fn get_terminal(&self, id: usize) -> DI::TerminalNode {
        DI::TerminalNode::from_usize(id)
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
    fn get_edge(&self, terminal: DI::TerminalNode) -> AllocResult<Own<Edge<'id, DI, TERMINALS>>> {
        const { assert!(TERMINALS > DI::TerminalNode::MAX_VALUE) };
        // SAFETY: `terminal.as_usize() <= Terminal::MAX_VALUE` is guaranteed
        // and we checked `TERMINALS > DI::TerminalNode::MAX_VALUE`. There are
        // no reference counters to update.
        Ok(unsafe { Own::from_raw(Edge::from_terminal_id(terminal.as_usize() as u32)) })
    }

    #[inline]
    fn iter<'a>(&'a self) -> Self::Iterator<'a> {
        StaticTerminalIterator::new((DI::TerminalNode::MAX_VALUE + 1) as u32)
    }

    #[inline(always)]
    fn gc(&self) -> u32 {
        0 // Nothing to collect
    }
}

pub struct StaticTerminalIterator<'id, DI: DiagramInfo<TERMINALS>, const TERMINALS: usize> {
    id: u32,
    count: u32,
    phantom: PhantomData<Edge<'id, DI, TERMINALS>>,
}

impl<DI: DiagramInfo<TERMINALS>, const TERMINALS: usize> StaticTerminalIterator<'_, DI, TERMINALS> {
    #[inline(always)]
    pub fn new(count: u32) -> Self {
        Self {
            id: 0,
            count,
            phantom: PhantomData,
        }
    }
}

impl<'id, DI: DiagramInfo<TERMINALS>, const TERMINALS: usize> Iterator
    for StaticTerminalIterator<'id, DI, TERMINALS>
{
    type Item = Own<Edge<'id, DI, TERMINALS>>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.id != self.count {
            let current = self.id;
            self.id += 1;
            // SAFETY: `terminal.as_usize() <= Terminal::MAX_VALUE` is
            // guaranteed and we checked `TERMINALS > Terminal::MAX_VALUE`.
            // There are no reference counters to update.
            Some(unsafe { Own::from_raw(Edge::from_terminal_id(current)) })
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

impl<DI: DiagramInfo<TERMINALS>, const TERMINALS: usize> FusedIterator
    for StaticTerminalIterator<'_, DI, TERMINALS>
{
}

impl<DI: DiagramInfo<TERMINALS>, const TERMINALS: usize> ExactSizeIterator
    for StaticTerminalIterator<'_, DI, TERMINALS>
{
    #[inline(always)]
    fn len(&self) -> usize {
        (self.count - self.id) as usize
    }
}
