use std::borrow::Borrow;
use std::hash::Hash;

use oxidd_core::util::AllocResult;

use crate::manager::Edge;

mod dynamic;
mod r#static;
pub use dynamic::*;
pub use r#static::*;

/// Manager for terminal nodes
///
/// Several methods of this trait require valid terminal node IDs. All [`Edge`]s
/// referencing terminal nodes "contain" valid terminal node IDs. These IDs can
/// be derived from the edges by stripping the tags (which is an implementation
/// detail of the `manager` module). `Edge`s referencing terminal nodes are
/// created by the [`get_edge()`][TerminalManager::get_edge()] and
/// [`iter()`][TerminalManager::iter()] methods via
/// [`Edge::from_terminal_id()`].
pub trait TerminalManager<'id, N, ET, const TERMINALS: usize>: Sized {
    /// The terminal node type
    type TerminalNode: Eq + Hash;
    /// References to [`Self::TerminalNode`]s
    ///
    /// Should either be a `&'a Self::Terminal` or just `Self::Terminal`. The
    /// latter is useful for the [`StaticTerminalManager`] which doesn't
    /// actually store nodes but performs the mapping between edges and terminal
    /// nodes on the fly.
    type TerminalNodeRef<'a>: Borrow<Self::TerminalNode> + Copy
    where
        Self: 'a;

    /// Iterator over all terminal nodes (as [`Edge`]s), see [`Self::iter()`]
    type Iterator<'a>: Iterator<Item = Edge<'id, N, ET>>
    where
        Self: 'a;

    /// Create a new `TerminalManager`
    ///
    /// `capacity` is a hint on the maximum number of terminal nodes and may be
    /// less than `TERMINALS`. If the number of terminals is constant, the
    /// implementation may simply ignore this parameter.
    fn with_capacity(capacity: u32) -> Self;

    /// Get the number of currently stored terminals
    #[must_use]
    fn len(&self) -> usize;

    /// Returns `true` iff currently no terminals are stored
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the terminal for `id`
    ///
    /// # Safety
    ///
    /// `id` must be a valid terminal node ID for this terminal manager.
    unsafe fn get_terminal(&self, id: usize) -> Self::TerminalNodeRef<'_>;

    /// Increment the reference counter of the terminal `id`
    ///
    /// # Safety
    ///
    /// `id` must be a valid terminal node ID for this terminal manager.
    unsafe fn retain(&self, id: usize);

    /// Decrement the reference counter of the terminal `id`
    ///
    /// # Safety
    ///
    /// `id` must be a valid terminal node ID for this terminal manager.
    unsafe fn release(&self, id: usize);

    /// Add a terminal to this manager (if it does not already exist) and return
    /// an [`Edge`] pointing to it
    fn get_edge(&self, terminal: Self::TerminalNode) -> AllocResult<Edge<'id, N, ET>>;

    /// Iterate over all terminals
    fn iter<'a>(&'a self) -> Self::Iterator<'a>
    where
        Self: 'a;

    /// Garbage collection: remove unused terminals
    fn gc(&self) -> u32;
}
