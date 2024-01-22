use std::borrow::Borrow;
use std::hash::Hash;
use std::ptr::NonNull;

use oxidd_core::util::AllocResult;

use crate::manager::Edge;

mod r#static;
pub use r#static::*;

/// Manager for terminal nodes
///
/// # Safety
///
/// [`TerminalManager::new_in()`] must properly initialize the given slot and
/// return a reference to the initialized slot.
pub unsafe trait TerminalManager<
    'id,
    InnerNode,
    EdgeTag,
    ManagerData,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>: Sized
{
    type TerminalNode: Eq + Hash;
    type TerminalNodeRef<'a>: Borrow<Self::TerminalNode> + Copy
    where
        Self: 'a;

    type Iterator<'a>: Iterator<Item = Edge<'id, InnerNode, EdgeTag, TAG_BITS>>
    where
        Self: 'a;

    /// Create a new `TerminalManager` in the given `slot`
    ///
    /// # Safety
    ///
    /// `slot` is valid for writes and properly aligned. When returning from
    /// this function, the location referenced by `slot` is initialized. With
    /// respect to Stacked / Tree borrows, `slot` is tagged as the root of the
    /// allocation.
    unsafe fn new_in(slot: *mut Self);

    /// Get a pointer to the terminal store
    fn terminal_manager(edge: &Edge<'id, InnerNode, EdgeTag, TAG_BITS>) -> NonNull<Self>;

    /// Get the number of currently stored terminals
    #[must_use]
    fn len(&self) -> usize;

    /// Returns `true` iff currently no terminals are stored
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dereference the given `edge`
    fn deref_edge(
        &self,
        edge: &Edge<'id, InnerNode, EdgeTag, TAG_BITS>,
    ) -> Self::TerminalNodeRef<'_>;

    /// Clone the given `edge`
    fn clone_edge(
        edge: &Edge<'id, InnerNode, EdgeTag, TAG_BITS>,
    ) -> Edge<'id, InnerNode, EdgeTag, TAG_BITS>;

    /// Drop the given `edge`
    fn drop_edge(edge: Edge<'id, InnerNode, EdgeTag, TAG_BITS>);

    /// Add a terminal to this manager (if it does not already exist) and return
    /// an [`Edge`] pointing to it
    ///
    /// # Safety
    ///
    /// `this` is valid for reads, properly aligned and initialized. With
    /// respect to Stacked / Tree Borrows, `this` is tagged as the root of the
    /// allocation.
    unsafe fn get(
        this: *const Self,
        terminal: Self::TerminalNode,
    ) -> AllocResult<Edge<'id, InnerNode, EdgeTag, TAG_BITS>>;

    /// Iterate over all terminals
    ///
    /// # Safety
    ///
    /// `this` is valid for reads, properly aligned and initialized during `'a`.
    /// With respect to Stacked / Tree Borrows, `this` is tagged as the root of
    /// the allocation.
    unsafe fn iter<'a>(this: *const Self) -> Self::Iterator<'a>
    where
        Self: 'a;

    /// Garbage collection: remove unused terminals
    fn gc(&self) -> usize;
}
