//! Pointer-based manager implementation

//#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::type_complexity)]

pub use arcslab::{PageSize, PageSizeInfo};

pub mod manager;
pub mod node;
pub mod terminal_manager;

mod util;

#[cfg(target_pointer_width = "16")]
compile_error!(
    "oxidd-manager-pointer assumes that for all `x: u32`, `x as usize` does not truncate"
);

/// Bundled type parameters for the manager implementation
///
/// This trait is typically implemented by a zero-sized type that is not
/// relevant otherwise.
pub trait DiagramInfo: Sized + Send + Sync {
    type PageSizeInfo: PageSizeInfo;

    type EdgeTag: oxidd_core::Tag + Send + Sync;
    const TAG_BITS: u32;

    type InnerNode<'id>: node::NodeBase
        + oxidd_core::InnerNode<manager::Edge<'id, Self>>
        + Send
        + Sync;

    type TerminalNode: Eq + std::hash::Hash + Send + Sync;
    type TerminalManager<'id>: terminal_manager::TerminalManager<'id, Self> + Send + Sync;

    type Rules;

    type ManagerData<'id>: oxidd_core::util::DropWith<manager::Edge<'id, Self>>
        + oxidd_core::ManagerEventSubscriber<manager::Manager<'id, Self>>
        + Send
        + Sync;
}
