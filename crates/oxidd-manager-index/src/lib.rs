//! Index-Based Manager Implementation

//#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::type_complexity)]

pub mod manager;
pub mod node;
pub mod terminal_manager;
pub mod workers;

mod util;

#[cfg(target_pointer_width = "16")]
compile_error!("oxidd-manager-index assumes that for all `x: u32`, `x as usize` does not truncate");

/// Bundled type parameters for the manager implementation
///
/// This trait is typically implemented by a zero-sized type that is not
/// relevant otherwise.
pub trait DiagramInfo<const TERMINALS: usize>: Sized + Send + Sync {
    type EdgeTag: oxidd_core::Tag + Send + Sync;

    type InnerNode<'id>: node::NodeBase
        + oxidd_core::InnerNode<manager::Edge<'id, Self, TERMINALS>>
        + Send
        + Sync;

    type TerminalNode: Eq + std::hash::Hash + Send + Sync;
    type TerminalManager<'id>: terminal_manager::TerminalManager<'id, Self, TERMINALS> + Send + Sync;

    type DiagramRules<'id>: oxidd_core::DiagramRules<
            manager::Edge<'id, Self, TERMINALS>,
            Self::InnerNode<'id>,
            Self::TerminalNode,
        >;

    type ManagerData<'id>: oxidd_core::util::DropWith<manager::Edge<'id, Self, TERMINALS>>
        + oxidd_core::ManagerEventSubscriber<manager::Manager<'id, Self, TERMINALS>>
        + Send
        + Sync;
}
