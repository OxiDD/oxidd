pub mod fixed_arity;

/// Base traits to be satisfied by inner nodes for oxidd-manager. This does not
/// include the [`InnerNode`][oxidd_core::InnerNode] trait.
///
/// # Safety
///
/// The reference counter must be initialized to 2.
pub unsafe trait NodeBase: arcslab::AtomicRefCounted + Eq + std::hash::Hash {
    /// Whether this node type contains additional data that needs to be dropped
    /// to avoid memory leaks.
    ///
    /// If the node only consists of [`Edge`][crate::manager::Edge]s and types
    /// that implement [`Copy`], an implementation may return `false`. This may
    /// speed up dropping a diagram a lot.
    #[inline(always)]
    fn needs_drop() -> bool {
        true
    }
}
