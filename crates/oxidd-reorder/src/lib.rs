//! Functionality to reorder levels in a decision diagram

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use oxidd_core::{HasLevel, LevelNo, LevelSwapper, LevelView, Manager, Node};

mod set_var_order;
pub use set_var_order::{set_var_order, set_var_order_seq};

/// Swap the level given by `upper_no` with the level directly below.
///
/// # Safety
///
/// Must be called from inside the closure of
/// [`manager.reorder()`][Manager::reorder]. `manager` must be derived from a
/// `&mut M` reference. This function may modify nodes at the level of
/// `upper_no` and `upper_no + 1` (i.e., the level below). There must not be any
/// concurrent modification of any nodes at these levels.
pub unsafe fn level_down<M: Manager>(manager: &M, upper_no: LevelNo)
where
    M::InnerNode: HasLevel,
    M::Rules: LevelSwapper<M>,
{
    assert!(upper_no + 1 < manager.num_levels());
    unsafe { M::Rules::swap(manager, upper_no, upper_no + 1, upper_no, upper_no + 1) };
    // SAFETY (next 2): follows from the above assertion
    update_level_no(manager, &unsafe { manager.level_unchecked(upper_no) });
    update_level_no(manager, &unsafe { manager.level_unchecked(upper_no + 1) });
}

/// Write the level number `level` to all nodes referenced from the `level`-th
/// level view.
fn update_level_no<M: Manager>(manager: &M, level: &M::LevelView<'_>)
where
    M::InnerNode: HasLevel,
{
    let level_no = level.level_no();
    for e in level.iter() {
        let Node::Inner(node) = manager.get_node(e) else {
            // SAFETY: all edges in level views refer to inner nodes
            unsafe { std::hint::unreachable_unchecked() }
        };
        // SAFETY: setting the level number re-establishes the invariant
        unsafe { node.set_level(level_no) };
    }
}
