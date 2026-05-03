//! Functionality to reorder levels in a decision diagram

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use smallvec::SmallVec;

use oxidd_core::error::OutOfMemory;
use oxidd_core::util::{AbortOnDrop, Borrowed, DropWith};
use oxidd_core::{
    DiagramRules, Edge, HasLevel, InnerNode, LevelNo, LevelView, Manager, Node, ReducedOrNew,
};

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
{
    assert!(upper_no + 1 < manager.num_levels());
    unsafe { level_swap(manager, upper_no, upper_no + 1, upper_no, upper_no + 1) };
    // SAFETY (next 2): follows from the above assertion
    update_level_no(manager, &unsafe { manager.level_unchecked(upper_no) });
    update_level_no(manager, &unsafe { manager.level_unchecked(upper_no + 1) });
}

/// Swap the levels given by `upper_no` and `lower_no`.
///
/// The upper level (designated by `upper_no`) must be above the lower level
/// (designated by `lower_no`), i.e., `upper_no < lower_no` must hold. Further,
/// there should be no level l between the upper and lower level with nodes at
/// the upper level referring to nodes at level l or nodes at level l referring
/// to nodes at the lower level.
///
/// This function does not update the level numbers inside nodes. When multiple
/// level swaps are needed to move a level to a different position, it should be
/// faster to only write the new level numbers at the very end. To allow these
/// lazy updates, we need to distinguish between the actual level numbers and
/// the level numbers inside the nodes. We suffix the latter by `_pre` as they
/// refer to the level numbers before the reordering operation started.
///
/// # Safety
///
/// - Must be called from inside the closure of
///   [`manager.reorder()`][Manager::reorder]
/// - `manager` must be derived from a `&mut M` reference (i.e., some caller had
///   exclusive access to all nodes and delegates exclusive access level-wise).
/// - `upper_no < manager.num_levels()` and `lower_no < manager.num_levels()`
///   must hold.
/// - This function may modify nodes at the level of `upper_no` and `lower_no`.
///   There must not be any concurrent modification of any nodes at these
///   levels.
/// - `upper_no_pre` must be the level number within all nodes at level
///   `upper_no`, and `lower_no_pre` the level number within all nodes at level
///   `lower_no`.
#[inline(never)]
unsafe fn level_swap<M: Manager>(
    manager: &M,
    upper_no: LevelNo,
    lower_no: LevelNo,
    upper_no_pre: LevelNo,
    lower_no_pre: LevelNo,
) where
    M::InnerNode: HasLevel,
{
    debug_assert!(upper_no < lower_no);
    debug_assert!(lower_no < manager.num_levels());
    debug_assert!(upper_no_pre < manager.num_levels());
    debug_assert!(lower_no_pre < manager.num_levels());

    // Note that the `Manager::level_unchecked()` may or may not acquire a lock.
    let mut upper = unsafe { manager.level_unchecked(upper_no) };
    let mut lower = unsafe { manager.level_unchecked(lower_no) };

    let abort_on_panic = AbortOnDrop("Some operation in `oxidd_reorder::level_down` panicked.");
    unsafe { upper.swap(&mut lower) };

    // `lower` now refers to the new lower level, `upper` to the new upper
    // level. Note that nodes on the new lower level may refer to the new upper
    // level, which is not allowed. We also fix this below.
    debug_assert!(upper
        .iter()
        .all(|e| manager.get_node(e).level() == lower_no_pre));

    let old_upper = LevelView::take(&mut lower);
    // SAFETY: reordering is in progress
    let old_upper = unsafe { old_upper.unwrap_unchecked() };
    lower.reserve(old_upper.len());
    for e in old_upper.iter() {
        let Node::Inner(node) = manager.get_node(e) else {
            // SAFETY: all edges in level views refer to inner nodes
            unsafe { std::hint::unreachable_unchecked() };
        };
        debug_assert_eq!(node.level(), upper_no_pre);

        // TODO: Maybe use `N::ARITY` once `generic_const_exprs` becomes
        // stable, and `arrayvec` instead of `smallvec`?
        let children: SmallVec<[Borrowed<M::Edge>; 2]> = node.children().collect();
        debug_assert_eq!(children.len(), M::InnerNode::ARITY);

        if children
            .iter()
            .all(|c| manager.get_node(c).level() != lower_no_pre)
        {
            // All children are below the lower level, we just move the node to
            // the lower level.
            drop(children);
            let e = manager.clone_edge(e);
            // SAFETY: the caller will update level numbers accordingly
            unsafe { lower.insert_unchecked(e) };
            continue;
        }

        // The node refers to the upper level. To fix this, we compute the
        // cofactors of the cofactors of this node, create new nodes at the
        // lower level, and keep the original node at the upper level (with the
        // children replaced by the newly created ones).

        let grandchildren: SmallVec<[_; 2]> = children
            .iter()
            .map(|c| {
                // A child of a node at the old upper level can only reference
                // a node at the old lower, i.e., the new upper level, or any
                // level below `lower_no`.
                match manager.get_node(c) {
                    Node::Inner(node) if node.level() == lower_no_pre => {
                        // We have exclusive access to the node
                        let children: SmallVec<[_; 2]> =
                            M::Rules::cofactors(c.tag(), node).collect();
                        debug_assert_eq!(children.len(), M::InnerNode::ARITY);
                        children
                    }
                    node => {
                        debug_assert!(node.level() > lower_no);
                        // The child is below the lower level, so we always have
                        // this child
                        (0..M::InnerNode::ARITY).map(|_| c.borrowed()).collect()
                    }
                }
            })
            .collect();

        let new_children: SmallVec<[_; 2]> = (0..M::InnerNode::ARITY)
            .map(|i| {
                let res = <M::Rules as DiagramRules<_, _, _>>::reduce(
                    manager,
                    upper_no_pre,
                    grandchildren.iter().map(|v| manager.clone_edge(&v[i])),
                );
                match res {
                    ReducedOrNew::Reduced(e) => e,
                    ReducedOrNew::New(node, tag) => if let Some(e) = old_upper.get(&node) {
                        node.drop_with_manager(manager);
                        manager.clone_edge(e)
                    } else {
                        // SAFETY: the caller will update level numbers accordingly. For now, all
                        // nodes at the new lower level (i.e., the old upper level) have
                        // `upper_no_pre` as their level number.
                        match unsafe { lower.get_or_insert_unchecked(node) } {
                            Ok(e) => e,
                            Err(OutOfMemory) => {
                                eprintln!("Out of memory");
                                std::process::abort();
                            }
                        }
                    }
                    .with_tag_owned(tag),
                }
            })
            .collect();

        drop(grandchildren);
        for child in children {
            // Revisit the "old" children of `e`. If these are the only
            // children, we may remove them, if they are on the old lower level.
            // (A child might also be at some lower level, in which case the
            // node could also be removed. However we must not access such a
            // node.)
            if let Node::Inner(child_node) = manager.get_node(&*child) {
                if child_node.level() == lower_no_pre && child_node.ref_count() == 1 {
                    // The reference stems from the old `node`, whose children
                    // we replace below. Hence, we can remove child node.
                    upper.remove(child_node);
                }
            }
        }

        upper.insert(manager.clone_edge(e));
        for (i, child) in new_children.into_iter().enumerate() {
            // SAFETY: we have exclusive access to all nodes at the old upper
            // level and no child is borrowed.
            manager.drop_edge(unsafe { node.set_child(i, child) });
        }
    }

    abort_on_panic.defuse();
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
