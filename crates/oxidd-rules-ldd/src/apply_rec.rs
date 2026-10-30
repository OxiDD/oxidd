//! Recursive apply algorithms for list decision diagrams.
//!
//! These are the core recursive operations on LDDs. They are generic over a
//! [`Recursor`], which determines whether independent sub-problems are solved
//! sequentially or split onto a worker pool (see [`crate::recursor`]).

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashMap;

use oxidd_core::function::{EdgeOfFunc, Function, INodeValOfFunc, OwnEdgeOfFunc};
use oxidd_core::util::{AllocResult, EdgeDropGuard, Own, Ref};
use oxidd_core::{ApplyCache, InnerNode, LevelNo, Manager, Node, NodeID};
use oxidd_derive::Function;

use crate::recursor::{Recursor, SequentialRecursor};
use crate::{
    LDDManager, LDDOp, LDDTerminal, LDDValue, RIGHT, RelationProductMeta, collect_children,
    make_node, stat,
};

/// Return a singleton LDD that contains only the list represented by `vector`.
///
/// The root resides at level 0. Panics if `vector` is longer than the number of
/// levels of `manager`.
fn singleton<M: LDDManager>(
    manager: &M,
    vector: &[M::InnerNodeValue],
) -> AllocResult<Own<M::Edge>> {
    assert!(
        vector.len() <= manager.num_levels() as usize,
        "vector of length {} exceeds the {} reserved levels",
        vector.len(),
        manager.num_levels()
    );

    let mut root = EdgeDropGuard::new(manager, manager.get_terminal(LDDTerminal::True)?);

    for (level, val) in vector.iter().enumerate().rev() {
        let empty = manager.get_terminal(LDDTerminal::Empty)?;
        root = EdgeDropGuard::new(
            manager,
            make_node(manager, level as LevelNo, val, root.into_edge(), empty)?,
        );
    }

    Ok(root.into_edge())
}

/// Recursively apply the 'union' operator to `a` and `b`, both residing at
/// `level`
fn apply_union<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    a: Ref<'_, M::Edge>,
    b: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        return apply_union(manager, SequentialRecursor, level, a, b);
    }
    stat!(call LDDOp::Union);

    // a ∪ a == a
    if a == b {
        return Ok(manager.clone_edge(a));
    }
    // ∅ ∪ b == b  and  a ∪ ∅ == a
    let (a_node, b_node) = match (manager.get_node(a), manager.get_node(b)) {
        (Node::Inner(a), Node::Inner(b)) => (a, b),
        (Node::Terminal(t), _) if *t.borrow() == LDDTerminal::Empty => {
            return Ok(manager.clone_edge(b));
        }
        (_, b_node) => {
            // It cannot be that both `a` and `b` refer to the `True` terminal
            // due to the case `a == b` above. Since `a` and `b` reside on the
            // same level, neither can refer to `True` while the other refers to
            // an inner node. So either `a` or `b` refers to the `Empty`
            // terminal.
            debug_assert!(
                b_node.is_terminal(&LDDTerminal::Empty),
                "a and b must reside on the same level"
            );
            return Ok(manager.clone_edge(a));
        }
    };

    // Query apply cache
    stat!(cache_query LDDOp::Union);
    if let Some(res) = manager.apply_cache().get(manager, LDDOp::Union, &[a, b]) {
        stat!(cache_hit LDDOp::Union);
        return Ok(res);
    }

    let a_value = a_node.value();
    let b_value = b_node.value();
    let (value, down, right) = match a_value.cmp(b_value) {
        Ordering::Less => {
            let (a_down, a_right) = collect_children(a_node);

            let right = apply_union(manager, rec, level, a_right, b)?;
            (a_value, manager.clone_edge(a_down), right)
        }
        Ordering::Greater => {
            // b carries the smaller value, which a lacks: keep b's value and
            // down-branch, and union a with b's remaining (right) values.
            let (a_down, b_right) = collect_children(b_node);

            let right = apply_union(manager, rec, level, a, b_right)?;
            (b_value, manager.clone_edge(a_down), right)
        }
        Ordering::Equal => {
            // Both carry the same value: union the down-branches (the
            // continuations) and the right-branches (the further values).
            let (a_down, a_right) = collect_children(a_node);
            let (b_down, b_right) = collect_children(b_node);

            let (low, high) = rec.binary(
                apply_union,
                manager,
                (level + 1, a_down, b_down),
                (level, a_right, b_right),
            )?;
            (b_value, low.into_edge(), high.into_edge())
        }
    };
    let result = make_node(manager, level, value, down, right)?;

    manager
        .apply_cache()
        .add(manager, LDDOp::Union, &[a, b], result.borrowed());

    Ok(result)
}

/// Computes the intersection `a ∩ b` of the two sets of vectors, both residing
/// at `level`.
fn apply_intersection<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    a: Ref<'_, M::Edge>,
    b: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        return apply_intersection(manager, SequentialRecursor, level, a, b);
    }
    stat!(call LDDOp::Intersection);

    // a ∩ a == a
    if a == b {
        return Ok(manager.clone_edge(a));
    }
    // ∅ ∩ b == ∅  and  a ∩ ∅ == ∅
    let (a_node, b_node) = match (manager.get_node(a), manager.get_node(b)) {
        (Node::Inner(a), Node::Inner(b)) => (a, b),
        (a_node, b_node) => {
            // It cannot be that both `a` and `b` refer to the `True` terminal
            // due to the case `a == b` above. Since `a` and `b` reside on the
            // same level, neither can refer to `True` while the other refers to
            // an inner node. So either `a` or `b` refers to the `Empty`
            // terminal.
            debug_assert!(
                a_node.is_terminal(&LDDTerminal::Empty) || b_node.is_terminal(&LDDTerminal::Empty),
                "a and b must reside on the same level"
            );
            return manager.get_terminal(LDDTerminal::Empty);
        }
    };

    stat!(cache_query LDDOp::Intersection);
    if let Some(res) = manager.apply_cache().get(manager, LDDOp::Intersection, &[a, b]) {
        stat!(cache_hit LDDOp::Intersection);
        return Ok(res);
    }

    let a_value = a_node.value();
    let (a_down, a_right) = collect_children(a_node);
    let b_value = b_node.value();
    let (b_down, b_right) = collect_children(b_node);

    let result = match a_value.cmp(b_value) {
        Ordering::Less => {
            // a's value is not in b; skip it.
            apply_intersection(manager, rec, level, a_right, b)?
        }
        Ordering::Greater => {
            // b's value is not in a; skip it.
            apply_intersection(manager, rec, level, a, b_right)?
        }
        Ordering::Equal => {
            let (down_result, right_result) = rec.binary(
                apply_intersection,
                manager,
                (level + 1, a_down, b_down),
                (level, a_right, b_right),
            )?;
            if manager
                .get_node(down_result.borrowed())
                .is_terminal(&LDDTerminal::Empty)
            {
                right_result.into_edge()
            } else {
                make_node(
                    manager,
                    level,
                    a_value,
                    down_result.into_edge(),
                    right_result.into_edge(),
                )?
            }
        }
    };

    manager
        .apply_cache()
        .add(manager, LDDOp::Intersection, &[a, b], result.borrowed());

    Ok(result)
}

/// Returns the largest subset of `a` that contains no element of `b`
/// (set difference `a \ b`), both residing at `level`.
fn apply_difference<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    a: Ref<'_, M::Edge>,
    b: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        return apply_difference(manager, SequentialRecursor, level, a, b);
    }
    stat!(call LDDOp::Difference);

    // a \ a == ∅
    if a == b {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    // ∅ \ b == ∅  and  a \ ∅ == a
    let (a_node, b_node) = match (manager.get_node(a), manager.get_node(b)) {
        (Node::Inner(a), Node::Inner(b)) => (a, b),
        (Node::Terminal(t), _) if *t.borrow() == LDDTerminal::Empty => {
            return manager.get_terminal(LDDTerminal::Empty);
        }
        (_, b_node) => {
            // It cannot be that both `a` and `b` refer to the `True` terminal
            // due to the case `a == b` above. Since `a` and `b` reside on the
            // same level, neither can refer to `True` while the other refers to
            // an inner node. So either `a` or `b` refers to the `Empty`
            // terminal.
            debug_assert!(
                b_node.is_terminal(&LDDTerminal::Empty),
                "a and b must reside on the same level"
            );
            return Ok(manager.clone_edge(a));
        }
    };

    stat!(cache_query LDDOp::Difference);
    if let Some(res) = manager.apply_cache().get(manager, LDDOp::Difference, &[a, b]) {
        stat!(cache_hit LDDOp::Difference);
        return Ok(res);
    }

    let a_value = a_node.value();
    let b_value = b_node.value();
    let result = match a_value.cmp(b_value) {
        Ordering::Less => {
            let (a_down, a_right) = collect_children(a_node);
            // b has no entry for this value - keep a's entry, advance a_right.
            let right_result = apply_difference(manager, rec, level, a_right, b)?;
            make_node(
                manager,
                level,
                a_value,
                manager.clone_edge(a_down),
                right_result,
            )?
        }
        Ordering::Equal => {
            let (a_down, a_right) = collect_children(a_node);
            let (b_down, b_right) = collect_children(b_node);
            let (down_result, right_result) = rec.binary(
                apply_difference,
                manager,
                (level + 1, a_down, b_down),
                (level, a_right, b_right),
            )?;
            if manager
                .get_node(down_result.borrowed())
                .is_terminal(&LDDTerminal::Empty)
            {
                right_result.into_edge()
            } else {
                make_node(
                    manager,
                    level,
                    a_value,
                    down_result.into_edge(),
                    right_result.into_edge(),
                )?
            }
        }
        Ordering::Greater => {
            // a has no entry for b's value - skip b's entry.
            apply_difference(manager, rec, level, a, b_node.child(RIGHT))?
        }
    };

    manager
        .apply_cache()
        .add(manager, LDDOp::Difference, &[a, b], result.borrowed());

    Ok(result)
}

/// Computes a meta LDD that is suitable for the [`project()`] function from the
/// given projection indices.
///
/// This function is useful to be able to cache the projection LDD instead of
/// computing it from the projection array every time.
fn compute_proj<M: LDDManager>(manager: &M, proj: &[u32]) -> AllocResult<Own<M::Edge>> {
    // Compute length of proj.
    let length = match proj.iter().max() {
        Some(x) => *x + 1,
        None => 0,
    };

    // Convert projection vectors to meta information.
    let mut result: Vec<M::InnerNodeValue> = Vec::new();
    for i in 0..length {
        // Requires worst-case quadratic time, but the projection array is
        // typically small.
        result.push(if proj.contains(&i) {
            M::InnerNodeValue::true_value()
        } else {
            M::InnerNodeValue::false_value()
        });
    }

    singleton(manager, &result)
}

/// Computes the set of vectors projected onto the given indices, where proj is
/// equal to `compute_proj([i_0, ..., i_k])`.
///
/// Formally, for a single vector `<x_0, ..., x_n>` we have that:
///
/// - `project(<x_0, ..., x_n>, i_0 < ... < i_k) = <x_(i_0), ..., x_(i_k)>`
/// - `project(X, i_0 < ... < i_k) = { project(x, i_0 < ... < i_k) | x in X }`.
///
/// Note that the indices are sorted in the definition, but compute_proj
/// can take any array and ignores both duplicates and order. Also, it
/// follows that i_k must be smaller than or equal to n as x_(i_k) is not
/// defined otherwise.
///
/// The result resides at `level`, the number of projected positions above it.
/// A projection yields new vectors, so the top-level call starts at level 0.
fn apply_project<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    set: Ref<'_, M::Edge>,
    proj: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        return apply_project(manager, SequentialRecursor, level, set, proj);
    }
    stat!(call LDDOp::Project);

    // Base case: if proj has reached the True terminal, the projection is
    // fully consumed - return the True terminal (empty vector).
    let proj_node = match manager.get_node(proj) {
        Node::Terminal(terminal) => {
            debug_assert_eq!(
                *terminal.borrow(),
                LDDTerminal::True,
                "proj should never be the Empty terminal"
            );
            return manager.get_terminal(LDDTerminal::True);
        }
        Node::Inner(proj_node) => proj_node,
    };

    // Base case: if set is the Empty terminal, the result is the empty set.
    let set_node = match manager.get_node(set) {
        Node::Terminal(terminal) => {
            debug_assert_eq!(
                *terminal.borrow(),
                LDDTerminal::Empty,
                "proj can be at most as high as set"
            );
            return manager.get_terminal(LDDTerminal::Empty);
        }
        Node::Inner(set_node) => set_node,
    };

    // Projection meta-LDDs with different prefixes may share a suffix, so the
    // operands do not determine the result level; it is part of the key.
    stat!(cache_query LDDOp::Project);
    if let Some(([res], [])) =
        manager
            .apply_cache()
            .get_extended(manager, LDDOp::Project, (&[set, proj], &[level]))
    {
        stat!(cache_hit LDDOp::Project);
        return Ok(res);
    }

    let proj_value = proj_node.value();
    let (proj_down, _proj_right) = collect_children(proj_node);
    let set_value = set_node.value();
    let (set_down, set_right) = collect_children(set_node);

    let result = if *proj_value == M::InnerNodeValue::false_value() {
        // This position is not in the projection: skip it and compute the union
        // of the right and down branches.
        let (right_result, down_result) = rec.binary(
            apply_project,
            manager,
            (level, set_right, proj),
            (level, set_down, proj_down),
        )?;
        apply_union(
            manager,
            rec,
            level,
            right_result.borrowed(),
            down_result.borrowed(),
        )
    } else if *proj_value == M::InnerNodeValue::true_value() {
        // This position is in the projection: keep the current value and
        // recurse on both branches.
        let (right_result, down_result) = rec.binary(
            apply_project,
            manager,
            (level, set_right, proj),
            (level + 1, set_down, proj_down),
        )?;

        if manager
            .get_node(down_result.borrowed())
            .is_terminal(&LDDTerminal::Empty)
        {
            // The down sub-result is empty - nothing to insert, return right
            // only.
            Ok(right_result.into_edge())
        } else {
            make_node(
                manager,
                level,
                set_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )
        }
    } else {
        panic!("proj has unexpected value");
    }?;

    manager.apply_cache().add_extended(
        manager,
        LDDOp::Project,
        (&[set, proj], &[level]),
        (&[result.borrowed()], &[]),
    );

    Ok(result)
}

/// Builds the meta-LDD encoding the read/write projection described by
/// `read_proj` and `write_proj`, together with the positions of the read and
/// write variables in that encoding.
///
/// Shared implementation behind [`LDDFunction::relation_product_meta`] and
/// [`mt::LDDFunctionMT::relation_product_meta`].
fn relation_product_meta<M: LDDManager>(
    manager: &M,
    read_proj: &[u32],
    write_proj: &[u32],
) -> AllocResult<RelationProductMeta<Own<M::Edge>>> {
    // Compute length of meta.
    let length = std::cmp::max(
        read_proj.iter().max().map_or(0, |x| *x + 1),
        write_proj.iter().max().map_or(0, |x| *x + 1),
    );

    // Convert projection vectors to meta.
    let mut read_positions = Vec::new();
    let mut write_positions = Vec::new();

    let mut meta: Vec<M::InnerNodeValue> = Vec::new();

    let mut offset: usize = 0;
    for i in 0..length {
        // Requires worst-case quadratic time, but the projection arrays are
        // typically small.
        let read = read_proj.contains(&i);
        let write = write_proj.contains(&i);
        if read && write {
            meta.push(M::InnerNodeValue::read_of_pair_value());
            meta.push(M::InnerNodeValue::write_of_pair_value());
            read_positions.push(offset);
            write_positions.push(offset + 1);
            offset += 2;
        } else if read {
            meta.push(M::InnerNodeValue::read_only_value());
            read_positions.push(offset);
            offset += 1;
        } else if write {
            meta.push(M::InnerNodeValue::write_only_value());
            write_positions.push(offset);
            offset += 1;
        } else {
            meta.push(M::InnerNodeValue::false_value());
        }
    }

    Ok(RelationProductMeta {
        meta: singleton(manager, &meta)?,
        read_positions,
        write_positions,
    })
}

/// Computes the set of vectors reachable in one step from `set` via the sparse
/// relation `rel`, guided by `meta` (produced by
/// [`LDDFunction::relation_product_meta`]).
///
/// Meta values at each level:
/// - 0 - position not in the relation: keep set values, advance meta only.
/// - 1 - read-only: match set and rel values; keep matched values in output.
/// - 2 - write-only: union all set values, then write each rel value.
/// - 3 - read half of a read+write pair: match values (not emitted); meta_down
///   is 4.
/// - 4 - write half of a read+write pair: write rel values.
///
/// The result resides at `level`. This equals the level of `set`, except at a
/// write half of a pair, where `set` was already descended at the read half.
///
/// # Cache
///
/// Therefore, `level` is part of the cache key. Usually `set` determines the
/// result level, since every inner node resides in the unique table of one
/// level. However, at a write half where the read half consumed the last
/// position of `set`, `set` is the `True` terminal. Then only `rel` and `meta`
/// remain, and they reside on meta levels, not result levels. A read+write pair
/// occupies two meta levels but one result level, so the same meta level
/// corresponds to different result levels depending on the number of pairs
/// before it.
///
/// # Example
///
/// The meta-LDDs `[3, 4, 3, 4]` and `[0, 0, 3, 4]` share the suffix `[4]` at
/// meta level 3, which is at result level 1 for sets of length 2 and at result
/// level 2 for sets of length 3. If `rel` also shares its suffix, both calls
/// have the operands `(True, rel, meta)`, and without `level` in the key a
/// cache hit would return a node from the wrong level. This cannot happen if
/// all sets have the same length `n`, since then `set` is only `True` at a
/// write half for result level `n - 1`.
fn apply_relational_product<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    set: Ref<'_, M::Edge>,
    rel: Ref<'_, M::Edge>,
    meta: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        return apply_relational_product(manager, SequentialRecursor, level, set, rel, meta);
    }
    stat!(call LDDOp::RelationalProduct);

    // meta == True means all meta levels consumed; return set unchanged.
    let meta_node = match manager.get_node(meta) {
        Node::Terminal(t) => {
            debug_assert_eq!(
                *t.borrow(),
                LDDTerminal::True,
                "meta should never reach the Empty terminal"
            );
            return Ok(manager.clone_edge(set));
        }
        Node::Inner(n) => n,
    };

    // Empty set or empty relation → empty result. Set can still be the `True`
    // terminal at a write half.
    let (set_node, rel_node) = match (manager.get_node(set), manager.get_node(rel)) {
        (Node::Inner(s), Node::Inner(r)) => (Some(s), r),
        (Node::Terminal(s), Node::Inner(r)) if *s.borrow() == LDDTerminal::True => (None, r),
        (s, r) => {
            debug_assert!(
                s.is_terminal(&LDDTerminal::Empty) || r.is_terminal(&LDDTerminal::Empty),
                "rel must have as many levels as meta"
            );
            return manager.get_terminal(LDDTerminal::Empty);
        }
    };

    // At a write half of a pair, `set` may already be the `True` terminal, so
    // the operands do not determine the result level; it is part of the key
    // (see the function documentation).
    stat!(cache_query LDDOp::RelationalProduct);
    if let Some(([res], [])) = manager.apply_cache().get_extended(
        manager,
        LDDOp::RelationalProduct,
        (&[set, rel, meta], &[level]),
    ) {
        stat!(cache_hit LDDOp::RelationalProduct);
        return Ok(res);
    }

    let meta_value = meta_node.value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::false_value() {
        // 0: not in relation - keep all set values, advance meta into next
        // level.
        let set_node = set_node.expect("set must have as many levels as meta");
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);

        let (down_result, right_result) = rec.ternary(
            apply_relational_product,
            manager,
            (level + 1, set_down, rel, meta_down),
            (level, set_right, rel, meta),
        )?;

        if manager
            .get_node(down_result.borrowed())
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                level,
                set_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else if *meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only - match set and rel values; keep matched values in
        // output.
        let set_node = set_node.expect("set must have as many levels as meta");
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // No rel entry for this set value; skip it.
                apply_relational_product(manager, rec, level, set_right, rel, meta)?
            }
            Ordering::Equal => {
                let (down_result, right_result) = rec.ternary(
                    apply_relational_product,
                    manager,
                    (level + 1, set_down, rel_down, meta_down),
                    (level, set_right, rel_right, meta),
                )?;
                if manager
                    .get_node(down_result.borrowed())
                    .is_terminal(&LDDTerminal::Empty)
                {
                    right_result.into_edge()
                } else {
                    make_node(
                        manager,
                        level,
                        set_value,
                        down_result.into_edge(),
                        right_result.into_edge(),
                    )?
                }
            }
            Ordering::Greater => {
                // No set entry for this rel value; skip the rel value.
                apply_relational_product(manager, rec, level, set, rel_right, meta)?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only - union all set down-branches (ignoring their values),
        // then write each rel value with that combined continuation.
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        // Collect union of all down-branches at this set level.
        let combined = {
            let mut acc = EdgeDropGuard::new(manager, manager.get_terminal(LDDTerminal::Empty)?);
            let mut cur = set;
            loop {
                let cur_node = manager.get_node(cur).unwrap_inner();
                let (cur_down, cur_right) = collect_children(cur_node);

                let new_acc = EdgeDropGuard::new(
                    manager,
                    apply_union(manager, rec, level + 1, acc.borrowed(), cur_down)?,
                );
                acc = new_acc;
                if manager.get_node(cur_right).is_terminal(&LDDTerminal::Empty) {
                    break;
                }
                cur = cur_right;
            }
            acc
        };

        let (down_result, right_result) = rec.ternary(
            apply_relational_product,
            manager,
            (level + 1, combined.borrowed(), rel_down, meta_down),
            (level, set, rel_right, meta),
        )?;

        if manager
            .get_node(down_result.borrowed())
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                level,
                rel_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair - match values (not emitted);
        // union the matched continuation with the remaining-pairs result.
        let set_node = set_node.expect("set must have as many levels as meta");
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => apply_relational_product(manager, rec, level, set_right, rel, meta)?,
            Ordering::Equal => {
                // meta_down should be write_of_pair_value (4); it introduces
                // the new values at this output level.  Union with the
                // right-siblings result so all (read, write) pairs at this
                // level are considered.
                let (down_result, right_result) = rec.ternary(
                    apply_relational_product,
                    manager,
                    (level, set_down, rel_down, meta_down),
                    (level, set_right, rel_right, meta),
                )?;
                apply_union(
                    manager,
                    rec,
                    level,
                    down_result.borrowed(),
                    right_result.borrowed(),
                )?
            }
            Ordering::Greater => {
                apply_relational_product(manager, rec, level, set, rel_right, meta)?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_of_pair_value() {
        // 4: write half of a read+write pair - emit rel values.
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        let (down_result, right_result) = rec.ternary(
            apply_relational_product,
            manager,
            (level + 1, set, rel_down, meta_down),
            (level, set, rel_right, meta),
        )?;

        if manager
            .get_node(down_result.borrowed())
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                level,
                rel_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else {
        panic!("meta has unexpected value");
    };

    manager.apply_cache().add_extended(
        manager,
        LDDOp::RelationalProduct,
        (&[set, rel, meta], &[level]),
        (&[result.borrowed()], &[]),
    );

    Ok(result)
}

/// Computes the set of source vectors in `universe` that can reach a vector in
/// `set` in one step via the sparse relation `rel`, guided by `meta` (produced
/// by [`LDDFunction::relation_product_meta`]).
///
/// This is the inverse of [`apply_relational_product`]: where the relational
/// product computes successors, this computes predecessors restricted to
/// `universe`.
///
/// Meta values at each level (see [`apply_relational_product`]):
/// - 0 - position not in the relation: keep values present in both `set` and
///   `universe`, advance meta only.
/// - 1 - read-only: the source value equals the target value; keep values that
///   appear in `set`, `rel`, and `universe`.
/// - 2 - write-only: the source value is unconstrained, ranging over all of
///   `universe`; the target value must appear in `set` and `rel`.
/// - 3 - read half of a read+write pair: emit the source value taken from
///   `universe` (matched against `rel`).
/// - 4 - write half of a read+write pair: consume the target value (matched
///   against `set` and `rel`); no value is emitted.
///
/// The result resides at `level`. This equals the level of `universe`, except
/// at a write half of a pair, where the source value was already emitted at the
/// read half. The operands thus determine the level, so it is not part of the
/// cache key.
fn apply_relational_predecessor<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    set: Ref<'_, M::Edge>,
    rel: Ref<'_, M::Edge>,
    meta: Ref<'_, M::Edge>,
    universe: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        let rec = SequentialRecursor;
        return apply_relational_predecessor(manager, rec, level, set, rel, meta, universe);
    }
    stat!(call LDDOp::RelationalPredecessor);

    // meta == True means all meta levels are consumed; the remaining source
    // vectors are exactly those in both `set` and `universe`, unless `rel` is
    // empty.
    let meta_node = match manager.get_node(meta) {
        Node::Terminal(t) => {
            debug_assert_eq!(
                *t.borrow(),
                LDDTerminal::True,
                "meta should never reach the Empty terminal"
            );
            if manager.get_node(rel).is_terminal(&LDDTerminal::Empty) {
                return manager.get_terminal(LDDTerminal::Empty);
            }
            return apply_intersection(manager, rec, level, set, universe);
        }
        Node::Inner(n) => n,
    };

    // An empty set, relation, or universe yields the empty set. Since `meta`
    // ends with a non-zero value, none of them can be the `True` terminal.
    let (set_node, rel_node, universe_node) = match (
        manager.get_node(set),
        manager.get_node(rel),
        manager.get_node(universe),
    ) {
        (Node::Inner(s), Node::Inner(r), Node::Inner(u)) => (s, r, u),
        (s, r, u) => {
            debug_assert!(
                s.is_terminal(&LDDTerminal::Empty)
                    || r.is_terminal(&LDDTerminal::Empty)
                    || u.is_terminal(&LDDTerminal::Empty),
                "set, rel and universe must have as many levels as meta"
            );
            return manager.get_terminal(LDDTerminal::Empty);
        }
    };

    stat!(cache_query LDDOp::RelationalPredecessor);
    if let Some(res) = manager.apply_cache().get(
        manager,
        LDDOp::RelationalPredecessor,
        &[set, rel, meta, universe],
    ) {
        stat!(cache_hit LDDOp::RelationalPredecessor);
        return Ok(res);
    }

    let meta_value = meta_node.value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::false_value() {
        // 0: not in relation - keep values present in both set and universe.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let universe_value = universe_node.value();
        let (universe_down, universe_right) = collect_children(universe_node);

        match set_value.cmp(universe_value) {
            Ordering::Less => {
                // This set value is not in the universe; skip it.
                apply_relational_predecessor(manager, rec, level, set_right, rel, meta, universe)?
            }
            Ordering::Greater => {
                // This universe value is not in the set; skip it.
                apply_relational_predecessor(manager, rec, level, set, rel, meta, universe_right)?
            }
            Ordering::Equal => {
                let (down_result, right_result) = rec.quaternary(
                    apply_relational_predecessor,
                    manager,
                    (level + 1, set_down, rel, meta_down, universe_down),
                    (level, set_right, rel, meta, universe_right),
                )?;
                if manager
                    .get_node(down_result.borrowed())
                    .is_terminal(&LDDTerminal::Empty)
                {
                    right_result.into_edge()
                } else {
                    make_node(
                        manager,
                        level,
                        set_value,
                        down_result.into_edge(),
                        right_result.into_edge(),
                    )?
                }
            }
        }
    } else if *meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only - the source value equals the target value, so emit
        // values that appear in set, rel, and universe simultaneously.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);
        let universe_value = universe_node.value();
        let (universe_down, universe_right) = collect_children(universe_node);

        if set_value == rel_value && rel_value == universe_value {
            // All three agree on the value: emit it and recurse downwards on
            // all of set, rel, and universe; the right siblings are explored by
            // advancing the relation.
            let (down_result, rest) = rec.quaternary(
                apply_relational_predecessor,
                manager,
                (level + 1, set_down, rel_down, meta_down, universe_down),
                (level, set, rel_right, meta, universe),
            )?;
            let node = EdgeDropGuard::new(
                manager,
                if manager
                    .get_node(down_result.borrowed())
                    .is_terminal(&LDDTerminal::Empty)
                {
                    manager.get_terminal(LDDTerminal::Empty)?
                } else {
                    let empty = manager.get_terminal(LDDTerminal::Empty)?;
                    make_node(
                        manager,
                        level,
                        universe_value,
                        down_result.into_edge(),
                        empty,
                    )?
                },
            );
            apply_union(manager, rec, level, node.borrowed(), rest.borrowed())?
        } else {
            // Advance whichever operands are below the maximum value so that
            // all three line up on a common value.
            let max = set_value.max(rel_value).max(universe_value);
            if set_value < max {
                apply_relational_predecessor(manager, rec, level, set_right, rel, meta, universe)?
            } else if universe_value < max {
                apply_relational_predecessor(manager, rec, level, set, rel, meta, universe_right)?
            } else {
                // rel_value < max
                apply_relational_predecessor(manager, rec, level, set, rel_right, meta, universe)?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only - the target value must appear in both set and rel; the
        // source value is unconstrained and ranges over the whole universe.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // This set value is not written by the relation; skip it.
                apply_relational_predecessor(manager, rec, level, set_right, rel, meta, universe)?
            }
            Ordering::Greater => {
                // This relation value is not in the set; skip it.
                apply_relational_predecessor(manager, rec, level, set, rel_right, meta, universe)?
            }
            Ordering::Equal => {
                // Emit a source node for every value in the universe, all
                // sharing the matched continuation.
                let down_result = EdgeDropGuard::new(
                    manager,
                    relational_predecessor_universe(
                        manager, rec, level, set_down, rel_down, meta_down, universe,
                    )?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_predecessor(
                        manager, rec, level, set, rel_right, meta, universe,
                    )?,
                );
                apply_union(
                    manager,
                    rec,
                    level,
                    down_result.borrowed(),
                    right_result.borrowed(),
                )?
            }
        }
    } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair - emit the source value, taken from
        // the universe and matched against the relation. The set and universe
        // are not descended here; the universe is descended at the paired write
        // level.
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);
        let universe_value = universe_node.value();
        let (_universe_down, universe_right) = collect_children(universe_node);

        match universe_value.cmp(rel_value) {
            Ordering::Less => {
                // This universe value cannot be read; skip it.
                apply_relational_predecessor(manager, rec, level, set, rel, meta, universe_right)?
            }
            Ordering::Greater => {
                // This relation value is not in the universe; skip it.
                apply_relational_predecessor(manager, rec, level, set, rel_right, meta, universe)?
            }
            Ordering::Equal => {
                // meta_down is the paired write level (4); the universe is kept
                // so that its down-branch is descended there.
                let (down_result, rest) = rec.quaternary(
                    apply_relational_predecessor,
                    manager,
                    (level + 1, set, rel_down, meta_down, universe),
                    (level, set, rel_right, meta, universe),
                )?;
                let node = EdgeDropGuard::new(
                    manager,
                    if manager
                        .get_node(down_result.borrowed())
                        .is_terminal(&LDDTerminal::Empty)
                    {
                        manager.get_terminal(LDDTerminal::Empty)?
                    } else {
                        let empty = manager.get_terminal(LDDTerminal::Empty)?;
                        make_node(
                            manager,
                            level,
                            universe_value,
                            down_result.into_edge(),
                            empty,
                        )?
                    },
                );
                apply_union(manager, rec, level, node.borrowed(), rest.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_of_pair_value() {
        // 4: write half of a read+write pair - consume the target value, which
        // must appear in both set and rel; nothing is emitted. The source value
        // was already produced at the paired read level, so descend the
        // universe here.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // This target value is not written by the relation; skip it.
                apply_relational_predecessor(manager, rec, level, set_right, rel, meta, universe)?
            }
            Ordering::Greater => {
                // This relation value is not in the set; skip it.
                apply_relational_predecessor(manager, rec, level, set, rel_right, meta, universe)?
            }
            Ordering::Equal => {
                let (universe_down, _universe_right) = collect_children(universe_node);
                let (down_result, right_result) = rec.quaternary(
                    apply_relational_predecessor,
                    manager,
                    (level, set_down, rel_down, meta_down, universe_down),
                    (level, set, rel_right, meta, universe),
                )?;
                apply_union(
                    manager,
                    rec,
                    level,
                    down_result.borrowed(),
                    right_result.borrowed(),
                )?
            }
        }
    } else {
        panic!("meta has unexpected value");
    };

    manager.apply_cache().add(
        manager,
        LDDOp::RelationalPredecessor,
        &[set, rel, meta, universe],
        result.borrowed(),
    );

    Ok(result)
}

/// Helper for the write-only ([`write_only_value`][LDDValue::write_only_value])
/// case of [`apply_relational_predecessor`].
///
/// Builds the union over every value `v` in `universe` of the source node
/// `(v, apply_relational_predecessor(set, rel, meta, universe_v_down))`, where
/// `universe_v_down` is the down-branch of the universe node carrying value
/// `v`. The `set`, `rel`, and `meta` arguments are already descended past the
/// write level.
fn relational_predecessor_universe<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    set: Ref<'_, M::Edge>,
    rel: Ref<'_, M::Edge>,
    meta: Ref<'_, M::Edge>,
    universe: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    let universe_node = match manager.get_node(universe) {
        // The right spine of the universe ends at the Empty terminal.
        Node::Terminal(_) => return manager.get_terminal(LDDTerminal::Empty),
        Node::Inner(n) => n,
    };
    let universe_value = universe_node.value();
    let (universe_down, universe_right) = collect_children(universe_node);

    let down_result = EdgeDropGuard::new(
        manager,
        apply_relational_predecessor(manager, rec, level + 1, set, rel, meta, universe_down)?,
    );
    let right_result = EdgeDropGuard::new(
        manager,
        relational_predecessor_universe(manager, rec, level, set, rel, meta, universe_right)?,
    );

    if manager
        .get_node(down_result.borrowed())
        .is_terminal(&LDDTerminal::Empty)
    {
        Ok(right_result.into_edge())
    } else {
        make_node(
            manager,
            level,
            universe_value,
            down_result.into_edge(),
            right_result.into_edge(),
        )
    }
}

/// Returns the number of vectors (lists) contained in the set `set`.
///
/// The empty set (the `Empty` terminal) contains no vectors; the set
/// containing only the empty vector (the `True` terminal) contains exactly
/// one. The counts of inner nodes are memoized in `cache` (keyed by
/// [`NodeID`]) so that shared sub-diagrams are only counted once.
fn len<M: LDDManager>(
    manager: &M,
    set: Ref<'_, M::Edge>,
    cache: &mut HashMap<NodeID, usize>,
) -> usize {
    let set_node = match manager.get_node(set) {
        Node::Terminal(t) => return *t.borrow() as usize,
        Node::Inner(n) => n,
    };

    let node_id = set.node_id();
    if let Some(n) = cache.get(&node_id) {
        return *n;
    }

    // Walk the right spine, summing the sizes of all down-branches. The right
    // spine is terminated by the Empty terminal.
    let mut result = 0;
    let mut current = set_node;
    loop {
        let (down, right) = collect_children(current);
        result += len(manager, down, cache);
        match manager.get_node(right) {
            Node::Inner(n) => current = n,
            Node::Terminal(t) => {
                debug_assert!(*t.borrow() == LDDTerminal::Empty);
                break;
            }
        }
    }

    cache.insert(node_id, result);
    result
}

/// Function backed by a list decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr_id = "LDD"]
#[repr(transparent)]
pub struct LDDFunction<F: Function>(F);

impl<F: Function> From<F> for LDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        LDDFunction(value)
    }
}

impl<F: Function> LDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

#[allow(private_bounds)]
impl<F: Function> LDDFunction<F>
where
    for<'id> F::Manager<'id>: LDDManager,
{
    #[inline]
    pub fn empty_set<'id>(manager: &<Self as Function>::Manager<'id>) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            manager.get_terminal(LDDTerminal::Empty)?,
        ))
    }

    #[inline]
    pub fn empty_vector<'id>(manager: &<Self as Function>::Manager<'id>) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            manager.get_terminal(LDDTerminal::True)?,
        ))
    }

    /// Returns an LDD containing only the given vector, i.e., { vector }.
    #[inline]
    pub fn singleton<'id>(
        manager: &<Self as Function>::Manager<'id>,
        vector: &[INodeValOfFunc<'id, Self>],
    ) -> AllocResult<Self> {
        Ok(Self::from_edge(manager, singleton(manager, vector)?))
    }

    #[inline]
    pub fn singleton_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        vector: &[INodeValOfFunc<'id, Self>],
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        singleton(manager, vector)
    }

    /// Compute the union `self ∪ other`.
    pub fn union(&self, other: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, a| {
            let edge = Self::union_edge(manager, 0, a, other.as_edge(manager))?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Compute the intersection `self ∩ other`.
    pub fn intsec(&self, other: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, a| {
            let edge = Self::intsec_edge(manager, 0, a, other.as_edge(manager))?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Compute the set difference `self \ other`.
    pub fn diff(&self, other: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, a| {
            let edge = Self::diff_edge(manager, 0, a, other.as_edge(manager))?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Compute the union `a ∪ b`
    ///
    /// `a` and `b` must reside at `level`, i.e., their roots are nodes for
    /// position `level` of the vectors (0 for an LDD that is not a sub-LDD).
    #[inline]
    pub fn union_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        a: Ref<'_, EdgeOfFunc<'id, Self>>,
        b: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_union(manager, SequentialRecursor, level, a, b)
    }

    /// Compute the intersection `a ∩ b`
    ///
    /// `a` and `b` must reside at `level`, i.e., their roots are nodes for
    /// position `level` of the vectors (0 for an LDD that is not a sub-LDD).
    #[inline]
    pub fn intsec_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        a: Ref<'_, EdgeOfFunc<'id, Self>>,
        b: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_intersection(manager, SequentialRecursor, level, a, b)
    }

    /// Compute the set difference `a \ b`
    ///
    /// `a` and `b` must reside at `level`, i.e., their roots are nodes for
    /// position `level` of the vectors (0 for an LDD that is not a sub-LDD).
    #[inline]
    pub fn diff_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        a: Ref<'_, EdgeOfFunc<'id, Self>>,
        b: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_difference(manager, SequentialRecursor, level, a, b)
    }

    /// Computes a meta-LDD encoding the projection onto the indices in `proj`,
    /// suitable as the second argument of [`project_edge`][Self::project_edge].
    #[inline]
    pub fn projection_meta<'id>(
        manager: &<Self as Function>::Manager<'id>,
        proj: &[u32],
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        compute_proj(manager, proj)
    }

    /// Projects the vectors in `self` onto the indices encoded by `proj`
    /// (produced by [`projection_meta`][Self::projection_meta]).
    pub fn project(&self, proj: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let edge = Self::project_edge(manager, set, proj.as_edge(manager))?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Projects the vectors in `set` onto the indices encoded by `proj` (built
    /// via [`projection_meta`][Self::projection_meta]).
    ///
    /// The projected vectors are new vectors, so the result resides at level 0.
    #[inline]
    pub fn project_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        set: Ref<'_, EdgeOfFunc<'id, Self>>,
        proj: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_project(manager, SequentialRecursor, 0, set, proj)
    }

    #[inline]
    pub fn relation_product_meta<'id>(
        manager: &<Self as Function>::Manager<'id>,
        read_proj: &[u32],
        write_proj: &[u32],
    ) -> AllocResult<RelationProductMeta<OwnEdgeOfFunc<'id, Self>>> {
        relation_product_meta(manager, read_proj, write_proj)
    }

    /// Computes the set of vectors reachable in one step from `self` via the
    /// sparse relation `rel`, guided by `meta` (produced by
    /// [`relation_product_meta`][Self::relation_product_meta]).
    pub fn relational_product(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let edge = Self::relational_product_edge(
                manager,
                0,
                set,
                rel.as_edge(manager),
                meta.as_edge(manager),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set of source vectors in `universe` from which a vector in
    /// `self` is reachable in one step via the sparse relation `rel`, guided by
    /// `meta` (produced by
    /// [`relation_product_meta`][Self::relation_product_meta]).
    ///
    /// This is the inverse of
    /// [`relational_product`][Self::relational_product].
    pub fn relational_predecessor(
        &self,
        rel: &Self,
        meta: &Self,
        universe: &Self,
    ) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let edge = Self::relational_predecessor_edge(
                manager,
                0,
                set,
                rel.as_edge(manager),
                meta.as_edge(manager),
                universe.as_edge(manager),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set of vectors reachable in one step from `set` via the
    /// sparse relation `rel`.  `meta` must be produced by
    /// [`relation_product_meta`][Self::relation_product_meta].
    ///
    /// `set` must reside at `level`; `rel` and `meta` are sub-LDDs matching the
    /// positions from `level` onwards.
    #[inline]
    pub fn relational_product_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        set: Ref<'_, EdgeOfFunc<'id, Self>>,
        rel: Ref<'_, EdgeOfFunc<'id, Self>>,
        meta: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_relational_product(manager, SequentialRecursor, level, set, rel, meta)
    }

    /// Computes the set of source vectors in `universe` that can reach a vector
    /// in `set` in one step via the sparse relation `rel`.  `meta` must be
    /// produced by [`relation_product_meta`][Self::relation_product_meta].
    ///
    /// This is the inverse of [`Self::relational_product_edge`].
    ///
    /// `set` and `universe` must reside at `level`.
    #[inline]
    pub fn relational_predecessor_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        set: Ref<'_, EdgeOfFunc<'id, Self>>,
        rel: Ref<'_, EdgeOfFunc<'id, Self>>,
        meta: Ref<'_, EdgeOfFunc<'id, Self>>,
        universe: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_relational_predecessor(manager, SequentialRecursor, level, set, rel, meta, universe)
    }

    /// Returns the number of vectors (lists) contained in `self`.
    pub fn len(&self) -> usize {
        self.with_manager_shared(|manager, set| Self::len_edge(manager, set))
    }

    /// Returns the number of vectors (lists) contained in the set rooted at
    /// `set`.
    #[inline]
    pub fn len_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        set: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> usize {
        len(manager, set, &mut HashMap::new())
    }

    /// Returns `true` if `self` is the empty set `∅`, i.e. contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.with_manager_shared(|manager, edge| {
            manager.get_node(edge).is_terminal(&LDDTerminal::Empty)
        })
    }
}

/// Multi-threaded list decision diagrams
#[cfg(feature = "multi-threading")]
pub mod mt {
    use oxidd_core::HasWorkers;

    use crate::recursor::mt::ParallelRecursor;

    use super::*;

    /// Set of vectors backed by a list decision diagram, multi-threaded version
    ///
    /// This is the parallel counterpart of [`LDDFunction`]: the apply
    /// algorithms split independent sub-problems onto the manager's worker
    /// pool. All other operations behave exactly like the sequential
    /// [`LDDFunction`].
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
    #[repr_id = "LDD"]
    #[repr(transparent)]
    pub struct LDDFunctionMT<F: Function>(F);

    impl<F: Function> From<F> for LDDFunctionMT<F> {
        #[inline(always)]
        fn from(value: F) -> Self {
            LDDFunctionMT(value)
        }
    }

    impl<F: Function> LDDFunctionMT<F> {
        /// Convert `self` into the underlying [`Function`]
        #[inline(always)]
        pub fn into_inner(self) -> F {
            self.0
        }
    }

    #[allow(private_bounds)]
    impl<F: Function> LDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: LDDManager<Edge: Send + Sync> + HasWorkers,
    {
        /// See [`LDDFunction::empty_set`].
        #[inline]
        pub fn empty_set<'id>(manager: &<Self as Function>::Manager<'id>) -> AllocResult<Self> {
            Ok(Self::from_edge(
                manager,
                manager.get_terminal(LDDTerminal::Empty)?,
            ))
        }

        /// See [`LDDFunction::empty_vector`].
        #[inline]
        pub fn empty_vector<'id>(manager: &<Self as Function>::Manager<'id>) -> AllocResult<Self> {
            Ok(Self::from_edge(
                manager,
                manager.get_terminal(LDDTerminal::True)?,
            ))
        }

        /// See [`LDDFunction::singleton`].
        #[inline]
        pub fn singleton<'id>(
            manager: &<Self as Function>::Manager<'id>,
            vector: &[INodeValOfFunc<'id, Self>],
        ) -> AllocResult<Self> {
            Ok(Self::from_edge(manager, singleton(manager, vector)?))
        }

        /// See [`LDDFunction::singleton_edge`].
        #[inline]
        pub fn singleton_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            vector: &[INodeValOfFunc<'id, Self>],
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            singleton(manager, vector)
        }

        /// See [`LDDFunction::union`].
        pub fn union(&self, other: &Self) -> AllocResult<Self> {
            self.with_manager_shared(|manager, a| {
                let edge = Self::union_edge(manager, 0, a, other.as_edge(manager))?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::intsec`].
        pub fn intsec(&self, other: &Self) -> AllocResult<Self> {
            self.with_manager_shared(|manager, a| {
                let edge = Self::intsec_edge(manager, 0, a, other.as_edge(manager))?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::diff`].
        pub fn diff(&self, other: &Self) -> AllocResult<Self> {
            self.with_manager_shared(|manager, a| {
                let edge = Self::diff_edge(manager, 0, a, other.as_edge(manager))?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::union_edge`].
        #[inline]
        pub fn union_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            a: Ref<'_, EdgeOfFunc<'id, Self>>,
            b: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            apply_union(manager, ParallelRecursor::new(manager), level, a, b)
        }

        /// See [`LDDFunction::intsec_edge`].
        #[inline]
        pub fn intsec_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            a: Ref<'_, EdgeOfFunc<'id, Self>>,
            b: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            apply_intersection(manager, ParallelRecursor::new(manager), level, a, b)
        }

        /// See [`LDDFunction::diff_edge`].
        #[inline]
        pub fn diff_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            a: Ref<'_, EdgeOfFunc<'id, Self>>,
            b: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            apply_difference(manager, ParallelRecursor::new(manager), level, a, b)
        }

        /// See [`LDDFunction::projection_meta`].
        #[inline]
        pub fn projection_meta<'id>(
            manager: &<Self as Function>::Manager<'id>,
            proj: &[u32],
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            compute_proj(manager, proj)
        }

        /// See [`LDDFunction::project`].
        pub fn project(&self, proj: &Self) -> AllocResult<Self> {
            self.with_manager_shared(|manager, set| {
                let edge = Self::project_edge(manager, set, proj.as_edge(manager))?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::project_edge`].
        #[inline]
        pub fn project_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            set: Ref<'_, EdgeOfFunc<'id, Self>>,
            proj: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            apply_project(manager, ParallelRecursor::new(manager), 0, set, proj)
        }

        /// See [`LDDFunction::relation_product_meta`].
        #[inline]
        pub fn relation_product_meta<'id>(
            manager: &<Self as Function>::Manager<'id>,
            read_proj: &[u32],
            write_proj: &[u32],
        ) -> AllocResult<RelationProductMeta<OwnEdgeOfFunc<'id, Self>>> {
            relation_product_meta(manager, read_proj, write_proj)
        }

        /// See [`LDDFunction::relational_product`].
        pub fn relational_product(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
            self.with_manager_shared(|manager, set| {
                let edge = Self::relational_product_edge(
                    manager,
                    0,
                    set,
                    rel.as_edge(manager),
                    meta.as_edge(manager),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::relational_predecessor`].
        pub fn relational_predecessor(
            &self,
            rel: &Self,
            meta: &Self,
            universe: &Self,
        ) -> AllocResult<Self> {
            self.with_manager_shared(|manager, set| {
                let edge = Self::relational_predecessor_edge(
                    manager,
                    0,
                    set,
                    rel.as_edge(manager),
                    meta.as_edge(manager),
                    universe.as_edge(manager),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::relational_product_edge`].
        #[inline]
        pub fn relational_product_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            set: Ref<'_, EdgeOfFunc<'id, Self>>,
            rel: Ref<'_, EdgeOfFunc<'id, Self>>,
            meta: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            apply_relational_product(
                manager,
                ParallelRecursor::new(manager),
                level,
                set,
                rel,
                meta,
            )
        }

        /// See [`LDDFunction::relational_predecessor_edge`].
        #[inline]
        pub fn relational_predecessor_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            set: Ref<'_, EdgeOfFunc<'id, Self>>,
            rel: Ref<'_, EdgeOfFunc<'id, Self>>,
            meta: Ref<'_, EdgeOfFunc<'id, Self>>,
            universe: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            let rec = ParallelRecursor::new(manager);
            apply_relational_predecessor(manager, rec, level, set, rel, meta, universe)
        }

        /// See [`LDDFunction::len`].
        pub fn len(&self) -> usize {
            self.with_manager_shared(|manager, set| Self::len_edge(manager, set))
        }

        /// See [`LDDFunction::len_edge`].
        #[inline]
        pub fn len_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            set: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> usize {
            len(manager, set, &mut HashMap::new())
        }

        /// See [`LDDFunction::is_empty`].
        pub fn is_empty(&self) -> bool {
            self.with_manager_shared(|manager, edge| {
                manager.get_node(edge).is_terminal(&LDDTerminal::Empty)
            })
        }
    }
}
