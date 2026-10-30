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
    LDDManager, LDDOp, LDDTerminal, LDDValue, RIGHT, RelationMeta, collect_children, make_node,
    stat,
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

/// Computes the union `a ∪ b` of the two sets of vectors, both residing at
/// `level`.
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
    if let Some(res) = manager
        .apply_cache()
        .get(manager, LDDOp::Intersection, &[a, b])
    {
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

/// Computes the difference `a \ b` of the two sets of vectors, both residing
/// at `level`.
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
    if let Some(res) = manager
        .apply_cache()
        .get(manager, LDDOp::Difference, &[a, b])
    {
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
fn compute_proj<M: LDDManager>(manager: &M, proj: &[LevelNo]) -> AllocResult<Own<M::Edge>> {
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
            M::InnerNodeValue::INCLUDE
        } else {
            M::InnerNodeValue::EXCLUDE
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

    // Base case: the projection of the empty set is the empty set, even onto
    // no positions.
    if manager.get_node(set).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }

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

    let result = if *proj_value == M::InnerNodeValue::EXCLUDE {
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
    } else if *proj_value == M::InnerNodeValue::INCLUDE {
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
/// write values in the relation vectors.
///
/// Let `R = read_proj`, `W = write_proj` and `L = max(R ∪ W) + 1`. The meta-LDD
/// is the singleton `{m}`, where `m = m_0 · m_1 · … · m_{L-1}` is the
/// concatenation of
///
/// ```text
/// m_i = [read_of_pair, write_of_pair]  if i ∈ R ∩ W
///       [read_only]                    if i ∈ R \ W
///       [write_only]                   if i ∈ W \ R
///       [false]                        otherwise (not in the relation)
/// ```
///
/// Here `read_only` stands for [`LDDValue::read_only_value`], and similarly for
/// the other names. A position in `R ∩ W` is a read+write pair, which occupies
/// two levels of the meta-LDD: its *read half* and its *write half*.
///
/// A relation for this meta-LDD is a set of vectors `t = t_0 · t_1 · … ·
/// t_{L-1}` with
///
/// ```text
/// t_i = [a_i, b_i]  if i ∈ R ∩ W
///       [a_i]       if i ∈ R \ W
///       [b_i]       if i ∈ W \ R
///       []          otherwise
/// ```
///
/// where `a_i` is the value read and `b_i` the value written at position `i`.
/// So `t` has one entry for every entry of `m` other than `false`.
/// `read_positions` and
/// `write_positions` are the indices of the `a_i` and `b_i` in `t` (not in
/// `m`), in increasing order of `i`.
///
/// For example, `R = {0, 3}` and `W = {1, 3}` give
///
/// ```text
/// m = [read_only, write_only, false, read_of_pair, write_of_pair]
/// t = [a_0,       b_1,               a_3,          b_3          ]
/// ```
///
/// so `read_positions = [0, 2]` and `write_positions = [1, 3]`.
///
/// Shared implementation behind [`LDDFunction::relation_meta`] and
/// [`mt::LDDFunctionMT::relation_meta`].
fn relation_meta<M: LDDManager>(
    manager: &M,
    read_proj: &[LevelNo],
    write_proj: &[LevelNo],
) -> AllocResult<RelationMeta<Own<M::Edge>>> {
    // Compute length of meta.
    let length = std::cmp::max(
        read_proj.iter().max().map_or(0, |x| *x + 1),
        write_proj.iter().max().map_or(0, |x| *x + 1),
    );

    // Convert projection vectors to meta.
    let mut read_positions = Vec::new();
    let mut write_positions = Vec::new();

    let mut meta: Vec<M::InnerNodeValue> = Vec::new();

    let mut offset: LevelNo = 0;
    for i in 0..length {
        // Requires worst-case quadratic time, but the projection arrays are
        // typically small.
        let read = read_proj.contains(&i);
        let write = write_proj.contains(&i);
        if read && write {
            meta.push(M::InnerNodeValue::READ_OF_PAIR);
            meta.push(M::InnerNodeValue::WRITE_OF_PAIR);
            read_positions.push(offset);
            write_positions.push(offset + 1);
            offset += 2;
        } else if read {
            meta.push(M::InnerNodeValue::READ_ONLY);
            read_positions.push(offset);
            offset += 1;
        } else if write {
            meta.push(M::InnerNodeValue::WRITE_ONLY);
            write_positions.push(offset);
            offset += 1;
        } else {
            meta.push(M::InnerNodeValue::SKIP);
        }
    }

    Ok(RelationMeta {
        meta: singleton(manager, &meta)?,
        read_positions,
        write_positions,
    })
}

/// Computes the set of vectors reachable in one step from `set` via the sparse
/// relation `rel`, guided by `meta` (produced by [`relation_meta`]).
///
/// # Definition
///
/// Let `R`, `W`, `L`, `a_i(t)` and `b_i(t)` be as in [`relation_meta`].
/// At the root call (`level = 0`), the result is
///
/// ```text
/// { y | ∃ x ∈ set, t ∈ rel. (∀ i ∈ R. x_i = a_i(t))
///                         ∧ (∀ i ∈ W. y_i = b_i(t))
///                         ∧ (∀ i ∉ W. y_i = x_i) }
/// ```
///
/// That is, `t` applies to `x` if `x` agrees with all read values of `t`. The
/// successor `y` is then `x` with every position `i ∈ W` overwritten by the
/// write value `b_i(t)`. All other positions, including those `≥ L`, are
/// copied, so `y` has the same length as `x`.
///
/// # Algorithm
///
/// The function descends `set`, `rel` and `meta` simultaneously, and the value
/// of `meta` determines how each level is treated:
///
/// - `false`: emit the values of `set`; `rel` is not descended.
/// - `read_only`: emit the values of `set` that also occur in `rel`.
/// - `write_only`: ignore the values of `set` and emit the values of `rel`.
/// - `read_of_pair`: keep the values of `set` that also occur in `rel`, but do
///   not emit them; the write half emits the new value instead.
/// - `write_of_pair`: emit the values of `rel`; `set` is not descended, since
///   the read half already did.
///
/// The result resides at `level`, and so does `set`, except at a write half,
/// where `set` resides at `level + 1`.
///
/// # Cache
///
/// Unlike for the other operations, `level` is part of the cache key. Usually
/// `set` determines the result level, since every inner node resides in the
/// unique table of exactly one level. However, at a write half where the read
/// half consumed the last position of `set`, `set` is the `True` terminal. Then
/// only `rel` and `meta` remain, and these reside on the levels of the relation
/// vectors and the meta-LDD respectively. Neither determines the result level,
/// since a read+write pair occupies two levels of both, but only one result
/// level.
///
/// For example, take `read_proj = write_proj = [0, 1]` for sets of length 2,
/// and `read_proj = [0, 2]`, `write_proj = [1, 2]` for sets of length 3. This
/// gives
///
/// ```text
/// m = [read_of_pair, write_of_pair, read_of_pair, write_of_pair]
/// t = [a_0,          b_0,           a_1,          b_1          ]
///
/// m = [read_only,    write_only,    read_of_pair, write_of_pair]
/// t = [a_0,          b_1,           a_2,          b_2          ]
/// ```
///
/// In both cases, the last write half is at level 3 of `meta` and `rel`, but at
/// result level 1 and 2, respectively. If both relations end with the same
/// write values, the calls at these write halves have equal operands `(True,
/// rel, meta)`, and without `level` in the key a cache hit would return a node
/// from the wrong level. This cannot happen if all sets have the same length
/// `n`, since then `set` is only `True` at a write half for result level
/// `n - 1`.
fn apply_relational_successors<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    set: Ref<'_, M::Edge>,
    rel: Ref<'_, M::Edge>,
    meta: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    if rec.should_switch_to_sequential() {
        return apply_relational_successors(manager, SequentialRecursor, level, set, rel, meta);
    }
    stat!(call LDDOp::RelationalSuccessors);

    // `meta` is `True` once all of its levels are consumed; the remaining
    // positions of `set` are copied unchanged. Then `rel` is `True` as well,
    // unless it is the empty relation (with no read or write positions).
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
            return Ok(manager.clone_edge(set));
        }
        Node::Inner(n) => n,
    };

    // An empty set or relation yields the empty set. Since `meta` ends with a
    // read or write level, `rel` cannot be the `True` terminal here, but `set`
    // can be at a write half (see the function documentation).
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

    // The operands do not always determine the result level, so it is part of
    // the key (see the function documentation).
    stat!(cache_query LDDOp::RelationalSuccessors);
    if let Some(([res], [])) = manager.apply_cache().get_extended(
        manager,
        LDDOp::RelationalSuccessors,
        (&[set, rel, meta], &[level]),
    ) {
        stat!(cache_hit LDDOp::RelationalSuccessors);
        return Ok(res);
    }

    let meta_value = meta_node.value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::SKIP {
        // Not in the relation: emit the values of `set` unchanged. `rel` does
        // not have a level for this position, so it is not descended.
        let set_node = set_node.expect("set must have as many levels as meta");
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);

        let (down_result, right_result) = rec.ternary(
            apply_relational_successors,
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
    } else if *meta_value == M::InnerNodeValue::READ_ONLY {
        // Read-only: emit the values of `set` that also occur in `rel`.
        let set_node = set_node.expect("set must have as many levels as meta");
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // `rel` does not read this value of `set`; skip it.
                apply_relational_successors(manager, rec, level, set_right, rel, meta)?
            }
            Ordering::Equal => {
                let (down_result, right_result) = rec.ternary(
                    apply_relational_successors,
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
                // `set` does not contain this value of `rel`; skip it.
                apply_relational_successors(manager, rec, level, set, rel_right, meta)?
            }
        }
    } else if *meta_value == M::InnerNodeValue::WRITE_ONLY {
        // Write-only: the value of `set` is overwritten, so continue with the
        // union of all down-branches of `set`, and emit each value of `rel`.
        // The union is computed once for all values of `rel`.
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

        relational_successors_write_only(manager, rec, level, combined.borrowed(), rel, meta_down)?
    } else if *meta_value == M::InnerNodeValue::READ_OF_PAIR {
        // Read half of a pair: keep the values of `set` that also occur in
        // `rel`, but do not emit them; the write half emits the new values at
        // this result level instead.
        let set_node = set_node.expect("set must have as many levels as meta");
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            // `rel` does not read this value of `set`; skip it.
            Ordering::Less => {
                apply_relational_successors(manager, rec, level, set_right, rel, meta)?
            }
            Ordering::Equal => {
                // `meta_down` is the write half, which stays at `level`.
                // Different read values may write the same value, so the
                // results of both branches can overlap and must be combined by
                // a union instead of `make_node`.
                let (down_result, right_result) = rec.ternary(
                    apply_relational_successors,
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
                // `set` does not contain this value of `rel`; skip it.
                apply_relational_successors(manager, rec, level, set, rel_right, meta)?
            }
        }
    } else if *meta_value == M::InnerNodeValue::WRITE_OF_PAIR {
        // Write half of a pair: emit each value of `rel`. `set` was already
        // descended at the read half, so it is passed on unchanged.
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        let (down_result, right_result) = rec.ternary(
            apply_relational_successors,
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
        LDDOp::RelationalSuccessors,
        (&[set, rel, meta], &[level]),
        (&[result.borrowed()], &[]),
    );

    Ok(result)
}

/// Helper for the write-only ([`write_only_value`][LDDValue::write_only_value])
/// case of [`apply_relational_successors`].
///
/// Builds the LDD with a node `(v, apply_relational_successors(combined,
/// rel_v_down, meta))` for every value `v` in the right spine of `rel`, where
/// `rel_v_down` is the down-branch of the `rel` node carrying `v`. Here
/// `combined` is the union of all down-branches of `set`, and `combined` and
/// `meta` are already descended past the write level. In contrast to recursing
/// on the right sibling of `rel` in [`apply_relational_successors`], this
/// computes `combined` only once.
fn relational_successors_write_only<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    level: LevelNo,
    combined: Ref<'_, M::Edge>,
    rel: Ref<'_, M::Edge>,
    meta: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    let rel_node = match manager.get_node(rel) {
        // The right spine of `rel` ends at the Empty terminal.
        Node::Terminal(_) => return manager.get_terminal(LDDTerminal::Empty),
        Node::Inner(n) => n,
    };
    let rel_value = rel_node.value();
    let (rel_down, rel_right) = collect_children(rel_node);

    let down_result = EdgeDropGuard::new(
        manager,
        apply_relational_successors(manager, rec, level + 1, combined, rel_down, meta)?,
    );
    let right_result = EdgeDropGuard::new(
        manager,
        relational_successors_write_only(manager, rec, level, combined, rel_right, meta)?,
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
            rel_value,
            down_result.into_edge(),
            right_result.into_edge(),
        )
    }
}

/// Computes the set of source vectors in `universe` that can reach a vector in
/// `set` in one step via the sparse relation `rel`, guided by `meta` (produced
/// by [`relation_meta`]).
///
/// This is the inverse of [`apply_relational_successors`]: where that computes
/// the successors, this computes the predecessors restricted to `universe`.
///
/// # Definition
///
/// Let `R`, `W`, `a_i(t)` and `b_i(t)` be as in [`relation_meta`]. At
/// the root call (`level = 0`), the result is
///
/// ```text
/// { x ∈ universe | ∃ y ∈ set, t ∈ rel. (∀ i ∈ R. x_i = a_i(t))
///                                    ∧ (∀ i ∈ W. y_i = b_i(t))
///                                    ∧ (∀ i ∉ W. y_i = x_i) }
/// ```
///
/// That is, `x` is a predecessor if some `t ∈ rel` applies to `x` (see
/// [`apply_relational_successors`]) and yields a successor `y ∈ set`.
///
/// # Algorithm
///
/// The function descends `set` (the targets `y`), `universe` (the sources `x`),
/// `rel` and `meta` simultaneously, and the value of `meta` determines how each
/// level is treated:
///
/// - `false`: the source value equals the target value; emit the values that
///   occur in both `set` and `universe`. `rel` is not descended.
/// - `read_only`: the source value equals the target value, and is read by
///   `rel`; emit the values that occur in `set`, `rel` and `universe`.
/// - `write_only`: the target value must occur in both `set` and `rel`, but the
///   source value is unconstrained; emit every value of `universe`.
/// - `read_of_pair`: emit the values of `universe` that are read by `rel`.
///   `set` and `universe` are not descended; the write half does this.
/// - `write_of_pair`: the target value must occur in both `set` and `rel`;
///   nothing is emitted, since the read half already emitted the source value.
///
/// # Cache
///
/// The result resides at `level`, and so does `universe`, except at a write
/// half, where `universe` still resides at the level of the source value that
/// the read half emitted, i.e. `level - 1`. Unlike `set` in
/// [`apply_relational_successors`], `universe` is never the `True` terminal
/// while `meta` is an inner node, since the write half still descends it. Hence
/// the operands determine the result level, and it is not part of the cache
/// key.
fn apply_relational_predecessors<M: LDDManager, R: Recursor<M>>(
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
        return apply_relational_predecessors(manager, rec, level, set, rel, meta, universe);
    }
    stat!(call LDDOp::RelationalPredecessors);

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

    stat!(cache_query LDDOp::RelationalPredecessors);
    if let Some(res) = manager.apply_cache().get(
        manager,
        LDDOp::RelationalPredecessors,
        &[set, rel, meta, universe],
    ) {
        stat!(cache_hit LDDOp::RelationalPredecessors);
        return Ok(res);
    }

    let meta_value = meta_node.value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::SKIP {
        // Not in the relation: emit the values that occur in both `set` and
        // `universe`. `rel` does not have a level for this position, so it is
        // not descended.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let universe_value = universe_node.value();
        let (universe_down, universe_right) = collect_children(universe_node);

        match set_value.cmp(universe_value) {
            Ordering::Less => {
                // This set value is not in the universe; skip it.
                apply_relational_predecessors(manager, rec, level, set_right, rel, meta, universe)?
            }
            Ordering::Greater => {
                // This universe value is not in the set; skip it.
                apply_relational_predecessors(manager, rec, level, set, rel, meta, universe_right)?
            }
            Ordering::Equal => {
                let (down_result, right_result) = rec.quaternary(
                    apply_relational_predecessors,
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
    } else if *meta_value == M::InnerNodeValue::READ_ONLY {
        // Read-only: the source value equals the target value, so emit the
        // values that occur in `set`, `rel` and `universe`.
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
                apply_relational_predecessors,
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
                apply_relational_predecessors(manager, rec, level, set_right, rel, meta, universe)?
            } else if universe_value < max {
                apply_relational_predecessors(manager, rec, level, set, rel, meta, universe_right)?
            } else {
                // rel_value < max
                apply_relational_predecessors(manager, rec, level, set, rel_right, meta, universe)?
            }
        }
    } else if *meta_value == M::InnerNodeValue::WRITE_ONLY {
        // Write-only: the target value must occur in both `set` and `rel`, but
        // the source value is unconstrained and ranges over all of `universe`.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // This set value is not written by the relation; skip it.
                apply_relational_predecessors(manager, rec, level, set_right, rel, meta, universe)?
            }
            Ordering::Greater => {
                // This relation value is not in the set; skip it.
                apply_relational_predecessors(manager, rec, level, set, rel_right, meta, universe)?
            }
            Ordering::Equal => {
                // Emit a source node for every value in the universe, all
                // sharing the matched continuation.
                let down_result = EdgeDropGuard::new(
                    manager,
                    relational_predecessors_universe(
                        manager, rec, level, set_down, rel_down, meta_down, universe,
                    )?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_predecessors(
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
    } else if *meta_value == M::InnerNodeValue::READ_OF_PAIR {
        // Read half of a pair: emit the values of `universe` that are read by
        // `rel`. `set` and `universe` are not descended here, but at the write
        // half.
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);
        let universe_value = universe_node.value();
        let (_universe_down, universe_right) = collect_children(universe_node);

        match universe_value.cmp(rel_value) {
            Ordering::Less => {
                // This universe value cannot be read; skip it.
                apply_relational_predecessors(manager, rec, level, set, rel, meta, universe_right)?
            }
            Ordering::Greater => {
                // This relation value is not in the universe; skip it.
                apply_relational_predecessors(manager, rec, level, set, rel_right, meta, universe)?
            }
            Ordering::Equal => {
                // `meta_down` is the write half; the universe is kept
                // so that its down-branch is descended there.
                let (down_result, rest) = rec.quaternary(
                    apply_relational_predecessors,
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
    } else if *meta_value == M::InnerNodeValue::WRITE_OF_PAIR {
        // Write half of a pair: the target value must occur in both `set` and
        // `rel`; nothing is emitted. The read half already emitted the source
        // value, so descend `universe` here.
        let set_value = set_node.value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // This target value is not written by the relation; skip it.
                apply_relational_predecessors(manager, rec, level, set_right, rel, meta, universe)?
            }
            Ordering::Greater => {
                // This relation value is not in the set; skip it.
                apply_relational_predecessors(manager, rec, level, set, rel_right, meta, universe)?
            }
            Ordering::Equal => {
                let (universe_down, _universe_right) = collect_children(universe_node);
                let (down_result, right_result) = rec.quaternary(
                    apply_relational_predecessors,
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
        LDDOp::RelationalPredecessors,
        &[set, rel, meta, universe],
        result.borrowed(),
    );

    Ok(result)
}

/// Helper for the write-only ([`write_only_value`][LDDValue::write_only_value])
/// case of [`apply_relational_predecessors`].
///
/// Builds the union over every value `v` in `universe` of the source node
/// `(v, apply_relational_predecessors(set, rel, meta, universe_v_down))`, where
/// `universe_v_down` is the down-branch of the universe node carrying value
/// `v`. The `set`, `rel`, and `meta` arguments are already descended past the
/// write level.
fn relational_predecessors_universe<M: LDDManager, R: Recursor<M>>(
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
        apply_relational_predecessors(manager, rec, level + 1, set, rel, meta, universe_down)?,
    );
    let right_result = EdgeDropGuard::new(
        manager,
        relational_predecessors_universe(manager, rec, level, set, rel, meta, universe_right)?,
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
        proj: &[LevelNo],
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
    pub fn relation_meta<'id>(
        manager: &<Self as Function>::Manager<'id>,
        read_proj: &[LevelNo],
        write_proj: &[LevelNo],
    ) -> AllocResult<RelationMeta<OwnEdgeOfFunc<'id, Self>>> {
        relation_meta(manager, read_proj, write_proj)
    }

    /// Computes the set of vectors reachable in one step from `self` via the
    /// sparse relation `rel`, guided by `meta` (produced by
    /// [`relation_meta`][Self::relation_meta]).
    ///
    /// This is also known as the relational product.
    pub fn relational_successors(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let edge = Self::relational_successors_edge(
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
    /// [`relation_meta`][Self::relation_meta]).
    ///
    /// This is the inverse of
    /// [`relational_successors`][Self::relational_successors].
    pub fn relational_predecessors(
        &self,
        rel: &Self,
        meta: &Self,
        universe: &Self,
    ) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let edge = Self::relational_predecessors_edge(
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
    /// [`relation_meta`][Self::relation_meta].
    ///
    /// `set` must reside at `level`; `rel` and `meta` are sub-LDDs matching the
    /// positions from `level` onwards.
    #[inline]
    pub fn relational_successors_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        set: Ref<'_, EdgeOfFunc<'id, Self>>,
        rel: Ref<'_, EdgeOfFunc<'id, Self>>,
        meta: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_relational_successors(manager, SequentialRecursor, level, set, rel, meta)
    }

    /// Computes the set of source vectors in `universe` that can reach a vector
    /// in `set` in one step via the sparse relation `rel`.  `meta` must be
    /// produced by [`relation_meta`][Self::relation_meta].
    ///
    /// This is the inverse of [`Self::relational_successors_edge`].
    ///
    /// `set` and `universe` must reside at `level`.
    #[inline]
    pub fn relational_predecessors_edge<'id>(
        manager: &<Self as Function>::Manager<'id>,
        level: LevelNo,
        set: Ref<'_, EdgeOfFunc<'id, Self>>,
        rel: Ref<'_, EdgeOfFunc<'id, Self>>,
        meta: Ref<'_, EdgeOfFunc<'id, Self>>,
        universe: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_relational_predecessors(manager, SequentialRecursor, level, set, rel, meta, universe)
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
            proj: &[LevelNo],
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

        /// See [`LDDFunction::relation_meta`].
        #[inline]
        pub fn relation_meta<'id>(
            manager: &<Self as Function>::Manager<'id>,
            read_proj: &[LevelNo],
            write_proj: &[LevelNo],
        ) -> AllocResult<RelationMeta<OwnEdgeOfFunc<'id, Self>>> {
            relation_meta(manager, read_proj, write_proj)
        }

        /// See [`LDDFunction::relational_successors`].
        pub fn relational_successors(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
            self.with_manager_shared(|manager, set| {
                let edge = Self::relational_successors_edge(
                    manager,
                    0,
                    set,
                    rel.as_edge(manager),
                    meta.as_edge(manager),
                )?;
                Ok(Self::from_edge(manager, edge))
            })
        }

        /// See [`LDDFunction::relational_predecessors`].
        pub fn relational_predecessors(
            &self,
            rel: &Self,
            meta: &Self,
            universe: &Self,
        ) -> AllocResult<Self> {
            self.with_manager_shared(|manager, set| {
                let edge = Self::relational_predecessors_edge(
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

        /// See [`LDDFunction::relational_successors_edge`].
        #[inline]
        pub fn relational_successors_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            set: Ref<'_, EdgeOfFunc<'id, Self>>,
            rel: Ref<'_, EdgeOfFunc<'id, Self>>,
            meta: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            apply_relational_successors(
                manager,
                ParallelRecursor::new(manager),
                level,
                set,
                rel,
                meta,
            )
        }

        /// See [`LDDFunction::relational_predecessors_edge`].
        #[inline]
        pub fn relational_predecessors_edge<'id>(
            manager: &<Self as Function>::Manager<'id>,
            level: LevelNo,
            set: Ref<'_, EdgeOfFunc<'id, Self>>,
            rel: Ref<'_, EdgeOfFunc<'id, Self>>,
            meta: Ref<'_, EdgeOfFunc<'id, Self>>,
            universe: Ref<'_, EdgeOfFunc<'id, Self>>,
        ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
            let rec = ParallelRecursor::new(manager);
            apply_relational_predecessors(manager, rec, level, set, rel, meta, universe)
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
