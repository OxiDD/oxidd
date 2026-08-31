//! Recursive apply algorithms for list decision diagrams.
//!
//! These are the core recursive operations on LDDs. They are generic over a
//! [`Recursor`], which determines whether independent sub-problems are solved
//! sequentially or split onto a worker pool (see [`crate::recursor`]).

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashMap;

use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::{ApplyCache, Edge, InnerNode, Manager, Node, NodeID};

use crate::recursor::{Recursor, SequentialRecursor};
use crate::stat;
use crate::{LDDManager, LDDOp, LDDTerminal, LDDValue, RelationProductMeta};

#[cfg(feature = "statistics")]
struct StatCounters {
    calls: std::sync::atomic::AtomicI64,
    cache_queries: std::sync::atomic::AtomicI64,
    cache_hits: std::sync::atomic::AtomicI64,
    reduced: std::sync::atomic::AtomicI64,
}

#[cfg(feature = "statistics")]
impl StatCounters {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: StatCounters = StatCounters {
        calls: std::sync::atomic::AtomicI64::new(0),
        cache_queries: std::sync::atomic::AtomicI64::new(0),
        cache_hits: std::sync::atomic::AtomicI64::new(0),
        reduced: std::sync::atomic::AtomicI64::new(0),
    };

    fn print<O: oxidd_core::Countable + std::fmt::Debug>(counters: &[Self]) {
        // spell-checker:ignore ctrs
        for (i, ctrs) in counters.iter().enumerate() {
            let calls = ctrs.calls.swap(0, std::sync::atomic::Ordering::Relaxed);
            let cache_queries = ctrs
                .cache_queries
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            let cache_hits = ctrs
                .cache_hits
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            let reduced = ctrs.reduced.swap(0, std::sync::atomic::Ordering::Relaxed);

            if calls == 0 {
                continue;
            }

            let terminal_percent = (calls - cache_queries) as f32 / calls as f32 * 100.0;
            let cache_hit_percent = cache_hits as f32 / cache_queries as f32 * 100.0;
            let op = <O as oxidd_core::Countable>::from_usize(i);
            eprintln!(
                "  {op:?}: calls: {calls}, cache queries: {cache_queries} ({terminal_percent} % terminal cases), cache hits: {cache_hits} ({cache_hit_percent} %), reduced: {reduced}"
            );
        }
    }
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [StatCounters; <LDDOp as oxidd_core::Countable>::MAX_VALUE + 1] =
    [StatCounters::INIT; <LDDOp as oxidd_core::Countable>::MAX_VALUE + 1];

/// Print statistics to stderr
#[cfg(feature = "statistics")]
pub fn print_stats() {
    eprintln!("[oxidd_rules_ldd]");
    StatCounters::print::<LDDOp>(&STAT_COUNTERS);
}

/// Builds the meta-LDD encoding the read/write projection described by
/// `read_proj` and `write_proj`, together with the positions of the read and
/// write variables in that encoding.
///
/// Shared implementation behind [`LDDFunction::relation_product_meta`] and
/// [`mt::LDDFunctionMT::relation_product_meta`].
pub(crate) fn relation_product_meta<M: LDDManager>(
    manager: &M,
    read_proj: &[u32],
    write_proj: &[u32],
) -> AllocResult<RelationProductMeta<M::Edge>> {
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
        let read = read_proj.contains(&i);
        let write = write_proj.contains(&i);
        if read && write {
            meta.push(<_ as LDDValue>::read_of_pair_value());
            meta.push(<_ as LDDValue>::write_of_pair_value());
            read_positions.push(offset);
            write_positions.push(offset + 1);
            offset += 2;
        } else if read {
            meta.push(<_ as LDDValue>::read_only_value());
            read_positions.push(offset);
            offset += 1;
        } else if write {
            meta.push(<_ as LDDValue>::write_only_value());
            write_positions.push(offset);
            offset += 1;
        } else {
            meta.push(<_ as LDDValue>::false_value());
        }
    }

    Ok(RelationProductMeta {
        meta: singleton(manager, &meta)?,
        read_positions,
        write_positions,
    })
}

/// Return a singleton LDD that contains only the list represented by `vector`.
pub(crate) fn singleton<M: LDDManager>(
    manager: &M,
    vector: &[M::InnerNodeValue],
) -> AllocResult<M::Edge> {
    let mut root = manager.get_terminal(LDDTerminal::True)?;

    for val in vector.iter().rev() {
        root = make_node(
            manager,
            val,
            root,
            manager.get_terminal(LDDTerminal::Empty)?,
        )?;
    }

    Ok(root)
}

/// Computes a meta LDD that is suitable for the [project] function from the
/// given projection indices.
///
/// This function is useful to be able to cache the projection LDD instead of
/// computing it from the projection array every time.
pub(crate) fn compute_proj<M: LDDManager>(manager: &M, proj: &[u32]) -> AllocResult<M::Edge> {
    // Compute length of proj.
    let length = match proj.iter().max() {
        Some(x) => *x + 1,
        None => 0,
    };

    // Convert projection vectors to meta information.
    let mut result: Vec<M::InnerNodeValue> = Vec::new();
    for i in 0..length {
        let included = proj.contains(&i);

        if included {
            result.push(M::InnerNodeValue::true_value());
        } else {
            result.push(M::InnerNodeValue::false_value());
        }
    }

    singleton(manager, &result)
}

/// Computes the set of vectors projected onto the given indices, where proj is
/// equal to compute_proj([i_0, ..., i_k]).
///
/// Formally, for a single vector <x_0, ..., x_n> we have that:
///     - project(<x_0, ..., x_n>, i_0 < ... < i_k) = <x_(i_0), ..., x_(i_k)>
///     - project(X, i_0 < ... < i_k) = { project(x, i_0 < ... < i_k) | x in X
///       }.
///
/// Note that the indices are sorted in the definition, but compute_proj
/// can take any array and ignores both duplicates and order. Also, it
/// follows that i_k must be smaller than or equal to n as x_(i_k) is not
/// defined otherwise.
pub(crate) fn project<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    set: Borrowed<M::Edge>,
    proj: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if rec.should_switch_to_sequential() {
        return project(manager, SequentialRecursor, set, proj);
    }
    stat!(call LDDOp::Project);

    // Base case: if proj has reached the True terminal, the projection is
    // fully consumed — return the True terminal (empty vector).
    let proj_node = match manager.get_node(&proj) {
        Node::Terminal(terminal) => {
            if *terminal.borrow() == LDDTerminal::True {
                return manager.get_terminal(LDDTerminal::True);
            }
            unreachable!("proj should never be the Empty terminal");
        }
        Node::Inner(proj_node) => proj_node,
    };

    // Base case: if set is the Empty terminal, the result is the empty set.
    let set_node = match manager.get_node(&set) {
        Node::Terminal(terminal) => {
            if *terminal.borrow() == LDDTerminal::Empty {
                return manager.get_terminal(LDDTerminal::Empty);
            }
            // set is the True terminal (empty vector) but proj still has
            // levels — proj extends beyond the length of set.
            debug_assert!(false, "proj can be at most as high as set");
            unreachable!();
        }
        Node::Inner(set_node) => set_node,
    };

    stat!(cache_query LDDOp::Project);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Project, &[set.borrowed(), proj.borrowed()])
    {
        stat!(cache_hit LDDOp::Project);
        return Ok(res);
    }

    let proj_value = proj_node.get_value();
    let (proj_down, _proj_right) = collect_children(proj_node);
    let set_value = set_node.get_value();
    let (set_down, set_right) = collect_children(set_node);

    let result = if *proj_value == M::InnerNodeValue::false_value() {
        // This position is not in the projection: skip it and compute the union of the
        // right and down branches.
        let (right_result, down_result) = rec.binary(
            project,
            manager,
            (set_right, proj.borrowed()),
            (set_down, proj_down),
        )?;
        apply_union(
            manager,
            rec,
            right_result.borrowed(),
            down_result.borrowed(),
        )
    } else if *proj_value == M::InnerNodeValue::true_value() {
        // This position is in the projection: keep the current value and
        // recurse on both branches.
        let (right_result, down_result) = rec.binary(
            project,
            manager,
            (set_right, proj.borrowed()),
            (set_down, proj_down),
        )?;

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            // The down sub-result is empty — nothing to insert, return right only.
            Ok(right_result.into_edge())
        } else {
            make_node(
                manager,
                set_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )
        }
    } else {
        panic!("proj has unexpected value");
    }?;

    manager
        .apply_cache()
        .add(manager, LDDOp::Project, &[set, proj], result.borrowed());

    Ok(result)
}

/// Recursively apply the 'union' operator to `f` and `g`
pub(crate) fn apply_union<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if rec.should_switch_to_sequential() {
        return apply_union(manager, SequentialRecursor, f, g);
    }
    stat!(call LDDOp::Union);

    if f == g {
        return Ok(manager.clone_edge(&f));
    }

    // Query apply cache
    stat!(cache_query LDDOp::Union);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Union, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit LDDOp::Union);
        return Ok(res);
    }

    // The Empty terminal is the identity for union.
    if manager.get_node(&f).is_terminal(&LDDTerminal::Empty) {
        return Ok(manager.clone_edge(&g));
    }
    if manager.get_node(&g).is_terminal(&LDDTerminal::Empty) {
        return Ok(manager.clone_edge(&f));
    }

    let f_node = match manager.get_node(&f) {
        Node::Inner(f_node) => f_node.borrow(),
        Node::Terminal(_) => unreachable!("Invalid terminal"),
    };

    let g_node = match manager.get_node(&g) {
        Node::Inner(g_node) => g_node.borrow(),
        Node::Terminal(_) => unreachable!("Invalid terminal"),
    };

    let result = match f_node.get_value().cmp(g_node.get_value()) {
        Ordering::Less => {
            let (f_down, f_right) = collect_children(f_node);

            let right = apply_union(manager, rec, f_right, g.borrowed())?;
            make_node(
                manager,
                f_node.get_value(),
                manager.clone_edge(&f_down),
                right,
            )
        }
        Ordering::Greater => {
            // g carries the smaller value, which f lacks: keep g's value and
            // down-branch, and union f with g's remaining (right) values.
            let (g_down, g_right) = collect_children(g_node);

            let right = apply_union(manager, rec, f.borrowed(), g_right)?;
            make_node(
                manager,
                g_node.get_value(),
                manager.clone_edge(&g_down),
                right,
            )
        }
        Ordering::Equal => {
            // Both carry the same value: union the down-branches (the
            // continuations) and the right-branches (the further values).
            let (f_down, f_right) = collect_children(f_node);
            let (g_down, g_right) = collect_children(g_node);

            let (low, high) =
                rec.binary(apply_union, manager, (f_down, g_down), (f_right, g_right))?;
            make_node(
                manager,
                g_node.get_value(),
                low.into_edge(),
                high.into_edge(),
            )
        }
    }?;

    manager
        .apply_cache()
        .add(manager, LDDOp::Union, &[f, g], result.borrowed());

    Ok(result)
}

/// Returns the largest subset of `a` that contains no element of `b`
/// (set difference `a \ b`).
pub(crate) fn apply_minus<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    a: Borrowed<M::Edge>,
    b: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if rec.should_switch_to_sequential() {
        return apply_minus(manager, SequentialRecursor, a, b);
    }
    stat!(call LDDOp::Minus);

    // a \ a == ∅  and  ∅ \ b == ∅
    if a == b || manager.get_node(&a).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    // a \ ∅ == a
    if manager.get_node(&b).is_terminal(&LDDTerminal::Empty) {
        return Ok(manager.clone_edge(&a));
    }

    stat!(cache_query LDDOp::Minus);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Minus, &[a.borrowed(), b.borrowed()])
    {
        stat!(cache_hit LDDOp::Minus);
        return Ok(res);
    }

    let a_node = match manager.get_node(&a) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!("a is not empty"),
    };
    let b_node = match manager.get_node(&b) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!("b is not empty"),
    };
    let a_value = a_node.get_value();
    let (a_down, a_right) = collect_children(a_node);
    let b_value = b_node.get_value();
    let (b_down, b_right) = collect_children(b_node);

    let result = match a_value.cmp(b_value) {
        Ordering::Less => {
            // b has no entry for this value — keep a's entry, advance a_right.
            let right_result =
                EdgeDropGuard::new(manager, apply_minus(manager, rec, a_right, b.borrowed())?);
            make_node(
                manager,
                a_value,
                manager.clone_edge(&a_down),
                right_result.into_edge(),
            )?
        }
        Ordering::Equal => {
            // Subtract b_down from a_down; subtract b_right from a_right.
            let (down_result, right_result) =
                rec.binary(apply_minus, manager, (a_down, b_down), (a_right, b_right))?;
            if manager
                .get_node(&down_result)
                .is_terminal(&LDDTerminal::Empty)
            {
                right_result.into_edge()
            } else {
                make_node(
                    manager,
                    a_value,
                    down_result.into_edge(),
                    right_result.into_edge(),
                )?
            }
        }
        Ordering::Greater => {
            // a has no entry for b's value — skip b's entry.
            apply_minus(manager, rec, a.borrowed(), b_right)?
        }
    };

    manager
        .apply_cache()
        .add(manager, LDDOp::Minus, &[a, b], result.borrowed());

    Ok(result)
}

/// Computes the set of vectors reachable in one step from `set` via the sparse
/// relation `rel`, guided by `meta` (produced by
/// [`LDDFunction::relation_product_meta`]).
///
/// Meta values at each level:
/// - 0 – position not in the relation: keep set values, advance meta only.
/// - 1 – read-only: match set and rel values; keep matched values in output.
/// - 2 – write-only: union all set values, then write each rel value.
/// - 3 – read half of a read+write pair: match values (not emitted); meta_down
///   is 4.
/// - 4 – write half of a read+write pair: write rel values.
pub(crate) fn apply_relational_product<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    set: Borrowed<M::Edge>,
    rel: Borrowed<M::Edge>,
    meta: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if rec.should_switch_to_sequential() {
        return apply_relational_product(manager, SequentialRecursor, set, rel, meta);
    }
    stat!(call LDDOp::RelationalProduct);

    // meta == True means all meta levels consumed; return set unchanged.
    match manager.get_node(&meta) {
        Node::Terminal(t) => {
            debug_assert_eq!(
                *t.borrow(),
                LDDTerminal::True,
                "meta should never reach the Empty terminal"
            );
            return Ok(manager.clone_edge(&set));
        }
        Node::Inner(_) => {}
    }

    // Empty set or empty relation → empty result.
    if manager.get_node(&set).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    if manager.get_node(&rel).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }

    stat!(cache_query LDDOp::RelationalProduct);
    if let Some(res) = manager.apply_cache().get(
        manager,
        LDDOp::RelationalProduct,
        &[set.borrowed(), rel.borrowed(), meta.borrowed()],
    ) {
        stat!(cache_hit LDDOp::RelationalProduct);
        return Ok(res);
    }

    let meta_node = match manager.get_node(&meta) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!(),
    };
    let meta_value = meta_node.get_value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::false_value() {
        // 0: not in relation — keep all set values, advance meta into next level.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);

        let (down_result, right_result) = rec.ternary(
            apply_relational_product,
            manager,
            (set_down, rel.borrowed(), meta_down),
            (set_right, rel.borrowed(), meta.borrowed()),
        )?;

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                set_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else if *meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only — match set and rel values; keep matched values in output.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // No rel entry for this set value; skip it.
                apply_relational_product(manager, rec, set_right, rel.borrowed(), meta.borrowed())?
            }
            Ordering::Equal => {
                let (down_result, right_result) = rec.ternary(
                    apply_relational_product,
                    manager,
                    (set_down, rel_down, meta_down),
                    (set_right, rel_right, meta.borrowed()),
                )?;
                if manager
                    .get_node(&down_result)
                    .is_terminal(&LDDTerminal::Empty)
                {
                    right_result.into_edge()
                } else {
                    make_node(
                        manager,
                        set_value,
                        down_result.into_edge(),
                        right_result.into_edge(),
                    )?
                }
            }
            Ordering::Greater => {
                // No set entry for this rel value; skip the rel value.
                apply_relational_product(manager, rec, set.borrowed(), rel_right, meta.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only — union all set down-branches (ignoring their values),
        // then write each rel value with that combined continuation.
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        // Collect union of all down-branches at this set level.
        let combined = {
            let mut acc = EdgeDropGuard::new(manager, manager.get_terminal(LDDTerminal::Empty)?);
            let mut cur = EdgeDropGuard::new(manager, manager.clone_edge(&set));
            loop {
                let (down_owned, right_owned, right_empty) = {
                    let cur_node = match manager.get_node(&cur) {
                        Node::Inner(n) => n.borrow(),
                        _ => unreachable!(),
                    };
                    let (cur_down, cur_right) = collect_children(cur_node);
                    let empty = manager
                        .get_node(&cur_right)
                        .is_terminal(&LDDTerminal::Empty);
                    (
                        EdgeDropGuard::new(manager, manager.clone_edge(&cur_down)),
                        EdgeDropGuard::new(manager, manager.clone_edge(&cur_right)),
                        empty,
                    )
                };
                let new_acc = EdgeDropGuard::new(
                    manager,
                    apply_union(manager, rec, acc.borrowed(), down_owned.borrowed())?,
                );
                // down_owned and acc are dropped via EdgeDropGuard
                drop(down_owned);
                acc = new_acc;
                if right_empty {
                    // right_owned is dropped via EdgeDropGuard
                    break;
                }
                // old cur is dropped via EdgeDropGuard, replaced by right_owned
                cur = right_owned;
            }
            // cur is dropped via EdgeDropGuard at end of block
            acc.into_edge()
        };

        let combined_guard = EdgeDropGuard::new(manager, combined);
        let (down_result, right_result) = rec.ternary(
            apply_relational_product,
            manager,
            (combined_guard.borrowed(), rel_down, meta_down),
            (set.borrowed(), rel_right, meta.borrowed()),
        )?;

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                rel_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair — match values (not emitted);
        // union the matched continuation with the remaining-pairs result.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                apply_relational_product(manager, rec, set_right, rel.borrowed(), meta.borrowed())?
            }
            Ordering::Equal => {
                // meta_down should be write_of_pair_value (4); it introduces
                // the new values.  Union with the right-siblings result so all
                // (read, write) pairs at this level are considered.
                let (down_result, right_result) = rec.ternary(
                    apply_relational_product,
                    manager,
                    (set_down, rel_down, meta_down),
                    (set_right, rel_right, meta.borrowed()),
                )?;
                apply_union(
                    manager,
                    rec,
                    down_result.borrowed(),
                    right_result.borrowed(),
                )?
            }
            Ordering::Greater => {
                apply_relational_product(manager, rec, set.borrowed(), rel_right, meta.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_of_pair_value() {
        // 4: write half of a read+write pair — emit rel values.
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        let (down_result, right_result) = rec.ternary(
            apply_relational_product,
            manager,
            (set.borrowed(), rel_down, meta_down),
            (set.borrowed(), rel_right, meta.borrowed()),
        )?;

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                rel_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else {
        panic!("meta has unexpected value");
    };

    manager.apply_cache().add(
        manager,
        LDDOp::RelationalProduct,
        &[set, rel, meta],
        result.borrowed(),
    );

    Ok(result)
}

/// Computes the intersection `a ∩ b` of the two sets of vectors.
pub(crate) fn apply_intersect<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    a: Borrowed<M::Edge>,
    b: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if rec.should_switch_to_sequential() {
        return apply_intersect(manager, SequentialRecursor, a, b);
    }
    stat!(call LDDOp::Intersect);

    // a ∩ a == a
    if a == b {
        return Ok(manager.clone_edge(&a));
    }
    // ∅ ∩ b == ∅  and  a ∩ ∅ == ∅
    if manager.get_node(&a).is_terminal(&LDDTerminal::Empty)
        || manager.get_node(&b).is_terminal(&LDDTerminal::Empty)
    {
        return manager.get_terminal(LDDTerminal::Empty);
    }

    stat!(cache_query LDDOp::Intersect);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Intersect, &[a.borrowed(), b.borrowed()])
    {
        stat!(cache_hit LDDOp::Intersect);
        return Ok(res);
    }

    // Both are non-empty and unequal. The `True` terminal is unique, so it is
    // handled by the `a == b` case above; hence both must be inner nodes (the
    // two operands always reside on the same level).
    let a_node = match manager.get_node(&a) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!("a and b reside on the same level"),
    };
    let b_node = match manager.get_node(&b) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!("a and b reside on the same level"),
    };
    let a_value = a_node.get_value();
    let (a_down, a_right) = collect_children(a_node);
    let b_value = b_node.get_value();
    let (b_down, b_right) = collect_children(b_node);

    let result = match a_value.cmp(b_value) {
        Ordering::Less => {
            // a's value is not in b; skip it.
            apply_intersect(manager, rec, a_right, b.borrowed())?
        }
        Ordering::Greater => {
            // b's value is not in a; skip it.
            apply_intersect(manager, rec, a.borrowed(), b_right)?
        }
        Ordering::Equal => {
            let (down_result, right_result) = rec.binary(
                apply_intersect,
                manager,
                (a_down, b_down),
                (a_right, b_right),
            )?;
            if manager
                .get_node(&down_result)
                .is_terminal(&LDDTerminal::Empty)
            {
                right_result.into_edge()
            } else {
                make_node(
                    manager,
                    a_value,
                    down_result.into_edge(),
                    right_result.into_edge(),
                )?
            }
        }
    };

    manager
        .apply_cache()
        .add(manager, LDDOp::Intersect, &[a, b], result.borrowed());

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
/// - 0 – position not in the relation: keep values present in both `set` and
///   `universe`, advance meta only.
/// - 1 – read-only: the source value equals the target value; keep values that
///   appear in `set`, `rel`, and `universe`.
/// - 2 – write-only: the source value is unconstrained, ranging over all of
///   `universe`; the target value must appear in `set` and `rel`.
/// - 3 – read half of a read+write pair: emit the source value taken from
///   `universe` (matched against `rel`).
/// - 4 – write half of a read+write pair: consume the target value (matched
///   against `set` and `rel`); no value is emitted.
pub(crate) fn apply_relational_predecessor<M: LDDManager, R: Recursor<M>>(
    manager: &M,
    rec: R,
    set: Borrowed<M::Edge>,
    rel: Borrowed<M::Edge>,
    meta: Borrowed<M::Edge>,
    universe: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if rec.should_switch_to_sequential() {
        return apply_relational_predecessor(manager, SequentialRecursor, set, rel, meta, universe);
    }
    stat!(call LDDOp::RelationalPredecessor);

    // An empty set, relation, or universe yields the empty set.
    if manager.get_node(&set).is_terminal(&LDDTerminal::Empty)
        || manager.get_node(&rel).is_terminal(&LDDTerminal::Empty)
        || manager.get_node(&universe).is_terminal(&LDDTerminal::Empty)
    {
        return manager.get_terminal(LDDTerminal::Empty);
    }

    // meta == True means all meta levels are consumed; the remaining source
    // vectors are exactly those in both `set` and `universe`.
    match manager.get_node(&meta) {
        Node::Terminal(t) => {
            debug_assert_eq!(
                *t.borrow(),
                LDDTerminal::True,
                "meta should never reach the Empty terminal"
            );
            return apply_intersect(manager, rec, set.borrowed(), universe.borrowed());
        }
        Node::Inner(_) => {}
    }

    stat!(cache_query LDDOp::RelationalPredecessor);
    if let Some(res) = manager.apply_cache().get(
        manager,
        LDDOp::RelationalPredecessor,
        &[
            set.borrowed(),
            rel.borrowed(),
            meta.borrowed(),
            universe.borrowed(),
        ],
    ) {
        stat!(cache_hit LDDOp::RelationalPredecessor);
        return Ok(res);
    }

    let meta_node = match manager.get_node(&meta) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!(),
    };
    let meta_value = meta_node.get_value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::false_value() {
        // 0: not in relation — keep values present in both set and universe.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let universe_node = match manager.get_node(&universe) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("universe must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let universe_value = universe_node.get_value();
        let (universe_down, universe_right) = collect_children(universe_node);

        match set_value.cmp(universe_value) {
            Ordering::Less => {
                // This set value is not in the universe; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set_right,
                    rel.borrowed(),
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
            Ordering::Greater => {
                // This universe value is not in the set; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel.borrowed(),
                    meta.borrowed(),
                    universe_right,
                )?
            }
            Ordering::Equal => {
                let (down_result, right_result) = rec.quaternary(
                    apply_relational_predecessor,
                    manager,
                    (set_down, rel.borrowed(), meta_down, universe_down),
                    (set_right, rel.borrowed(), meta.borrowed(), universe_right),
                )?;
                if manager
                    .get_node(&down_result)
                    .is_terminal(&LDDTerminal::Empty)
                {
                    right_result.into_edge()
                } else {
                    make_node(
                        manager,
                        set_value,
                        down_result.into_edge(),
                        right_result.into_edge(),
                    )?
                }
            }
        }
    } else if *meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only — the source value equals the target value, so emit
        // values that appear in set, rel, and universe simultaneously.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let universe_node = match manager.get_node(&universe) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("universe must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);
        let universe_value = universe_node.get_value();
        let (universe_down, universe_right) = collect_children(universe_node);

        if set_value == rel_value && rel_value == universe_value {
            // All three agree on the value: emit it and recurse downwards on
            // all of set, rel, and universe; the right siblings are explored by
            // advancing the relation.
            let (down_result, rest) = rec.quaternary(
                apply_relational_predecessor,
                manager,
                (set_down, rel_down, meta_down, universe_down),
                (
                    set.borrowed(),
                    rel_right,
                    meta.borrowed(),
                    universe.borrowed(),
                ),
            )?;
            let node = EdgeDropGuard::new(
                manager,
                if manager
                    .get_node(&down_result)
                    .is_terminal(&LDDTerminal::Empty)
                {
                    manager.get_terminal(LDDTerminal::Empty)?
                } else {
                    make_node(
                        manager,
                        universe_value,
                        manager.clone_edge(&down_result),
                        manager.get_terminal(LDDTerminal::Empty)?,
                    )?
                },
            );
            apply_union(manager, rec, node.borrowed(), rest.borrowed())?
        } else {
            // Advance whichever operands are below the maximum value so that
            // all three line up on a common value.
            let max = set_value.max(rel_value).max(universe_value);
            if set_value < max {
                apply_relational_predecessor(
                    manager,
                    rec,
                    set_right,
                    rel.borrowed(),
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            } else if universe_value < max {
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel.borrowed(),
                    meta.borrowed(),
                    universe_right,
                )?
            } else {
                // rel_value < max
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel_right,
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only — the target value must appear in both set and rel; the
        // source value is unconstrained and ranges over the whole universe.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // This set value is not written by the relation; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set_right,
                    rel.borrowed(),
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
            Ordering::Greater => {
                // This relation value is not in the set; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel_right,
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
            Ordering::Equal => {
                // Emit a source node for every value in the universe, all
                // sharing the matched continuation.
                let down_result = EdgeDropGuard::new(
                    manager,
                    relational_predecessor_universe(
                        manager,
                        rec,
                        set_down,
                        rel_down,
                        meta_down,
                        universe.borrowed(),
                    )?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_predecessor(
                        manager,
                        rec,
                        set.borrowed(),
                        rel_right,
                        meta.borrowed(),
                        universe.borrowed(),
                    )?,
                );
                apply_union(
                    manager,
                    rec,
                    down_result.borrowed(),
                    right_result.borrowed(),
                )?
            }
        }
    } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair — emit the source value, taken from
        // the universe and matched against the relation. The set and universe
        // are not descended here; the universe is descended at the paired write
        // level.
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let universe_node = match manager.get_node(&universe) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("universe must have as many levels as meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);
        let universe_value = universe_node.get_value();
        let (_universe_down, universe_right) = collect_children(universe_node);

        match universe_value.cmp(rel_value) {
            Ordering::Less => {
                // This universe value cannot be read; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel.borrowed(),
                    meta.borrowed(),
                    universe_right,
                )?
            }
            Ordering::Greater => {
                // This relation value is not in the universe; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel_right,
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
            Ordering::Equal => {
                // meta_down is the paired write level (4); the universe is kept
                // so that its down-branch is descended there.
                let (down_result, rest) = rec.quaternary(
                    apply_relational_predecessor,
                    manager,
                    (set.borrowed(), rel_down, meta_down, universe.borrowed()),
                    (
                        set.borrowed(),
                        rel_right,
                        meta.borrowed(),
                        universe.borrowed(),
                    ),
                )?;
                let node = EdgeDropGuard::new(
                    manager,
                    if manager
                        .get_node(&down_result)
                        .is_terminal(&LDDTerminal::Empty)
                    {
                        manager.get_terminal(LDDTerminal::Empty)?
                    } else {
                        make_node(
                            manager,
                            universe_value,
                            manager.clone_edge(&down_result),
                            manager.get_terminal(LDDTerminal::Empty)?,
                        )?
                    },
                );
                apply_union(manager, rec, node.borrowed(), rest.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_of_pair_value() {
        // 4: write half of a read+write pair — consume the target value, which
        // must appear in both set and rel; nothing is emitted. The source value
        // was already produced at the paired read level, so descend the
        // universe here.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // This target value is not written by the relation; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set_right,
                    rel.borrowed(),
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
            Ordering::Greater => {
                // This relation value is not in the set; skip it.
                apply_relational_predecessor(
                    manager,
                    rec,
                    set.borrowed(),
                    rel_right,
                    meta.borrowed(),
                    universe.borrowed(),
                )?
            }
            Ordering::Equal => {
                let universe_node = match manager.get_node(&universe) {
                    Node::Inner(n) => n.borrow(),
                    _ => unreachable!("universe must have as many levels as meta"),
                };
                let (universe_down, _universe_right) = collect_children(universe_node);
                let (down_result, right_result) = rec.quaternary(
                    apply_relational_predecessor,
                    manager,
                    (set_down, rel_down, meta_down, universe_down),
                    (
                        set.borrowed(),
                        rel_right,
                        meta.borrowed(),
                        universe.borrowed(),
                    ),
                )?;
                apply_union(
                    manager,
                    rec,
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
    set: Borrowed<M::Edge>,
    rel: Borrowed<M::Edge>,
    meta: Borrowed<M::Edge>,
    universe: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    let universe_node = match manager.get_node(&universe) {
        // The right spine of the universe ends at the Empty terminal.
        Node::Terminal(_) => return manager.get_terminal(LDDTerminal::Empty),
        Node::Inner(n) => n.borrow(),
    };
    let universe_value = universe_node.get_value();
    let (universe_down, universe_right) = collect_children(universe_node);

    let down_result = EdgeDropGuard::new(
        manager,
        apply_relational_predecessor(
            manager,
            rec,
            set.borrowed(),
            rel.borrowed(),
            meta.borrowed(),
            universe_down,
        )?,
    );
    let right_result = EdgeDropGuard::new(
        manager,
        relational_predecessor_universe(manager, rec, set, rel, meta, universe_right)?,
    );

    if manager
        .get_node(&down_result)
        .is_terminal(&LDDTerminal::Empty)
    {
        Ok(right_result.into_edge())
    } else {
        make_node(
            manager,
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
pub(crate) fn len<M: LDDManager>(
    manager: &M,
    set: Borrowed<M::Edge>,
    cache: &mut HashMap<NodeID, usize>,
) -> usize {
    match manager.get_node(&set) {
        Node::Terminal(t) => {
            return if *t.borrow() == LDDTerminal::True {
                1
            } else {
                0
            };
        }
        Node::Inner(_) => {}
    }

    let node_id = set.node_id();
    if let Some(n) = cache.get(&node_id) {
        return *n;
    }

    // Walk the right spine, summing the sizes of all down-branches. The right
    // spine is terminated by the Empty terminal.
    let mut result = 0;
    let mut current = EdgeDropGuard::new(manager, manager.clone_edge(&set));
    loop {
        let (down, right) = {
            let node = match manager.get_node(&current) {
                Node::Inner(n) => n.borrow(),
                _ => unreachable!("the right spine ends at the Empty terminal"),
            };
            let (down, right) = collect_children(node);
            (
                EdgeDropGuard::new(manager, manager.clone_edge(&down)),
                EdgeDropGuard::new(manager, manager.clone_edge(&right)),
            )
        };
        result += len(manager, down.borrowed(), cache);
        if manager.get_node(&right).is_terminal(&LDDTerminal::Empty) {
            break;
        }
        current = right;
    }

    cache.insert(node_id, result);
    result
}

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<'_, E>, Borrowed<'_, E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = node.children();
    let f_down = it.next().unwrap();
    let f_right = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (f_down, f_right)
}

/// Create a node in `manager` if necessary
#[inline(always)]
fn make_node<M: LDDManager>(
    manager: &M,
    value: &<M as Manager>::InnerNodeValue,
    down: M::Edge,
    right: M::Edge,
) -> AllocResult<M::Edge> {
    oxidd_core::LevelView::get_or_insert(
        &mut manager.level(0),
        InnerNode::new(0, [down, right], value.clone()),
    )
}
