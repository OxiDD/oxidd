//! Recursive apply algorithms

use std::hash::BuildHasher;

use bitvec::vec::BitVec;

use oxidd_core::{
    function::{
        BooleanFunction, BooleanFunctionQuant, BooleanOperator, EdgeOfFunc, Function, FunctionSubst,
    },
    util::{
        AllocResult, Borrowed, EdgeDropGuard, EdgeVecDropGuard, OptBool, SatCountCache,
        SatCountNumber,
    },
    ApplyCache, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager, Node, NodeID, Tag,
};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use crate::{complement_edge::add_literal_to_cube, stat};
use crate::{
    complement_edge::is_false,
    recursor::{Recursor, SequentialRecursor},
};

use super::collect_cofactors;
use super::get_terminal;
use super::not;
use super::not_owned;
use super::reduce;
use super::BCDDOp;
use super::BCDDTerminal;
use super::EdgeTag;
use super::NodesOrDone;
#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
///
/// Using `Borrowed<M::Edge>` instead of `&M::Edge` means that we actually
/// pass the edge by value, which saves a few indirections.
fn apply_bin<M, R: Recursor<M>, const OP: u8>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_bin::<M, _, OP>(manager, SequentialRecursor, f, g);
    }
    stat!(call OP);

    let (op, f, fnode, g, gnode) = if OP == BCDDOp::And as u8 {
        match super::terminal_and(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => {
                (BCDDOp::And, f.borrowed(), fnode, g.borrowed(), gnode)
            }
            // `And` is commutative, hence we swap `f` and `g` in the apply
            // cache key if `f > g` to have a unique representation of the set
            // `{f, g}`.
            NodesOrDone::Nodes(fnode, gnode) => {
                (BCDDOp::And, g.borrowed(), gnode, f.borrowed(), fnode)
            }
            NodesOrDone::Done(h) => return Ok(h.into_edge()),
        }
    } else {
        assert_eq!(OP, BCDDOp::Xor as u8);
        match super::terminal_xor(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => {
                (BCDDOp::Xor, f.borrowed(), fnode, g.borrowed(), gnode)
            }
            NodesOrDone::Nodes(fnode, gnode) => {
                (BCDDOp::Xor, g.borrowed(), gnode, f.borrowed(), fnode)
            }
            NodesOrDone::Done(h) => {
                return Ok(h.into_edge());
            }
        }
    };

    // Query apply cache
    stat!(cache_query OP);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, op, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit OP);
        return Ok(h);
    }

    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };

    let (t, e) = rec.binary(apply_bin::<M, R, OP>, manager, (ft, gt), (fe, ge))?;

    let h = reduce(manager, level, t.into_edge(), e.into_edge(), op)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, op, &[f, g], h.borrowed());

    Ok(h)
}

/// Shorthand for `apply_bin_rec::<M, R, { BCDDOp::And as u8 }>(manager, f, g)`
#[inline(always)]
fn apply_and<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    apply_bin::<M, R, { BCDDOp::And as u8 }>(manager, rec, f, g)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
fn apply_ite<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_ite(manager, SequentialRecursor, f, g, h);
    }
    stat!(call BCDDOp::Ite);

    // Terminal cases
    let gu = g.with_tag(EdgeTag::None); // untagged
    let hu = h.with_tag(EdgeTag::None);
    if gu == hu {
        return Ok(if g.tag() == h.tag() {
            manager.clone_edge(&g)
        } else {
            not_owned(apply_bin::<M, R, { BCDDOp::Xor as u8 }>(
                manager, rec, f, g,
            )?) // f ↔ g
        });
    }
    let fu = f.with_tag(EdgeTag::None);
    if fu == gu {
        return if f.tag() == g.tag() {
            Ok(not_owned(apply_and(manager, rec, not(&f), not(&h))?)) // f ∨ h
        } else {
            apply_and(manager, rec, not(&f), h) // f < h
        };
    }
    if fu == hu {
        return if f.tag() == h.tag() {
            apply_and(manager, rec, f, g)
        } else {
            // f → g = ¬f ∨ g = ¬(f ∧ ¬g)
            Ok(not_owned(apply_and(manager, rec, f, not(&g))?))
        };
    }
    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(_) => {
            return Ok(manager.clone_edge(&*if f.tag() == EdgeTag::None { g } else { h }))
        }
    };
    let (gnode, hnode) = match (manager.get_node(&g), manager.get_node(&h)) {
        (Node::Inner(gn), Node::Inner(hn)) => (gn, hn),
        (Node::Terminal(_), Node::Inner(_)) => {
            return if g.tag() == EdgeTag::None {
                // f ∨ h
                Ok(not_owned(apply_and(manager, rec, not(&f), not(&h))?))
            } else {
                apply_and(manager, rec, not(&f), h) // f < h
            };
        }
        (_gnode, Node::Terminal(_)) => {
            debug_assert!(_gnode.is_inner());
            return if h.tag() == EdgeTag::None {
                Ok(not_owned(apply_and(manager, rec, f, not(&g))?)) // f → g
            } else {
                apply_and(manager, rec, f, g)
            };
        }
    };

    // Query apply cache
    stat!(cache_query BCDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        BCDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit BCDDOp::Ite);
        return Ok(res);
    }

    // Get the top-most level of the three
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = std::cmp::min(std::cmp::min(flevel, glevel), hlevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (ht, he) = if hlevel == level {
        collect_cofactors(h.tag(), hnode)
    } else {
        (h.borrowed(), h.borrowed())
    };

    let (t, e) = rec.ternary(apply_ite, manager, (ft, gt, ht), (fe, ge, he))?;
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), BCDDOp::Ite)?;

    manager
        .apply_cache()
        .add(manager, BCDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

/// Prepare a substitution
///
/// The result is a vector that maps levels to replacement functions. The levels
/// below the lowest variable (of `vars`) are ignored. Levels above which are
/// not referenced from `vars` are mapped to the function representing the
/// variable at that level. The latter is the reason why we return the owned
/// edges.
fn substitute_prepare<'a, M>(
    manager: &'a M,
    pairs: impl Iterator<Item = (Borrowed<'a, M::Edge>, Borrowed<'a, M::Edge>)>,
) -> AllocResult<EdgeVecDropGuard<'a, M>>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>,
    M::Edge: 'a,
    M::InnerNode: HasLevel,
{
    let mut subst = Vec::with_capacity(manager.num_levels() as usize);
    for (v, r) in pairs {
        let level = super::var_level(manager, v) as usize;
        if level >= subst.len() {
            subst.resize_with(level + 1, || None);
        }
        debug_assert!(
            subst[level].is_none(),
            "Variable at level {level} occurs twice in the substitution, but a \
            substitution should be a mapping from variables to replacement \
            functions"
        );
        subst[level] = Some(r);
    }

    let mut res = EdgeVecDropGuard::new(manager, Vec::with_capacity(subst.len()));
    for (level, e) in subst.into_iter().enumerate() {
        use oxidd_core::LevelView;

        res.push(if let Some(e) = e {
            manager.clone_edge(&e)
        } else {
            let t = EdgeDropGuard::new(manager, get_terminal(manager, true));
            let e = EdgeDropGuard::new(manager, get_terminal(manager, false));
            manager
                .level(level as LevelNo)
                .get_or_insert(InnerNode::new(
                    level as LevelNo,
                    [t.into_edge(), e.into_edge()],
                ))?
        });
    }

    Ok(res)
}

fn substitute<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    subst: &[M::Edge],
    cache_id: u32,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return substitute(manager, SequentialRecursor, f, subst, cache_id);
    }
    stat!(call BCDDOp::Substitute);

    let Node::Inner(node) = manager.get_node(&f) else {
        return Ok(manager.clone_edge(&f));
    };
    let level = node.level();
    if level as usize >= subst.len() {
        return Ok(manager.clone_edge(&f));
    }

    // Query apply cache
    stat!(cache_query BCDDOp::Substitute);
    if let Some(h) = manager.apply_cache().get_with_numeric(
        manager,
        BCDDOp::Substitute,
        &[f.borrowed()],
        &[cache_id],
    ) {
        stat!(cache_hit BCDDOp::Substitute);
        return Ok(h);
    }

    let (t, e) = collect_cofactors(f.tag(), node);
    let (t, e) = rec.subst(
        substitute,
        manager,
        (t, subst, cache_id),
        (e, subst, cache_id),
    )?;
    let res = apply_ite(
        manager,
        rec,
        subst[level as usize].borrowed(),
        t.borrowed(),
        e.borrowed(),
    )?;

    // Insert into apply cache
    manager.apply_cache().add_with_numeric(
        manager,
        BCDDOp::Substitute,
        &[f.borrowed()],
        &[cache_id],
        res.borrowed(),
    );

    Ok(res)
}

/// Result of [`restrict_inner()`]
enum RestrictInnerResult<'a, M: Manager> {
    Done(M::Edge),
    Rec {
        vars: Borrowed<'a, M::Edge>,
        f: Borrowed<'a, M::Edge>,
        f_neg: bool,
        fnode: &'a M::InnerNode,
    },
}

/// Tail-recursive part of [`restrict()`]. `f` is the function of which the
/// variables should be restricted to constant values according to `vars`.
///
/// Invariant: `f` points to `fnode` at `flevel`, `vars` points to `vnode`
///
/// We expose this, because it can be reused for the multi-threaded version.
#[inline]
#[allow(clippy::too_many_arguments)]
fn restrict_inner<'a, M>(
    manager: &'a M,
    f: Borrowed<'a, M::Edge>,
    f_neg: bool,
    fnode: &'a M::InnerNode,
    flevel: LevelNo,
    vars: Borrowed<'a, M::Edge>,
    vars_neg: bool,
    vnode: &'a M::InnerNode,
) -> RestrictInnerResult<'a, M>
where
    M: Manager<EdgeTag = EdgeTag>,
    M::InnerNode: HasLevel,
{
    debug_assert!(std::ptr::eq(manager.get_node(&f).unwrap_inner(), fnode));
    debug_assert_eq!(fnode.level(), flevel);
    debug_assert!(std::ptr::eq(manager.get_node(&vars).unwrap_inner(), vnode));

    let vlevel = vnode.level();
    if vlevel > flevel {
        // f above vars
        return RestrictInnerResult::Rec {
            vars: vars.edge_with_tag(if vars_neg {
                EdgeTag::Complemented
            } else {
                EdgeTag::None
            }),
            f,
            f_neg,
            fnode,
        };
    }

    let (f, complement) = 'ret_f: {
        let vt = vnode.child(0);
        if vlevel < flevel {
            // vars above f
            if let Node::Inner(n) = manager.get_node(&vt) {
                debug_assert!(
                    manager.get_node(&vnode.child(1)).is_any_terminal(),
                    "vars must be a conjunction of literals (but both children are non-terminals)"
                );

                debug_assert_eq!(
                    vnode.child(1).tag(),
                    if vars_neg {
                        EdgeTag::None
                    } else {
                        EdgeTag::Complemented
                    },
                    "vars must be a conjunction of literals (but is of shape ¬x ∨ {}φ)",
                    if vars_neg { "¬" } else { "" }
                );
                // shape: x ∧ if vars_neg { ¬φ } else { φ }
                let vars_neg = vars_neg ^ (vt.tag() == EdgeTag::Complemented);
                return restrict_inner(manager, f, f_neg, fnode, flevel, vt, vars_neg, n);
            }
            // then edge of vars edge points to ⊤
            if vars_neg {
                // shape ¬x ∧ φ
                let ve = vnode.child(1);
                if let Node::Inner(n) = manager.get_node(&ve) {
                    // `vars` is currently negated, hence `!=`
                    let vars_neg = ve.tag() != EdgeTag::Complemented;
                    return restrict_inner(manager, f, f_neg, fnode, flevel, ve, vars_neg, n);
                }
                // shape ¬x
            } else {
                debug_assert!(
                    manager.get_node(&vnode.child(1)).is_any_terminal(),
                    "vars must be a conjunction of literals (but is of shape x ∨ φ)"
                );
                // shape x
            }
            // `vars` is a single variable above `f` ⇒ return `f`
            break 'ret_f (f, f_neg);
        }

        debug_assert_eq!(vlevel, flevel);
        // top var at the level of f ⇒ select accordingly
        let (f, vars, vars_neg, vnode) = if let Node::Inner(n) = manager.get_node(&vt) {
            debug_assert!(
                manager.get_node(&vnode.child(1)).is_any_terminal(),
                "vars must be a conjunction of literals (but both children are non-terminals)"
            );

            debug_assert_eq!(
                vnode.child(1).tag(),
                if vars_neg {
                    EdgeTag::None
                } else {
                    EdgeTag::Complemented
                },
                "vars must be a conjunction of literals (but is of shape ¬x ∨ {}φ)",
                if vars_neg { "¬" } else { "" }
            );
            // shape: x ∧ if vars_neg { ¬φ } else { φ } ⇒ select then branch
            let vars_neg = vars_neg ^ (vt.tag() == EdgeTag::Complemented);
            (fnode.child(0), vt, vars_neg, n)
        } else {
            // then edge of vars edge points to ⊤

            if !vars_neg {
                debug_assert!(
                    manager.get_node(&vnode.child(1)).is_any_terminal(),
                    "vars must be a conjunction of literals (but is of shape x ∨ φ)"
                );

                // shape x ⇒ select then branch
                let f = fnode.child(0);
                let f_neg = f_neg ^ (f.tag() == EdgeTag::Complemented);
                break 'ret_f (f, f_neg);
            }

            // shape ¬x ∧ φ ⇒ select else branch
            let f = fnode.child(1);
            let ve = vnode.child(1);
            if let Node::Inner(n) = manager.get_node(&ve) {
                // `vars` is currently negated, hence `!=`
                let vars_neg = ve.tag() != EdgeTag::Complemented;
                (f, ve, vars_neg, n)
            } else {
                // shape `¬x` ⇒ return
                let f_neg = f_neg ^ (f.tag() == EdgeTag::Complemented);
                break 'ret_f (f, f_neg);
            }
        };

        let f_neg = f_neg ^ (f.tag() == EdgeTag::Complemented);
        if let Node::Inner(fnode) = manager.get_node(&f) {
            let flevel = fnode.level();
            return restrict_inner(manager, f, f_neg, fnode, flevel, vars, vars_neg, vnode);
        }
        (f, f_neg)
    };

    RestrictInnerResult::Done(manager.clone_edge(&f).with_tag_owned(if complement {
        EdgeTag::Complemented
    } else {
        EdgeTag::None
    }))
}

fn restrict<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return restrict(manager, SequentialRecursor, f, vars);
    }
    stat!(call BCDDOp::Restrict);

    let (Node::Inner(fnode), Node::Inner(vnode)) = (manager.get_node(&f), manager.get_node(&vars))
    else {
        return Ok(manager.clone_edge(&f));
    };

    let inner_res = {
        let f_neg = f.tag() == EdgeTag::Complemented;
        let flevel = fnode.level();
        let vars_neg = vars.tag() == EdgeTag::Complemented;
        restrict_inner(manager, f, f_neg, fnode, flevel, vars, vars_neg, vnode)
    };
    match inner_res {
        RestrictInnerResult::Done(result) => Ok(result),
        RestrictInnerResult::Rec {
            vars,
            f,
            f_neg,
            fnode,
        } => {
            // f above top-most restrict variable
            let f_untagged = f.with_tag(EdgeTag::None);
            let f_tag = if f_neg {
                EdgeTag::Complemented
            } else {
                EdgeTag::None
            };

            // Query apply cache
            stat!(cache_query BCDDOp::Restrict);
            if let Some(result) = manager.apply_cache().get(
                manager,
                BCDDOp::Restrict,
                &[f_untagged.borrowed(), vars.borrowed()],
            ) {
                stat!(cache_hit BCDDOp::Restrict);
                let result_tag = result.tag();
                return Ok(result.with_tag_owned(result_tag ^ f_tag));
            }

            let (t, e) = rec.binary(
                restrict,
                manager,
                (fnode.child(0), vars.borrowed()),
                (fnode.child(1), vars.borrowed()),
            )?;

            let result = reduce(
                manager,
                fnode.level(),
                t.into_edge(),
                e.into_edge(),
                BCDDOp::Restrict,
            )?;

            manager.apply_cache().add(
                manager,
                BCDDOp::Restrict,
                &[f_untagged, vars],
                result.borrowed(),
            );

            let result_tag = result.tag();
            Ok(result.with_tag_owned(result_tag ^ f_tag))
        }
    }
}

/// Compute the quantification `Q` over `vars`
///
/// `Q` is one of `BCDDOp::Forall`, `BCDDOp::Exist`, or `BCDDOp::Forall` as
/// `u8`.
fn quant<M, R: Recursor<M>, const Q: u8>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return quant::<M, _, Q>(manager, SequentialRecursor, f, vars);
    }
    let operator = match () {
        _ if Q == BCDDOp::Forall as u8 => BCDDOp::Forall,
        _ if Q == BCDDOp::Exist as u8 => BCDDOp::Exist,
        _ if Q == BCDDOp::Unique as u8 => BCDDOp::Unique,
        _ => unreachable!("invalid quantifier"),
    };

    stat!(call operator);
    // Terminal cases
    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(_) => {
            return Ok(
                if operator != BCDDOp::Unique || manager.get_node(&vars).is_any_terminal() {
                    manager.clone_edge(&f)
                } else {
                    get_terminal(manager, false)
                },
            );
        }
    };
    let flevel = fnode.level();

    let vars = if operator != BCDDOp::Unique {
        // We can ignore all variables above the top-most variable. Removing
        // them before querying the apply cache should increase the hit ratio by
        // a lot.
        crate::set_pop(manager, vars, flevel)
    } else {
        // No need to pop variables here, if the variable is above `fnode`,
        // i.e., does not occur in `f`, then the result is `f ⊕ f ≡ ⊥`. We
        // handle this below.
        vars
    };
    let vnode = match manager.get_node(&vars) {
        Node::Inner(n) => n,
        Node::Terminal(_) => return Ok(manager.clone_edge(&f)),
    };
    let vlevel = vnode.level();
    if operator == BCDDOp::Unique && vlevel < flevel {
        // `vnode` above `fnode`, i.e., the variable does not occur in `f` (see above)
        return Ok(get_terminal(manager, false));
    }
    debug_assert!(flevel <= vlevel);
    let vars = vars.borrowed();

    // Query apply cache
    stat!(cache_query operator);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, operator, &[f.borrowed(), vars.borrowed()])
    {
        stat!(cache_hit operator);
        return Ok(res);
    }

    let (ft, fe) = collect_cofactors(f.tag(), fnode);
    let vt = if vlevel == flevel {
        vnode.child(0)
    } else {
        vars.borrowed()
    };
    let (t, e) = rec.binary(
        quant::<M, R, Q>,
        manager,
        (ft, vt.borrowed()),
        (fe, vt.borrowed()),
    )?;

    let res = if flevel == vlevel {
        match operator {
            BCDDOp::Forall => apply_and(manager, rec, t.borrowed(), e.borrowed())?,
            BCDDOp::Exist => not_owned(apply_and(manager, rec, not(&t), not(&e))?),
            BCDDOp::Unique => {
                apply_bin::<M, R, { BCDDOp::Xor as u8 }>(manager, rec, t.borrowed(), e.borrowed())?
            }
            _ => unreachable!(),
        }
    } else {
        reduce(manager, flevel, t.into_edge(), e.into_edge(), operator)?
    };

    manager
        .apply_cache()
        .add(manager, operator, &[f, vars], res.borrowed());

    Ok(res)
}

/// Recursively apply the binary operator `OP` to `f` and `g` while quantifying
/// `Q` over `vars`. This is more efficient then computing then an apply
/// operation followed by a quantification.
///
/// One example usage is for the relational product, i.e., computing
/// `∃ s, t. S(s) ∧ T(s, t)`, where `S` is a boolean function representing the
/// states and `T` a boolean function representing the transition relation.
///
/// `Q` is one of [`BCDDOp::Forall`], [`BCDDOp::Exist`], or [`BCDDOp::Forall`]
/// as `u8`. We use a `const` parameter `OP` to have specialized version of this
/// function for each operator ([`BCDDOp::And`], [`BCDDOp::Xor`], or
/// specifically [`BCDDOp::UniqueNand`], each as `u8`).
fn apply_quant<'a, M, R: Recursor<M>, const Q: u8, const OP: u8>(
    manager: &'a M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_quant::<M, _, Q, OP>(manager, SequentialRecursor, f, g, vars);
    }

    let operator = const { BCDDOp::from_apply_quant(Q, OP) };

    stat!(call operator);
    // Handle the terminal cases
    let (f, fnode, g, gnode) = if OP == BCDDOp::And as u8 || OP == BCDDOp::UniqueNand as u8 {
        match super::terminal_and(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => (f.borrowed(), fnode, g.borrowed(), gnode),
            // `And` is commutative, hence we swap `f` and `g` in the apply
            // cache key if `f > g` to have a unique representation of the set
            // `{f, g}`.
            NodesOrDone::Nodes(fnode, gnode) => (g.borrowed(), gnode, f.borrowed(), fnode),
            NodesOrDone::Done(h) if OP == BCDDOp::UniqueNand as u8 => {
                return quant::<M, R, Q>(manager, rec, not(&h), vars)
            }
            NodesOrDone::Done(h) => return quant::<M, R, Q>(manager, rec, h.borrowed(), vars),
        }
    } else {
        assert_eq!(OP, BCDDOp::Xor as u8);
        match super::terminal_xor(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => (f.borrowed(), fnode, g.borrowed(), gnode),
            NodesOrDone::Nodes(fnode, gnode) => (g.borrowed(), gnode, f.borrowed(), fnode),
            NodesOrDone::Done(h) => return quant::<M, R, Q>(manager, rec, h.borrowed(), vars),
        }
    };

    let flevel = fnode.level();
    let glevel = gnode.level();
    let min_level = std::cmp::min(fnode.level(), gnode.level());

    let vars = if Q != BCDDOp::Unique as u8 {
        // We can ignore all variables above the top-most variable. Removing
        // them before querying the apply cache should increase the hit ratio by
        // a lot.
        crate::set_pop(manager, vars, min_level)
    } else {
        // No need to pop variables here. If the variable is above `min_level`,
        // i.e., does not occur in `f` or `g`, then the result is `f ⊕ f ≡ ⊥`. We
        // handle this below.
        vars
    };

    let vnode = match manager.get_node(&vars) {
        Node::Inner(n) => n,
        // Empty variable set: just apply operation
        Node::Terminal(_) if OP == BCDDOp::UniqueNand as u8 => {
            return Ok(not_owned(apply_and(manager, rec, f, g)?))
        }
        Node::Terminal(_) => return apply_bin::<M, R, OP>(manager, rec, f, g),
    };

    let vlevel = vnode.level();
    if vlevel < min_level && Q == BCDDOp::Unique as u8 {
        // `vnode` above `fnode` and `gnode`, i.e., the variable does not occur in `f`
        // or `g` (see above)
        return Ok(get_terminal(manager, false));
    }

    if min_level > vlevel {
        // We are beyond the variables to be quantified, so simply apply.
        if OP == BCDDOp::UniqueNand as u8 {
            return Ok(not_owned(apply_and(manager, rec, f, g)?));
        }
        return apply_bin::<M, R, OP>(manager, rec, f, g);
    }

    // Query the cache
    stat!(cache_query operator);
    if let Some(res) = manager.apply_cache().get(
        manager,
        operator,
        &[f.borrowed(), g.borrowed(), vars.borrowed()],
    ) {
        stat!(cache_hit operator);
        return Ok(res);
    }

    let vt = if vlevel == min_level {
        vnode.child(0)
    } else {
        vars.borrowed()
    };

    let (ft, fe) = if flevel <= glevel {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };

    let (gt, ge) = if flevel >= glevel {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };

    let (t, e) = rec.ternary(
        apply_quant::<M, R, Q, OP>,
        manager,
        (ft, gt, vt.borrowed()),
        (fe, ge, vt.borrowed()),
    )?;

    let res = if min_level == vlevel {
        if Q == BCDDOp::Forall as u8 {
            apply_and(manager, rec, t.borrowed(), e.borrowed())?
        } else if Q == BCDDOp::Exist as u8 {
            not_owned(apply_and(manager, rec, not(&t), not(&e))?)
        } else if Q == BCDDOp::Unique as u8 {
            apply_bin::<M, R, { BCDDOp::Xor as u8 }>(manager, rec, t.borrowed(), e.borrowed())?
        } else {
            unreachable!()
        }
    } else {
        reduce(manager, min_level, t.into_edge(), e.into_edge(), operator)?
    };

    manager
        .apply_cache()
        .add(manager, operator, &[f, g, vars], res.borrowed());

    Ok(res)
}

/// Dynamic dispatcher for [`apply_quant()`] and universal/existential
/// quantification
///
/// In contrast to [`apply_quant()`], the operator is not a const but a runtime
/// parameter. `QN` is the "negated" version of `Q`: If `Q` is
/// [`BCDDOp::Forall`] (as u8) then, `QN` is [`BCDDOp::Exist`], and vice versa.
fn apply_quant_dispatch<'a, M, R: Recursor<M>, const Q: u8, const QN: u8>(
    manager: &'a M,
    rec: R,
    op: BooleanOperator,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    use BooleanOperator::*;
    const OA: u8 = BCDDOp::And as u8;
    const OX: u8 = BCDDOp::Xor as u8;

    const {
        assert!(
            (Q == BCDDOp::Forall as u8 && QN == BCDDOp::Exist as u8)
                || (Q == BCDDOp::Exist as u8 && QN == BCDDOp::Forall as u8)
        );
    }

    match op {
        And => apply_quant::<M, R, Q, OA>(manager, rec, f, g, vars),
        Or => {
            let tmp = apply_quant::<M, R, QN, OA>(manager, rec, not(&f), not(&g), vars)?;
            Ok(not_owned(tmp))
        }
        Xor => apply_quant::<M, R, Q, OX>(manager, rec, f, g, vars),
        Equiv => {
            let tmp = apply_quant::<M, R, QN, OX>(manager, rec, f, g, vars)?;
            Ok(not_owned(tmp))
        }
        Nand => {
            let tmp = apply_quant::<M, R, QN, OA>(manager, rec, f, g, vars)?;
            Ok(not_owned(tmp))
        }
        Nor => apply_quant::<M, R, Q, OA>(manager, rec, not(&f), not(&g), vars),
        Imp => {
            let tmp = apply_quant::<M, R, QN, OA>(manager, rec, f, not(&g), vars)?;
            Ok(not_owned(tmp))
        }
        ImpStrict => apply_quant::<M, R, Q, OA>(manager, rec, not(&f), g, vars),
    }
}

/// Dynamic dispatcher for [`apply_quant()`] and unique quantification
///
/// In contrast to [`apply_quant()`], the operator is not a const but a runtime
/// parameter.
fn apply_quant_unique_dispatch<'a, M, R: Recursor<M>>(
    manager: &'a M,
    rec: R,
    op: BooleanOperator,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, BCDDOp>,
    M::InnerNode: HasLevel,
{
    use BooleanOperator::*;
    const Q: u8 = BCDDOp::Unique as u8;
    const OA: u8 = BCDDOp::And as u8;
    const OX: u8 = BCDDOp::Xor as u8;
    const ONA: u8 = BCDDOp::UniqueNand as u8;

    match op {
        And => apply_quant::<M, R, Q, OA>(manager, rec, f, g, vars),
        Or => apply_quant::<M, R, Q, ONA>(manager, rec, not(&f), not(&g), vars),
        Xor => apply_quant::<M, R, Q, OX>(manager, rec, f, g, vars),
        Equiv => apply_quant::<M, R, Q, OX>(manager, rec, not(&f), g, vars),
        Nand => apply_quant::<M, R, Q, ONA>(manager, rec, f, g, vars),
        Nor => apply_quant::<M, R, Q, OA>(manager, rec, not(&f), not(&g), vars),
        Imp => apply_quant::<M, R, Q, ONA>(manager, rec, f, not(&g), vars),
        ImpStrict => apply_quant::<M, R, Q, OA>(manager, rec, not(&f), g, vars),
    }
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasBCDDOpApplyCache<M: Manager>: HasApplyCache<M, BCDDOp> {}
impl<M: Manager + HasApplyCache<M, BCDDOp>> HasBCDDOpApplyCache<M> for M {}

/// Boolean function backed by a complement edge binary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr(transparent)]
pub struct BCDDFunction<F: Function>(F);

impl<F: Function> From<F> for BCDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        BCDDFunction(value)
    }
}

impl<F: Function> BCDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function> FunctionSubst for BCDDFunction<F>
where
    for<'id> F::Manager<'id>:
        Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasBCDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    fn substitute_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        substitution: impl oxidd_core::util::Substitution<
            Var = Borrowed<'a, EdgeOfFunc<'id, Self>>,
            Replacement = Borrowed<'a, EdgeOfFunc<'id, Self>>,
        >,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let subst = substitute_prepare(manager, substitution.pairs())?;
        substitute(manager, rec, edge.borrowed(), &subst, substitution.id())
    }
}

impl<F: Function> BooleanFunction for BCDDFunction<F>
where
    for<'id> F::Manager<'id>:
        Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasBCDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    #[inline]
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let t = get_terminal(manager, true);
        let e = get_terminal(manager, false);
        let edge = manager.add_level(|level| InnerNode::new(level, [t, e]))?;
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        get_terminal(manager, false)
    }
    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        get_terminal(manager, true)
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(manager.clone_edge(edge)))
    }
    #[inline]
    fn not_edge_owned<'id>(
        _manager: &Self::Manager<'id>,
        edge: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(edge))
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_and(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(Self::nor_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(Self::and_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_and(manager, rec, not(lhs), not(rhs))
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BCDDOp::Xor as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(Self::xor_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        Ok(not_owned(apply_and(
            manager,
            rec,
            lhs.borrowed(),
            not(rhs),
        )?))
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_and(manager, rec, not(lhs), rhs.borrowed())
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_ite(
            manager,
            SequentialRecursor,
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }

    #[inline]
    fn sat_count_edge<'id, N: SatCountNumber, S: BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        fn inner<M, N: SatCountNumber, S: BuildHasher>(
            manager: &M,
            e: Borrowed<M::Edge>,
            terminal_val: &N,
            cache: &mut SatCountCache<N, S>,
        ) -> N
        where
            M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>,
        {
            let node = match manager.get_node(&e) {
                Node::Inner(node) => node,
                Node::Terminal(_) => return terminal_val.clone(),
            };

            // query cache
            let node_id = e.node_id();
            if let Some(n) = cache.map.get(&node_id) {
                return n.clone();
            }

            // recursive case
            let mut iter = node.children().map(|c| {
                let n = inner(manager, c.borrowed(), terminal_val, cache);
                match c.tag() {
                    EdgeTag::None => n,
                    EdgeTag::Complemented => {
                        let mut res = terminal_val.clone();
                        res -= &n;
                        res
                    }
                }
            });

            let mut n = iter.next().unwrap();
            n += &iter.next().unwrap();
            debug_assert!(iter.next().is_none());
            n >>= 1u32;
            cache.map.insert(node_id, n.clone());
            n
        }

        // This function does not use the identity `|f| = num_vars - |f'|` to
        // avoid rounding issues
        fn inner_floating<M, N, S>(
            manager: &M,
            e: Borrowed<M::Edge>,
            terminal_val: &N,
            cache: &mut SatCountCache<N, S>,
        ) -> N
        where
            M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>,
            N: SatCountNumber,
            S: BuildHasher,
        {
            let tag = e.tag();
            let node = match manager.get_node(&e) {
                Node::Inner(node) => node,
                Node::Terminal(_) if tag == EdgeTag::None => return terminal_val.clone(),
                Node::Terminal(_) => return N::from(0u32),
            };
            // MSB of NodeIDs is reserved [for us :)]
            let node_id = e.node_id() | ((tag as NodeID) << (NodeID::BITS - 1));
            if let Some(n) = cache.map.get(&node_id) {
                return n.clone();
            }
            let (e0, e1) = collect_cofactors(tag, node);
            let mut n = inner_floating(manager, e0, terminal_val, cache);
            n += &inner_floating(manager, e1, terminal_val, cache);
            n >>= 1u32;
            cache.map.insert(node_id, n.clone());
            n
        }

        cache.clear_if_invalid(manager, vars);

        if N::FLOATING_POINT {
            let mut terminal_val = N::from(1u32);
            let scale_exp = (-N::MIN_EXP) as u32;
            terminal_val <<= if vars >= scale_exp {
                // scale down to increase the precision if we have many variables
                vars - scale_exp
            } else {
                vars
            };
            let mut res = inner_floating(manager, edge.borrowed(), &terminal_val, cache);
            if vars >= scale_exp {
                res <<= scale_exp; // scale up again
            }
            res
        } else {
            let mut terminal_val = N::from(1u32);
            terminal_val <<= vars;
            let n = inner(manager, edge.borrowed(), &terminal_val, cache);
            match edge.tag() {
                EdgeTag::None => n,
                EdgeTag::Complemented => {
                    terminal_val -= &n;
                    terminal_val
                }
            }
        }
    }

    #[inline]
    fn pick_cube_edge<'id, 'a, I>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
    {
        #[inline] // this function is tail-recursive
        fn inner<M: Manager<EdgeTag = EdgeTag>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            cube: &mut [OptBool],
            mut choice: impl FnMut(&M, &M::Edge, LevelNo) -> bool,
        ) where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return;
            };
            let tag = edge.tag();
            let level = node.level();
            let (t, e) = collect_cofactors(tag, node);
            let c = if is_false(manager, &t) {
                false
            } else if is_false(manager, &e) {
                true
            } else {
                choice(manager, &edge, level)
            };
            cube[level as usize] = OptBool::from(c);
            inner(manager, if c { t } else { e }, cube, choice);
        }

        let order = order.into_iter();
        debug_assert!(
            order.len() == 0 || order.len() == manager.num_levels() as usize,
            "order must be empty or contain all variables"
        );

        if manager.get_node(edge).is_any_terminal() {
            return match edge.tag() {
                EdgeTag::None => Some(vec![OptBool::None; manager.num_levels() as usize]),
                EdgeTag::Complemented => None,
            };
        }

        let mut cube = vec![OptBool::None; manager.num_levels() as usize];
        inner(manager, edge.borrowed(), &mut cube, choice);

        Some(if order.len() == 0 {
            cube
        } else {
            order
                .map(|e| cube[manager.get_node(e).unwrap_inner().level() as usize])
                .collect()
        })
    }

    #[inline]
    fn pick_cube_dd_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        fn inner<M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            mut choice: impl FnMut(&M, &M::Edge, LevelNo) -> bool,
        ) -> AllocResult<M::Edge>
        where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return Ok(manager.clone_edge(&edge));
            };

            let (t, e) = collect_cofactors(edge.tag(), node);
            let level = node.level();
            let c = if is_false(manager, &t) {
                false
            } else if is_false(manager, &e) {
                true
            } else {
                choice(manager, &edge, level)
            };

            let sub = inner(manager, if c { t } else { e }, choice)?;
            add_literal_to_cube(manager, sub, level, c)
        }

        inner(manager, edge.borrowed(), choice)
    }

    fn pick_cube_dd_set_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        literal_set: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        fn inner<M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            literal_set: Borrowed<M::Edge>,
        ) -> AllocResult<M::Edge>
        where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return Ok(manager.clone_edge(&edge));
            };
            let level = node.level();

            let literal_set = crate::set_pop(manager, literal_set, level);
            let (literal_set, c) = match manager.get_node(&literal_set) {
                Node::Inner(node) if node.level() == level => {
                    let (t, e) = collect_cofactors(literal_set.tag(), node);
                    if is_false(manager, &e) {
                        (e, true)
                    } else {
                        (t, false)
                    }
                }
                _ => (literal_set, false),
            };

            let (t, e) = collect_cofactors(edge.tag(), node);
            let c = if is_false(manager, &t) {
                false
            } else if is_false(manager, &e) {
                true
            } else {
                c
            };

            let sub = inner(manager, if c { t } else { e }, literal_set)?;
            add_literal_to_cube(manager, sub, level, c)
        }

        inner(manager, edge.borrowed(), literal_set.borrowed())
    }

    #[inline]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
    ) -> bool {
        let mut values = BitVec::new();
        values.resize(manager.num_levels() as usize, false);
        for (edge, val) in args {
            let node = manager
                .get_node(&edge)
                .expect_inner("edges in `args` must refer to inner nodes");
            values.set(node.level() as usize, val);
        }

        #[inline] // this function is tail-recursive
        fn inner<M>(manager: &M, edge: Borrowed<M::Edge>, complement: bool, values: BitVec) -> bool
        where
            M: Manager<EdgeTag = EdgeTag>,
            M::InnerNode: HasLevel,
        {
            let complement = complement ^ (edge.tag() == EdgeTag::Complemented);
            match manager.get_node(&edge) {
                Node::Inner(node) => {
                    let edge = node.child((!values[node.level() as usize]) as usize);
                    inner(manager, edge, complement, values)
                }
                Node::Terminal(_) => !complement,
            }
        }

        inner(manager, edge.borrowed(), false, values)
    }
}

impl<F: Function> BooleanFunctionQuant for BCDDFunction<F>
where
    for<'id> F::Manager<'id>:
        Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasBCDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    #[inline]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        restrict(manager, rec, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        quant::<_, _, { BCDDOp::Forall as u8 }>(manager, rec, root.borrowed(), vars.borrowed())
    }
    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        quant::<_, _, { BCDDOp::Exist as u8 }>(manager, rec, root.borrowed(), vars.borrowed())
    }
    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        quant::<_, _, { BCDDOp::Unique as u8 }>(manager, rec, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn apply_forall_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_quant_dispatch::<_, _, { BCDDOp::Forall as u8 }, { BCDDOp::Exist as u8 }>(
            manager,
            SequentialRecursor,
            op,
            lhs.borrowed(),
            rhs.borrowed(),
            vars.borrowed(),
        )
    }
    #[inline]
    fn apply_exist_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_quant_dispatch::<_, _, { BCDDOp::Exist as u8 }, { BCDDOp::Forall as u8 }>(
            manager,
            SequentialRecursor,
            op,
            lhs.borrowed(),
            rhs.borrowed(),
            vars.borrowed(),
        )
    }
    #[inline]
    fn apply_unique_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let (lhs, rhs, vars) = (lhs.borrowed(), rhs.borrowed(), vars.borrowed());
        apply_quant_unique_dispatch(manager, rec, op, lhs, rhs, vars)
    }
}

impl<F: Function, T: Tag> DotStyle<T> for BCDDFunction<F> {}

#[cfg(feature = "multi-threading")]
pub mod mt {
    use oxidd_core::HasWorkers;

    use crate::recursor::mt::ParallelRecursor;

    use super::*;

    /// Boolean function backed by a complement edge binary decision diagram
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
    #[repr(transparent)]
    pub struct BCDDFunctionMT<F: Function>(F);

    impl<F: Function> From<F> for BCDDFunctionMT<F> {
        #[inline(always)]
        fn from(value: F) -> Self {
            BCDDFunctionMT(value)
        }
    }

    impl<F: Function> BCDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: HasWorkers,
    {
        /// Convert `self` into the underlying [`Function`]
        #[inline(always)]
        pub fn into_inner(self) -> F {
            self.0
        }
    }

    impl<F: Function> FunctionSubst for BCDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
            + HasBCDDOpApplyCache<F::Manager<'id>>
            + HasWorkers,
        for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
        for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
    {
        fn substitute_edge<'id, 'a>(
            manager: &'a Self::Manager<'id>,
            edge: &'a EdgeOfFunc<'id, Self>,
            substitution: impl oxidd_core::util::Substitution<
                Var = Borrowed<'a, EdgeOfFunc<'id, Self>>,
                Replacement = Borrowed<'a, EdgeOfFunc<'id, Self>>,
            >,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let subst = substitute_prepare(manager, substitution.pairs())?;
            let edge = edge.borrowed();
            let cache_id = substitution.id();
            let rec = ParallelRecursor::new(manager);
            substitute(manager, rec, edge, &subst, cache_id)
        }
    }

    impl<F: Function> BooleanFunction for BCDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
            + HasBCDDOpApplyCache<F::Manager<'id>>
            + HasWorkers,
        for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
        for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
    {
        #[inline]
        fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
            let t = get_terminal(manager, true);
            let e = get_terminal(manager, false);
            let edge = manager.add_level(|level| InnerNode::new(level, [t, e]))?;
            Ok(Self::from_edge(manager, edge))
        }

        #[inline]
        fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            get_terminal(manager, false)
        }
        #[inline]
        fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            get_terminal(manager, true)
        }

        #[inline]
        fn not_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            Ok(not_owned(manager.clone_edge(edge)))
        }
        #[inline]
        fn not_edge_owned<'id>(
            _manager: &Self::Manager<'id>,
            edge: EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            Ok(not_owned(edge))
        }

        #[inline]
        fn and_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_and(manager, ParallelRecursor::new(manager), lhs, rhs)
        }
        #[inline]
        fn or_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (nl, nr) = (not(lhs), not(rhs));
            let nor = apply_and(manager, ParallelRecursor::new(manager), nl, nr)?;
            Ok(not_owned(nor))
        }
        #[inline]
        fn nand_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let and = apply_and(manager, ParallelRecursor::new(manager), lhs, rhs)?;
            Ok(not_owned(and))
        }
        #[inline]
        fn nor_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (nl, nr) = (not(lhs), not(rhs));
            apply_and(manager, ParallelRecursor::new(manager), nl, nr)
        }
        #[inline]
        fn xor_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BCDDOp::Xor as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn equiv_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            Ok(not_owned(apply_bin::<_, _, { BCDDOp::Xor as u8 }>(
                manager, rec, lhs, rhs,
            )?))
        }
        #[inline]
        fn imp_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            // a → b ≡ ¬a ∨ b ≡ ¬(a ∧ ¬b)
            let (lhs, nr) = (lhs.borrowed(), not(rhs));
            let not_imp = apply_and(manager, ParallelRecursor::new(manager), lhs, nr)?;
            Ok(not_owned(not_imp))
        }
        #[inline]
        fn imp_strict_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (nl, rhs) = (not(lhs), rhs.borrowed());
            apply_and(manager, ParallelRecursor::new(manager), nl, rhs)
        }

        #[inline]
        fn ite_edge<'id>(
            manager: &Self::Manager<'id>,
            f: &EdgeOfFunc<'id, Self>,
            g: &EdgeOfFunc<'id, Self>,
            h: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (f, g, h) = (f.borrowed(), g.borrowed(), h.borrowed());
            apply_ite(manager, ParallelRecursor::new(manager), f, g, h)
        }

        #[inline]
        fn sat_count_edge<'id, N: SatCountNumber, S: std::hash::BuildHasher>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            vars: LevelNo,
            cache: &mut SatCountCache<N, S>,
        ) -> N {
            BCDDFunction::<F>::sat_count_edge(manager, edge, vars, cache)
        }

        #[inline]
        fn pick_cube_edge<'id, 'a, I>(
            manager: &'a Self::Manager<'id>,
            edge: &'a EdgeOfFunc<'id, Self>,
            order: impl IntoIterator<IntoIter = I>,
            choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
        ) -> Option<Vec<OptBool>>
        where
            I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
        {
            BCDDFunction::<F>::pick_cube_edge(manager, edge, order, choice)
        }
        #[inline]
        fn pick_cube_dd_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            BCDDFunction::<F>::pick_cube_dd_edge(manager, edge, choice)
        }
        #[inline]
        fn pick_cube_dd_set_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            literal_set: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            BCDDFunction::<F>::pick_cube_dd_set_edge(manager, edge, literal_set)
        }

        #[inline]
        fn eval_edge<'id, 'a>(
            manager: &'a Self::Manager<'id>,
            edge: &'a EdgeOfFunc<'id, Self>,
            args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
        ) -> bool {
            BCDDFunction::<F>::eval_edge(manager, edge, args)
        }
    }

    impl<F: Function> BooleanFunctionQuant for BCDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
            + HasBCDDOpApplyCache<F::Manager<'id>>
            + HasWorkers,
        for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
        for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
    {
        #[inline]
        fn restrict_edge<'id>(
            manager: &Self::Manager<'id>,
            root: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (root, vars) = (root.borrowed(), vars.borrowed());
            restrict(manager, ParallelRecursor::new(manager), root, vars)
        }

        #[inline]
        fn forall_edge<'id>(
            manager: &Self::Manager<'id>,
            root: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (root, vars) = (root.borrowed(), vars.borrowed());
            let rec = ParallelRecursor::new(manager);
            quant::<_, _, { BCDDOp::Forall as u8 }>(manager, rec, root, vars)
        }
        #[inline]
        fn exist_edge<'id>(
            manager: &Self::Manager<'id>,
            root: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (root, vars) = (root.borrowed(), vars.borrowed());
            let rec = ParallelRecursor::new(manager);
            quant::<_, _, { BCDDOp::Exist as u8 }>(manager, rec, root, vars)
        }
        #[inline]
        fn unique_edge<'id>(
            manager: &Self::Manager<'id>,
            root: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (root, vars) = (root.borrowed(), vars.borrowed());
            let rec = ParallelRecursor::new(manager);
            quant::<_, _, { BCDDOp::Unique as u8 }>(manager, rec, root, vars)
        }

        #[inline]
        fn apply_forall_edge<'id>(
            manager: &Self::Manager<'id>,
            op: BooleanOperator,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs, vars) = (lhs.borrowed(), rhs.borrowed(), vars.borrowed());
            apply_quant_dispatch::<_, _, { BCDDOp::Forall as u8 }, { BCDDOp::Exist as u8 }>(
                manager,
                ParallelRecursor::new(manager),
                op,
                lhs,
                rhs,
                vars,
            )
        }
        #[inline]
        fn apply_exist_edge<'id>(
            manager: &Self::Manager<'id>,
            op: BooleanOperator,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs, vars) = (lhs.borrowed(), rhs.borrowed(), vars.borrowed());
            apply_quant_dispatch::<_, _, { BCDDOp::Exist as u8 }, { BCDDOp::Forall as u8 }>(
                manager,
                ParallelRecursor::new(manager),
                op,
                lhs,
                rhs,
                vars,
            )
        }
        #[inline]
        fn apply_unique_edge<'id>(
            manager: &Self::Manager<'id>,
            op: BooleanOperator,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs, vars) = (lhs.borrowed(), rhs.borrowed(), vars.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_quant_unique_dispatch(manager, rec, op, lhs, rhs, vars)
        }
    }

    impl<F: Function, T: Tag> DotStyle<T> for BCDDFunctionMT<F> {}
}
