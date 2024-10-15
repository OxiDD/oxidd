//! Recursive apply algorithms

use std::borrow::Borrow;
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
    ApplyCache, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager, Node, Tag,
};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use crate::recursor::{Recursor, SequentialRecursor};
use crate::stat;

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{collect_children, reduce, BDDOp, BDDTerminal, Operation};

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the 'not' operator to `f`
fn apply_not<M, R: Recursor<M>>(manager: &M, rec: R, f: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_not(manager, SequentialRecursor, f);
    }
    stat!(call BDDOp::Not);

    let node = match manager.get_node(&f) {
        Node::Inner(node) => node,
        Node::Terminal(t) => return Ok(manager.get_terminal(!*t.borrow()).unwrap()),
    };

    // Query apply cache
    stat!(cache_query BDDOp::Not);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, BDDOp::Not, &[f.borrowed()])
    {
        stat!(cache_hit BDDOp::Not);
        return Ok(h);
    }

    let (ft, fe) = collect_children(node);
    let level = node.level();

    let (t, e) = rec.unary(apply_not, manager, ft, fe)?;
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), BDDOp::Not)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, BDDOp::Not, &[f.borrowed()], h.borrowed());

    Ok(h)
}

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M, R: Recursor<M>, const OP: u8>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_bin::<M, _, OP>(manager, SequentialRecursor, f, g);
    }
    stat!(call OP);

    let (operator, op1, op2) = match super::terminal_bin::<M, OP>(manager, &f, &g) {
        Operation::Binary(o, op1, op2) => (o, op1, op2),
        Operation::Not(f) => {
            return apply_not(manager, rec, f);
        }
        Operation::Done(h) => return Ok(h),
    };

    // Query apply cache
    stat!(cache_query OP);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, operator, &[op1.borrowed(), op2.borrowed()])
    {
        stat!(cache_hit OP);
        return Ok(h);
    }

    let fnode = manager.get_node(&f).unwrap_inner();
    let gnode = manager.get_node(&g).unwrap_inner();
    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_children(gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };

    let (t, e) = rec.binary(apply_bin::<M, R, OP>, manager, (ft, gt), (fe, ge))?;
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), operator)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, operator, &[op1, op2], h.borrowed());

    Ok(h)
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
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    use BDDTerminal::*;
    if rec.should_switch_to_sequential() {
        return apply_ite(manager, SequentialRecursor, f, g, h);
    }
    stat!(call BDDOp::Ite);

    // Terminal cases
    if g == h {
        return Ok(manager.clone_edge(&g));
    }
    if f == g {
        return apply_bin::<M, R, { BDDOp::Or as u8 }>(manager, rec, f, h);
    }
    if f == h {
        return apply_bin::<M, R, { BDDOp::And as u8 }>(manager, rec, f, g);
    }
    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(t) => {
            return Ok(manager.clone_edge(&*if *t.borrow() == True { g } else { h }))
        }
    };
    let (gnode, hnode) = match (manager.get_node(&g), manager.get_node(&h)) {
        (Node::Inner(gn), Node::Inner(hn)) => (gn, hn),
        (Node::Terminal(t), Node::Inner(_)) => {
            return match t.borrow() {
                True => apply_bin::<M, R, { BDDOp::Or as u8 }>(manager, rec, f, h),
                False => apply_bin::<M, R, { BDDOp::ImpStrict as u8 }>(manager, rec, f, h),
            };
        }
        (Node::Inner(_), Node::Terminal(t)) => {
            return match t.borrow() {
                True => apply_bin::<M, R, { BDDOp::Imp as u8 }>(manager, rec, f, g),
                False => apply_bin::<M, R, { BDDOp::And as u8 }>(manager, rec, f, g),
            };
        }
        (Node::Terminal(gt), Node::Terminal(_ht)) => {
            debug_assert_ne!(gt.borrow(), _ht.borrow()); // g == h is handled above
            return match gt.borrow() {
                False => apply_not(manager, rec, f), // if f { ⊥ } else { ⊤ }
                True => Ok(manager.clone_edge(&f)),  // if f { ⊤ } else { ⊥ }
            };
        }
    };

    // Query apply cache
    stat!(cache_query BDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        BDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit BDDOp::Ite);
        return Ok(res);
    }

    // Get the top-most level of the three
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = std::cmp::min(std::cmp::min(flevel, glevel), hlevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_children(gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (ht, he) = if hlevel == level {
        collect_children(hnode)
    } else {
        (h.borrowed(), h.borrowed())
    };

    let (t, e) = rec.ternary(apply_ite, manager, (ft, gt, ht), (fe, ge, he))?;
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), BDDOp::Ite)?;

    manager
        .apply_cache()
        .add(manager, BDDOp::Ite, &[f, g, h], res.borrowed());

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
    M: Manager<Terminal = BDDTerminal>,
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
            let t = EdgeDropGuard::new(manager, manager.get_terminal(BDDTerminal::True)?);
            let e = EdgeDropGuard::new(manager, manager.get_terminal(BDDTerminal::False)?);
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
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return substitute(manager, SequentialRecursor, f, subst, cache_id);
    }
    stat!(call BDDOp::Substitute);

    let Node::Inner(node) = manager.get_node(&f) else {
        return Ok(manager.clone_edge(&f));
    };
    let level = node.level();
    if level as usize >= subst.len() {
        return Ok(manager.clone_edge(&f));
    }

    // Query apply cache
    stat!(cache_query BDDOp::Substitute);
    if let Some(h) = manager.apply_cache().get_with_numeric(
        manager,
        BDDOp::Substitute,
        &[f.borrowed()],
        &[cache_id],
    ) {
        stat!(cache_hit BDDOp::Substitute);
        return Ok(h);
    }

    let (t, e) = collect_children(node);
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
        BDDOp::Substitute,
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
        fnode: &'a M::InnerNode,
    },
}

/// Tail-recursive part of [`restrict()`]
///
/// Invariant: `f` points to `fnode` at `flevel`, `vars` points to `vnode`
///
/// We expose this, because it can be reused for the multi-threaded version.
#[inline]
fn restrict_inner<'a, M>(
    manager: &'a M,
    f: Borrowed<'a, M::Edge>,
    fnode: &'a M::InnerNode,
    flevel: LevelNo,
    vars: Borrowed<'a, M::Edge>,
    vnode: &'a M::InnerNode,
) -> RestrictInnerResult<'a, M>
where
    M: Manager<Terminal = BDDTerminal>,
    M::InnerNode: HasLevel,
{
    use BDDTerminal::*;

    debug_assert!(std::ptr::eq(manager.get_node(&f).unwrap_inner(), fnode));
    debug_assert_eq!(fnode.level(), flevel);
    debug_assert!(std::ptr::eq(manager.get_node(&vars).unwrap_inner(), vnode));

    let vlevel = vnode.level();
    if vlevel > flevel {
        // f above vars
        return RestrictInnerResult::Rec { vars, f, fnode };
    }

    let vt = vnode.child(0);
    if vlevel < flevel {
        // vars above f
        return match manager.get_node(&vt) {
            Node::Inner(n) => restrict_inner(manager, f, fnode, flevel, vt, n),
            Node::Terminal(t) if *t.borrow() == True => {
                RestrictInnerResult::Done(manager.clone_edge(&f))
            }
            Node::Terminal(_) => {
                let ve = vnode.child(1);
                if let Node::Inner(n) = manager.get_node(&ve) {
                    restrict_inner(manager, f, fnode, flevel, ve, n)
                } else {
                    RestrictInnerResult::Done(manager.clone_edge(&f))
                }
            }
        };
    }

    debug_assert_eq!(vlevel, flevel);
    // top var at the level of f ⇒ select accordingly
    let (f, vars, vnode) = match manager.get_node(&vt) {
        Node::Inner(n) => {
            debug_assert!(
                manager.get_node(&vnode.child(1)).is_terminal(&False),
                "vars must be a conjunction of literals"
            );
            // positive literal ⇒ select then branch
            (fnode.child(0), vt, n)
        }
        Node::Terminal(t) if *t.borrow() == True => {
            debug_assert!(
                manager.get_node(&vnode.child(1)).is_terminal(&False),
                "vars must be a conjunction of literals"
            );
            // positive literal ⇒ select then branch
            return RestrictInnerResult::Done(manager.clone_edge(&fnode.child(0)));
        }
        Node::Terminal(_) => {
            // negative literal ⇒ select else branch
            let f = fnode.child(1);
            let ve = vnode.child(1);
            if let Node::Inner(n) = manager.get_node(&ve) {
                (f, ve, n)
            } else {
                return RestrictInnerResult::Done(manager.clone_edge(&f));
            }
        }
    };

    if let Node::Inner(fnode) = manager.get_node(&f) {
        restrict_inner(manager, f, fnode, fnode.level(), vars, vnode)
    } else {
        RestrictInnerResult::Done(manager.clone_edge(&f))
    }
}

fn restrict<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return restrict(manager, SequentialRecursor, f, vars);
    }
    stat!(call BDDOp::Restrict);

    let (Node::Inner(fnode), Node::Inner(vnode)) = (manager.get_node(&f), manager.get_node(&vars))
    else {
        return Ok(manager.clone_edge(&f));
    };

    match restrict_inner(manager, f, fnode, fnode.level(), vars, vnode) {
        RestrictInnerResult::Done(res) => Ok(res),
        RestrictInnerResult::Rec { vars, f, fnode } => {
            // f above top-most restrict variable

            // Query apply cache
            stat!(cache_query BDDOp::Restrict);
            if let Some(res) = manager.apply_cache().get(
                manager,
                BDDOp::Restrict,
                &[f.borrowed(), vars.borrowed()],
            ) {
                stat!(cache_hit BDDOp::Restrict);
                return Ok(res);
            }

            let (ft, fe) = collect_children(fnode);
            let (t, e) = rec.binary(
                restrict,
                manager,
                (ft, vars.borrowed()),
                (fe, vars.borrowed()),
            )?;

            let res = reduce(
                manager,
                fnode.level(),
                t.into_edge(),
                e.into_edge(),
                BDDOp::Restrict,
            )?;

            manager
                .apply_cache()
                .add(manager, BDDOp::Restrict, &[f, vars], res.borrowed());

            Ok(res)
        }
    }
}

/// Compute the quantification `Q` over `vars`
///
/// Note that `Q` is one of `BDDOp::And`, `BDDOp::Or`, or `BDDOp::Xor` as `u8`.
/// This saves us another case distinction in the code (would not be present at
/// runtime).
fn quant<M, R: Recursor<M>, const Q: u8>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return quant::<M, _, Q>(manager, SequentialRecursor, f, vars);
    }
    let operator = match () {
        _ if Q == BDDOp::And as u8 => BDDOp::Forall,
        _ if Q == BDDOp::Or as u8 => BDDOp::Exist,
        _ if Q == BDDOp::Xor as u8 => BDDOp::Unique,
        _ => unreachable!("invalid quantifier"),
    };
    stat!(call operator);

    // Terminal cases
    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(_) => {
            return if operator != BDDOp::Unique || manager.get_node(&vars).is_any_terminal() {
                Ok(manager.clone_edge(&f))
            } else {
                // ∃! x. ⊤ ≡ ⊤ ⊕ ⊤ ≡ ⊥
                manager.get_terminal(BDDTerminal::False)
            };
        }
    };
    let flevel = fnode.level();

    let vars = if operator != BDDOp::Unique {
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
    if operator == BDDOp::Unique && vlevel < flevel {
        // `vnode` above `fnode`, i.e., the variable does not occur in `f` (see above)
        return manager.get_terminal(BDDTerminal::False);
    }
    debug_assert!(flevel <= vlevel);

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

    let (ft, fe) = collect_children(fnode);
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
        apply_bin::<M, _, Q>(manager, rec, t.borrowed(), e.borrowed())
    } else {
        reduce(manager, flevel, t.into_edge(), e.into_edge(), operator)
    }?;

    manager
        .apply_cache()
        .add(manager, operator, &[f, vars], res.borrowed());

    Ok(res)
}

/// Recursively apply the binary operator `OP` to `f` and `g` while quantifying
/// `Q` over `vars`. This is more efficient then computing then an apply
/// operation followed by a quantification.
///
/// One example usage is for the relational product, i.e., computing `∃ s,
/// t: S(s) ∧ T(s, t)`, where `S` is a boolean function representing the states
/// and `T` a boolean function representing the transition relation.
///
/// Note that `Q` is one of `BDDOp::And`, `BDDOp::Or`, or `BDDOp::Xor` as `u8`.
/// This saves us another case distinction in the code (would not be present at
/// runtime). We use a `const` parameter `OP` to have specialized version of
/// this function for each operator.
fn apply_quant<M, R: Recursor<M>, const Q: u8, const OP: u8>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_quant::<M, _, Q, OP>(manager, SequentialRecursor, f, g, vars);
    }
    let operator = const { BDDOp::from_apply_quant(Q, OP) };
    stat!(call operator);

    // Handle the terminal cases
    let (f, g) = match super::terminal_bin::<M, OP>(manager, &f, &g) {
        Operation::Binary(_, f, g) => (f, g),
        Operation::Not(h) => {
            let inverse = EdgeDropGuard::new(manager, apply_not(manager, rec, h)?);
            return quant::<M, R, Q>(manager, rec, inverse.borrowed(), vars);
        }
        Operation::Done(h) => {
            let h = EdgeDropGuard::new(manager, h);
            return quant::<M, R, Q>(manager, rec, h.borrowed(), vars);
        }
    };

    // Handle cases where f, g are below the variables.
    let fnode = match manager.get_node(&f) {
        Node::Inner(fnode) => fnode,
        Node::Terminal(_) => unreachable!("Terminal cases handled above"),
    };

    let gnode = match manager.get_node(&g) {
        Node::Inner(gnode) => gnode,
        Node::Terminal(_) => unreachable!("Terminal cases handled above"),
    };

    let flevel = fnode.level();
    let glevel = gnode.level();
    let min_level = std::cmp::min(fnode.level(), gnode.level());

    let vars = if Q != BDDOp::Xor as u8 {
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
        Node::Terminal(_) => return apply_bin::<M, R, OP>(manager, rec, f, g),
    };

    let vlevel = vnode.level();
    if vlevel < min_level && Q == BDDOp::Xor as u8 {
        // `vnode` above `fnode` and `gnode`, i.e., the variable does not occur in `f`
        // or `g` (see above)
        return manager.get_terminal(BDDTerminal::False);
    }

    if min_level > vlevel {
        // We are beyond the variables to be quantified, so simply apply.
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
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };

    let (gt, ge) = if flevel >= glevel {
        collect_children(gnode)
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
        apply_bin::<M, R, Q>(manager, rec, t.borrowed(), e.borrowed())?
    } else {
        reduce(manager, min_level, t.into_edge(), e.into_edge(), operator)?
    };

    manager
        .apply_cache()
        .add(manager, operator, &[f, g, vars], res.borrowed());

    Ok(res)
}

/// Dynamic dispatcher for [`apply_quant()`] and unique quantification
///
/// In contrast to [`apply_quant()`], the operator is not a const but a runtime
/// parameter.
fn apply_quant_dispatch<'a, M, R: Recursor<M>, const Q: u8>(
    manager: &'a M,
    rec: R,
    op: BooleanOperator,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp>,
    M::InnerNode: HasLevel,
{
    use BooleanOperator::*;
    match op {
        And => apply_quant::<_, _, Q, { BDDOp::And as u8 }>(manager, rec, f, g, vars),
        Or => apply_quant::<_, _, Q, { BDDOp::Or as u8 }>(manager, rec, f, g, vars),
        Xor => apply_quant::<_, _, Q, { BDDOp::Xor as u8 }>(manager, rec, f, g, vars),
        Equiv => apply_quant::<_, _, Q, { BDDOp::Equiv as u8 }>(manager, rec, f, g, vars),
        Nand => apply_quant::<_, _, Q, { BDDOp::Nand as u8 }>(manager, rec, f, g, vars),
        Nor => apply_quant::<_, _, Q, { BDDOp::Nor as u8 }>(manager, rec, f, g, vars),
        Imp => apply_quant::<_, _, Q, { BDDOp::Imp as u8 }>(manager, rec, f, g, vars),
        ImpStrict => apply_quant::<_, _, Q, { BDDOp::ImpStrict as u8 }>(manager, rec, f, g, vars),
    }
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasBDDOpApplyCache<M: Manager>: HasApplyCache<M, BDDOp> {}
impl<M: Manager + HasApplyCache<M, BDDOp>> HasBDDOpApplyCache<M> for M {}

/// Boolean function backed by a binary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr(transparent)]
pub struct BDDFunction<F: Function>(F);

impl<F: Function> From<F> for BDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        BDDFunction(value)
    }
}

impl<F: Function> BDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function> FunctionSubst for BDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>>,
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

impl<F: Function> BooleanFunction for BDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    #[inline]
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let ft = manager.get_terminal(BDDTerminal::True).unwrap();
        let fe = manager.get_terminal(BDDTerminal::False).unwrap();
        let edge = manager.add_level(|level| InnerNode::new(level, [ft, fe]))?;
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(BDDTerminal::False).unwrap()
    }
    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(BDDTerminal::True).unwrap()
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_not(manager, SequentialRecursor, edge.borrowed())
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::And as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::Or as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::Nand as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::Nor as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::Xor as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::Equiv as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::Imp as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_bin::<_, _, { BDDOp::ImpStrict as u8 }>(manager, rec, lhs.borrowed(), rhs.borrowed())
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

    fn sat_count_edge<'id, N: SatCountNumber, S: BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        fn inner<M: Manager<Terminal = BDDTerminal>, N: SatCountNumber, S: BuildHasher>(
            manager: &M,
            e: Borrowed<M::Edge>,
            terminal_val: &N,
            cache: &mut SatCountCache<N, S>,
        ) -> N {
            let node = match manager.get_node(&e) {
                Node::Inner(node) => node,
                Node::Terminal(t) => {
                    return if *t.borrow() == BDDTerminal::True {
                        terminal_val.clone()
                    } else {
                        N::from(0u32)
                    };
                }
            };
            let node_id = e.node_id();
            if let Some(n) = cache.map.get(&node_id) {
                return n.clone();
            }
            let (e0, e1) = collect_children(node);
            let mut n = inner(manager, e0, terminal_val, cache);
            n += &inner(manager, e1, terminal_val, cache);
            n >>= 1u32;
            cache.map.insert(node_id, n.clone());
            n
        }

        cache.clear_if_invalid(manager, vars);

        let mut terminal_val = N::from(1u32);
        let scale_exp = (-N::MIN_EXP) as u32;
        terminal_val <<= if scale_exp != 0 && vars >= scale_exp {
            // scale down to increase the precision if we use floating point
            // numbers and have many variables
            vars - scale_exp
        } else {
            vars
        };
        let mut res = inner(manager, edge.borrowed(), &terminal_val, cache);
        if scale_exp != 0 && vars >= scale_exp {
            res <<= scale_exp; // scale up again
        }
        res
    }

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
        fn inner<M: Manager<Terminal = BDDTerminal>>(
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
            let level = node.level();
            let (t, e) = collect_children(node);
            let c = if manager.get_node(&t).is_terminal(&BDDTerminal::False) {
                false
            } else if manager.get_node(&e).is_terminal(&BDDTerminal::False) {
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

        match manager.get_node(edge) {
            Node::Inner(_) => {}
            Node::Terminal(t) => {
                return match *t.borrow() {
                    BDDTerminal::False => None,
                    BDDTerminal::True => Some(vec![OptBool::None; manager.num_levels() as usize]),
                }
            }
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
        fn inner<M: Manager<Terminal = BDDTerminal>>(
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

            let (t, e) = collect_children(node);
            let level = node.level();
            let c = if manager.get_node(&t).is_terminal(&BDDTerminal::False) {
                false
            } else if manager.get_node(&e).is_terminal(&BDDTerminal::False) {
                true
            } else {
                choice(manager, &edge, level)
            };

            let sub = EdgeDropGuard::new(manager, inner(manager, if c { t } else { e }, choice)?);
            debug_assert!(!manager.get_node(&sub).is_terminal(&BDDTerminal::False));
            let f = manager.get_terminal(BDDTerminal::False)?;
            let sub = sub.into_edge();
            let children = if c { [sub, f] } else { [f, sub] };

            oxidd_core::LevelView::get_or_insert(
                &mut manager.level(level),
                M::InnerNode::new(level, children),
            )
        }

        inner(manager, edge.borrowed(), choice)
    }

    #[inline]
    fn pick_cube_dd_set_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        literal_set: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        fn inner<M: Manager<Terminal = BDDTerminal>>(
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
                    let (t, e) = collect_children(node);
                    if manager.get_node(&e).is_terminal(&BDDTerminal::False) {
                        (e, true)
                    } else {
                        (t, false)
                    }
                }
                _ => (literal_set, false),
            };

            let (t, e) = collect_children(node);
            let c = if manager.get_node(&t).is_terminal(&BDDTerminal::False) {
                false
            } else if manager.get_node(&e).is_terminal(&BDDTerminal::False) {
                true
            } else {
                c
            };

            let sub =
                EdgeDropGuard::new(manager, inner(manager, if c { t } else { e }, literal_set)?);
            debug_assert!(!manager.get_node(&sub).is_terminal(&BDDTerminal::False));
            let f = manager.get_terminal(BDDTerminal::False)?;
            let sub = sub.into_edge();
            let children = if c { [sub, f] } else { [f, sub] };

            oxidd_core::LevelView::get_or_insert(
                &mut manager.level(level),
                M::InnerNode::new(level, children),
            )
        }

        inner(manager, edge.borrowed(), literal_set.borrowed())
    }

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
        fn inner<M>(manager: &M, edge: Borrowed<M::Edge>, values: BitVec) -> bool
        where
            M: Manager<Terminal = BDDTerminal>,
            M::InnerNode: HasLevel,
        {
            match manager.get_node(&edge) {
                Node::Inner(node) => {
                    let edge = node.child((!values[node.level() as usize]) as usize);
                    inner(manager, edge, values)
                }
                Node::Terminal(t) => *t.borrow() == BDDTerminal::True,
            }
        }

        inner(manager, edge.borrowed(), values)
    }
}

impl<F: Function> BooleanFunctionQuant for BDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>>,
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
        quant::<_, _, { BDDOp::And as u8 }>(manager, rec, root.borrowed(), vars.borrowed())
    }
    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        quant::<_, _, { BDDOp::Or as u8 }>(manager, rec, root.borrowed(), vars.borrowed())
    }
    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        quant::<_, _, { BDDOp::Xor as u8 }>(manager, rec, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn apply_forall_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let (lhs, rhs, vars) = (lhs.borrowed(), rhs.borrowed(), vars.borrowed());
        apply_quant_dispatch::<_, _, { BDDOp::And as u8 }>(manager, rec, op, lhs, rhs, vars)
    }
    #[inline]
    fn apply_exist_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let (lhs, rhs, vars) = (lhs.borrowed(), rhs.borrowed(), vars.borrowed());
        apply_quant_dispatch::<_, _, { BDDOp::Or as u8 }>(manager, rec, op, lhs, rhs, vars)
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
        apply_quant_dispatch::<_, _, { BDDOp::Xor as u8 }>(manager, rec, op, lhs, rhs, vars)
    }
}

impl<F: Function, T: Tag> DotStyle<T> for BDDFunction<F> {}

#[cfg(feature = "multi-threading")]
pub mod mt {
    use oxidd_core::HasWorkers;

    use crate::recursor::mt::ParallelRecursor;

    use super::*;

    /// Boolean function backed by a binary decision diagram, multi-threaded
    /// version
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
    #[repr(transparent)]
    pub struct BDDFunctionMT<F: Function>(F);

    impl<F: Function> From<F> for BDDFunctionMT<F> {
        #[inline(always)]
        fn from(value: F) -> Self {
            BDDFunctionMT(value)
        }
    }

    impl<F: Function> BDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: HasWorkers,
    {
        /// Convert `self` into the underlying [`Function`]
        #[inline(always)]
        pub fn into_inner(self) -> F {
            self.0
        }
    }

    impl<F: Function> FunctionSubst for BDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>:
            Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>> + HasWorkers,
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
            let cache_id = substitution.id();
            let edge = edge.borrowed();
            let rec = ParallelRecursor::new(manager);
            substitute(manager, rec, edge.borrowed(), &subst, cache_id)
        }
    }

    impl<F: Function> BooleanFunction for BDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>:
            Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>> + HasWorkers,
        for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
        for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
    {
        #[inline]
        fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
            let ft = manager.get_terminal(BDDTerminal::True).unwrap();
            let fe = manager.get_terminal(BDDTerminal::False).unwrap();
            let edge = manager.add_level(|level| InnerNode::new(level, [ft, fe]))?;
            Ok(Self::from_edge(manager, edge))
        }

        #[inline]
        fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            manager.get_terminal(BDDTerminal::False).unwrap()
        }
        #[inline]
        fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            manager.get_terminal(BDDTerminal::True).unwrap()
        }

        #[inline]
        fn not_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let edge = edge.borrowed();
            apply_not(manager, ParallelRecursor::new(manager), edge)
        }

        #[inline]
        fn and_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::And as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn or_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::Or as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn nand_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::Nand as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn nor_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::Nor as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn xor_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::Xor as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn equiv_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::Equiv as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn imp_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::Imp as u8 }>(manager, rec, lhs, rhs)
        }
        #[inline]
        fn imp_strict_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            apply_bin::<_, _, { BDDOp::ImpStrict as u8 }>(manager, rec, lhs, rhs)
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
            BDDFunction::<F>::sat_count_edge(manager, edge, vars, cache)
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
            BDDFunction::<F>::pick_cube_edge(manager, edge, order, choice)
        }
        #[inline]
        fn pick_cube_dd_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            BDDFunction::<F>::pick_cube_dd_edge(manager, edge, choice)
        }
        #[inline]
        fn pick_cube_dd_set_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            literal_set: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            BDDFunction::<F>::pick_cube_dd_set_edge(manager, edge, literal_set)
        }

        #[inline]
        fn eval_edge<'id, 'a>(
            manager: &'a Self::Manager<'id>,
            edge: &'a EdgeOfFunc<'id, Self>,
            args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
        ) -> bool {
            BDDFunction::<F>::eval_edge(manager, edge, args)
        }
    }

    impl<F: Function> BooleanFunctionQuant for BDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>:
            Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>> + HasWorkers,
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
            quant::<_, _, { BDDOp::And as u8 }>(manager, rec, root, vars)
        }
        #[inline]
        fn exist_edge<'id>(
            manager: &Self::Manager<'id>,
            root: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (root, vars) = (root.borrowed(), vars.borrowed());
            let rec = ParallelRecursor::new(manager);
            quant::<_, _, { BDDOp::Or as u8 }>(manager, rec, root, vars)
        }
        #[inline]
        fn unique_edge<'id>(
            manager: &Self::Manager<'id>,
            root: &EdgeOfFunc<'id, Self>,
            vars: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (root, vars) = (root.borrowed(), vars.borrowed());
            let rec = ParallelRecursor::new(manager);
            quant::<_, _, { BDDOp::Xor as u8 }>(manager, rec, root, vars)
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
            let rec = ParallelRecursor::new(manager);
            apply_quant_dispatch::<_, _, { BDDOp::And as u8 }>(manager, rec, op, lhs, rhs, vars)
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
            let rec = ParallelRecursor::new(manager);
            apply_quant_dispatch::<_, _, { BDDOp::Or as u8 }>(manager, rec, op, lhs, rhs, vars)
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
            apply_quant_dispatch::<_, _, { BDDOp::Xor as u8 }>(manager, rec, op, lhs, rhs, vars)
        }
    }

    impl<F: Function, T: Tag> DotStyle<T> for BDDFunctionMT<F> {}
}
