//! Recursive single-threaded apply algorithms

use std::hash::BuildHasher;

use bitvec::vec::BitVec;

use oxidd_core::function::BooleanFunction;
use oxidd_core::function::BooleanFunctionQuant;
use oxidd_core::function::EdgeOfFunc;
use oxidd_core::function::Function;
use oxidd_core::util::AllocResult;
use oxidd_core::util::Borrowed;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::util::OptBool;
use oxidd_core::util::SatCountCache;
use oxidd_core::util::SatCountNumber;
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Manager;
use oxidd_core::Node;
use oxidd_core::NodeID;
use oxidd_core::Tag;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use crate::stat;

use super::collect_cofactors;
use super::get_terminal;
use super::not;
use super::not_owned;
use super::reduce;
use super::BCDDOp;
use super::BCDDTerminal;
use super::EdgeTag;
use super::NodesOrDone;

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
///
/// Using `Borrowed<M::Edge>` instead of `&M::Edge` means that we actually
/// pass the edge by value, which saves a few indirections.
pub(super) fn apply_bin<M, const OP: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, Operator = BCDDOp>,
    M::InnerNode: HasLevel,
{
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
            NodesOrDone::Done(h) => return Ok(h),
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
                return Ok(h);
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

    let t = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, ft, gt)?);
    let e = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, fe, ge)?);

    let h = reduce(manager, level, t.into_edge(), e.into_edge(), op)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, op, &[f, g], h.borrowed());

    Ok(h)
}

/// Shorthand for `apply_bin_rec::<M, { BCDDOp::And as u8 }>(manager, f, g)`
#[inline(always)]
fn apply_and<M>(manager: &M, f: Borrowed<M::Edge>, g: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, Operator = BCDDOp>,
    M::InnerNode: HasLevel,
{
    apply_bin::<M, { BCDDOp::And as u8 }>(manager, f, g)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
pub(super) fn apply_ite<M>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = BCDDTerminal> + HasApplyCache<M, Operator = BCDDOp>,
    M::InnerNode: HasLevel,
{
    stat!(call BCDDOp::Ite);

    // Terminal cases
    let gu = g.with_tag(EdgeTag::None); // untagged
    let hu = h.with_tag(EdgeTag::None);
    if gu == hu {
        return Ok(if g.tag() == h.tag() {
            manager.clone_edge(&g)
        } else {
            not_owned(apply_bin::<M, { BCDDOp::Xor as u8 }>(manager, f, g)?) // f ↔ g
        });
    }
    let fu = f.with_tag(EdgeTag::None);
    if fu == gu {
        return if f.tag() == g.tag() {
            Ok(not_owned(apply_and(manager, not(&f), not(&h))?)) // f ∨ h
        } else {
            apply_and(manager, not(&f), h) // f < h
        };
    }
    if fu == hu {
        return if f.tag() == h.tag() {
            apply_and(manager, f, g)
        } else {
            // f → g = ¬f ∨ g = ¬(f ∧ ¬g)
            Ok(not_owned(apply_and(manager, f, not(&g))?))
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
                Ok(not_owned(apply_and(manager, not(&f), not(&h))?)) // f ∨ h
            } else {
                apply_and(manager, not(&f), h) // f < h
            };
        }
        (_gnode, Node::Terminal(_)) => {
            debug_assert!(_gnode.is_inner());
            return if h.tag() == EdgeTag::None {
                Ok(not_owned(apply_and(manager, f, not(&g))?)) // f → g
            } else {
                apply_and(manager, f, g)
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

    let t = EdgeDropGuard::new(manager, apply_ite(manager, ft, gt, ht)?);
    let e = EdgeDropGuard::new(manager, apply_ite(manager, fe, ge, he)?);
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), BCDDOp::Ite)?;

    manager
        .apply_cache()
        .add(manager, BCDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

/// Result of [`restrict_inner()`]
pub(super) enum RestrictInnerResult<'a, M: Manager> {
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
pub(super) fn restrict_inner<'a, M>(
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

pub(super) fn restrict<M>(
    manager: &M,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, Operator = BCDDOp>,
    M::InnerNode: HasLevel,
{
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

            let t =
                EdgeDropGuard::new(manager, restrict(manager, fnode.child(0), vars.borrowed())?);
            let e =
                EdgeDropGuard::new(manager, restrict(manager, fnode.child(1), vars.borrowed())?);

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
pub(super) fn quant<M, const Q: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, Operator = BCDDOp>,
    M::InnerNode: HasLevel,
{
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
    let t = EdgeDropGuard::new(manager, quant::<M, Q>(manager, ft, vt.borrowed())?);
    let e = EdgeDropGuard::new(manager, quant::<M, Q>(manager, fe, vt.borrowed())?);

    let res = if flevel == vlevel {
        match operator {
            BCDDOp::Forall => apply_and(manager, t.borrowed(), e.borrowed())?,
            BCDDOp::Exist => not_owned(apply_and(manager, not(&t), not(&e))?),
            BCDDOp::Unique => {
                apply_bin::<M, { BCDDOp::Xor as u8 }>(manager, t.borrowed(), e.borrowed())?
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

// --- Function Interface ------------------------------------------------------

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

impl<F: Function> BooleanFunction for BCDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + super::HasBCDDOpApplyCache<F::Manager<'id>>,
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
        apply_and(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Ok(not_owned(apply_and(manager, not(lhs), not(rhs))?))
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
        apply_and(manager, not(lhs), not(rhs))
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BCDDOp::Xor as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
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
        Ok(not_owned(apply_and(manager, lhs.borrowed(), not(rhs))?))
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_and(manager, not(lhs), rhs.borrowed())
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

        let mut terminal_val = N::from(1u32);
        terminal_val <<= vars;
        if N::FLOATING_POINT {
            inner_floating(manager, edge.borrowed(), &terminal_val, cache)
        } else {
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
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
    {
        #[inline] // this function is tail-recursive
        fn inner<M: Manager<EdgeTag = EdgeTag>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            cube: &mut [OptBool],
            mut choice: impl FnMut(&M, &M::Edge) -> bool,
        ) where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return;
            };
            let tag = edge.tag();
            let (t, e) = collect_cofactors(tag, node);
            let c = if manager.get_node(&t).is_any_terminal() && t.tag() == EdgeTag::Complemented {
                false
            } else if manager.get_node(&e).is_any_terminal() && e.tag() == EdgeTag::Complemented {
                true
            } else {
                choice(manager, &edge)
            };
            cube[node.level() as usize] = OptBool::from(c);
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
    for<'id> F::Manager<'id>: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag>
        + super::HasBCDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    #[inline]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        restrict(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        quant::<_, { BCDDOp::Forall as u8 }>(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        quant::<_, { BCDDOp::Exist as u8 }>(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        quant::<_, { BCDDOp::Unique as u8 }>(manager, root.borrowed(), vars.borrowed())
    }
}

impl<F: Function, T: Tag> DotStyle<T> for BCDDFunction<F> {}
