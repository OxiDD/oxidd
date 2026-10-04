//! Recursive single-threaded apply algorithms

use std::borrow::Borrow;

use fixedbitset::FixedBitSet;

use oxidd_core::function::{
    EdgeOfFunc, Function, INodeOfFunc, NumberBase, OwnEdgeOfFunc, PseudoBooleanFunction,
};
use oxidd_core::util::{AllocResult, EdgeDropGuard, Own, Ref};
use oxidd_core::{ApplyCache, HasApplyCache, HasLevel, InnerNode, Manager, Node, Tag, VarNo};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{MTBDDOp, Operation, collect_children, reduce, stat};

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

trait MTBDDManager: Manager<InnerNodeValue = ()> + HasApplyCache<Self, MTBDDOp> {}
impl<M: Manager<InnerNodeValue = ()> + HasApplyCache<Self, MTBDDOp>> MTBDDManager for M {}

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M: MTBDDManager, const OP: u8>(
    manager: &M,
    f: Ref<'_, M::Edge>,
    g: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>>
where
    M::InnerNode: HasLevel,
    M::Terminal: NumberBase,
{
    stat!(call OP);
    let (operator, op1, op2) = match super::terminal_bin::<M, OP>(manager, f, g)? {
        Operation::Binary(o, op1, op2) => (o, op1, op2),
        Operation::Done(h) => return Ok(h),
    };

    // Query apply cache
    stat!(cache_query OP);
    if let Some(h) = manager.apply_cache().get(manager, operator, &[op1, op2]) {
        stat!(cache_hit OP);
        return Ok(h);
    }

    let fnode = manager.get_node(f);
    let gnode = manager.get_node(g);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (f0, f1) = if flevel == level {
        collect_children(fnode.unwrap_inner())
    } else {
        (f, f)
    };
    let (g0, g1) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g, g)
    };

    let t = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f0, g0)?);
    let e = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f1, g1)?);
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), operator)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, operator, &[op1, op2], h.borrowed());

    Ok(h)
}

/// Recursively restrict a set of `vars` (a conjunction of literals) to
/// constant values in `f`
fn restrict<'a, M, T>(
    manager: &'a M,
    mut f: Ref<'a, M::Edge>,
    mut vars: Ref<'a, M::Edge>,
) -> AllocResult<Own<M::Edge>>
where
    M: Manager<Terminal = T, InnerNodeValue = ()> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Restrict);

    let (Node::Inner(mut fnode), Node::Inner(mut vnode)) =
        (manager.get_node(f), manager.get_node(vars))
    else {
        return Ok(manager.clone_edge(f));
    };

    let mut flevel = fnode.level();
    loop {
        debug_assert!(std::ptr::eq(manager.get_node(f).unwrap_inner(), fnode));
        debug_assert_eq!(fnode.level(), flevel);
        debug_assert!(std::ptr::eq(manager.get_node(vars).unwrap_inner(), vnode));

        let vlevel = vnode.level();
        if vlevel > flevel {
            // f above vars
            break;
        }

        let vt = vnode.child(0);
        if vlevel < flevel {
            // vars above f
            (vars, vnode) = match manager.get_node(vt) {
                Node::Inner(n) => (vt, n),
                Node::Terminal(t) if t.borrow().is_one() => return Ok(manager.clone_edge(f)),
                Node::Terminal(_) => {
                    let ve = vnode.child(1);
                    if let Node::Inner(n) = manager.get_node(ve) {
                        (ve, n)
                    } else {
                        return Ok(manager.clone_edge(f));
                    }
                }
            };
            continue;
        }

        debug_assert_eq!(vlevel, flevel);
        // top var at the level of f ⇒ select accordingly
        (vars, vnode) = match manager.get_node(vt) {
            Node::Inner(n) => {
                debug_assert!(
                    matches!(manager.get_node(vnode.child(1)), Node::Terminal(t) if t.borrow().is_zero()),
                    "vars must be a conjunction of literals"
                );
                // positive literal ⇒ select then branch
                f = fnode.child(0);
                (vt, n)
            }
            Node::Terminal(t) if t.borrow().is_one() => {
                debug_assert!(
                    matches!(manager.get_node(vnode.child(1)), Node::Terminal(t) if t.borrow().is_zero()),
                    "vars must be a conjunction of literals"
                );
                // positive literal ⇒ select then branch
                return Ok(manager.clone_edge(fnode.child(0)));
            }
            Node::Terminal(_) => {
                // negative literal ⇒ select else branch
                f = fnode.child(1);
                let ve = vnode.child(1);
                if let Node::Inner(n) = manager.get_node(ve) {
                    (ve, n)
                } else {
                    return Ok(manager.clone_edge(f));
                }
            }
        };

        if let Node::Inner(n) = manager.get_node(f) {
            fnode = n;
            flevel = n.level();
        } else {
            return Ok(manager.clone_edge(f));
        }
    }

    // f above top-most restrict variable

    // Query apply cache
    stat!(cache_query MTBDDOp::Restrict);
    if let Some(res) = manager
        .apply_cache()
        .get(manager, MTBDDOp::Restrict, &[f, vars])
    {
        stat!(cache_hit MTBDDOp::Restrict);
        return Ok(res);
    }

    let (ft, fe) = collect_children(fnode);
    let t = EdgeDropGuard::new(manager, restrict(manager, ft, vars)?);
    let e = EdgeDropGuard::new(manager, restrict(manager, fe, vars)?);
    let res = reduce(
        manager,
        fnode.level(),
        t.into_edge(),
        e.into_edge(),
        MTBDDOp::Restrict,
    )?;

    manager
        .apply_cache()
        .add(manager, MTBDDOp::Restrict, &[f, vars], res.borrowed());

    Ok(res)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
///
/// `f` must be a 0-1-valued MTBDD (see [`PseudoBooleanFunction::ite_edge`]).
/// As an extension of the classical restriction, terminals of `f` other than
/// `0` and `1` are treated as "truthy" (`debug_assert`-ed against,
/// since this indicates a violation of the documented precondition).
fn apply_ite<M, T>(
    manager: &M,
    f: Ref<'_, M::Edge>,
    g: Ref<'_, M::Edge>,
    h: Ref<'_, M::Edge>,
) -> AllocResult<Own<M::Edge>>
where
    M: Manager<Terminal = T, InnerNodeValue = ()> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Ite);

    // The condition is irrelevant if both branches agree.
    if g == h {
        return Ok(manager.clone_edge(g));
    }

    // Terminal cases for `f`. We decide as soon as `f` resolves to a
    // terminal, which is what makes this a 0-1-valued-condition restricted
    // "ite", as opposed to a fully generic ternary operator.
    let fnode = match manager.get_node(f) {
        Node::Inner(node) => node,
        Node::Terminal(t) => {
            let t = t.borrow();
            return Ok(if t.is_zero() {
                manager.clone_edge(h)
            } else {
                debug_assert!(t.is_one(), "the condition of `ite` must be 0-1-valued");
                manager.clone_edge(g)
            });
        }
    };

    // Query apply cache
    stat!(cache_query MTBDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(manager, MTBDDOp::Ite, &[f, g, h]) {
        stat!(cache_hit MTBDDOp::Ite);
        return Ok(res);
    }

    let gnode = manager.get_node(g);
    let hnode = manager.get_node(h);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = flevel.min(glevel).min(hlevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_children(fnode)
    } else {
        (f, f)
    };
    let (gt, ge) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g, g)
    };
    let (ht, he) = if hlevel == level {
        collect_children(hnode.unwrap_inner())
    } else {
        (h, h)
    };

    let t = EdgeDropGuard::new(manager, apply_ite(manager, ft, gt, ht)?);
    let e = EdgeDropGuard::new(manager, apply_ite(manager, fe, ge, he)?);
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), MTBDDOp::Ite)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, MTBDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

/// Boolean function backed by a binary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr_id = "MTBDD"]
#[repr(transparent)]
pub struct MTBDDFunction<F: Function>(F);

impl<F: Function> From<F> for MTBDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        MTBDDFunction(value)
    }
}

impl<F: Function> MTBDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function, T: NumberBase> PseudoBooleanFunction for MTBDDFunction<F>
where
    for<'id> F::Manager<'id>: MTBDDManager + Manager<Terminal = T>,
    for<'id> INodeOfFunc<'id, F>: HasLevel,
{
    type Number = T;

    #[inline]
    fn constant_edge<'id>(
        manager: &Self::Manager<'id>,
        value: Self::Number,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        manager.get_terminal(value)
    }

    #[inline]
    fn var_edge<'id>(
        manager: &Self::Manager<'id>,
        var: VarNo,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        let level = manager.var_to_level(var);
        let t = EdgeDropGuard::new(manager, manager.get_terminal(T::one())?);
        let e = EdgeDropGuard::new(manager, manager.get_terminal(T::zero())?);
        oxidd_core::LevelView::get_or_insert(
            &mut manager.level(level),
            InnerNode::new(level, [t.into_edge(), e.into_edge()], ()),
        )
    }

    #[inline]
    fn add_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: Ref<'_, EdgeOfFunc<'id, Self>>,
        rhs: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_bin::<_, { MTBDDOp::Add as u8 }>(manager, lhs, rhs)
    }

    #[inline]
    fn sub_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: Ref<'_, EdgeOfFunc<'id, Self>>,
        rhs: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_bin::<_, { MTBDDOp::Sub as u8 }>(manager, lhs, rhs)
    }

    #[inline]
    fn mul_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: Ref<'_, EdgeOfFunc<'id, Self>>,
        rhs: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_bin::<_, { MTBDDOp::Mul as u8 }>(manager, lhs, rhs)
    }

    #[inline]
    fn div_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: Ref<'_, EdgeOfFunc<'id, Self>>,
        rhs: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_bin::<_, { MTBDDOp::Div as u8 }>(manager, lhs, rhs)
    }

    #[inline]
    fn min_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: Ref<'_, EdgeOfFunc<'id, Self>>,
        rhs: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_bin::<_, { MTBDDOp::Min as u8 }>(manager, lhs, rhs)
    }

    #[inline]
    fn max_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: Ref<'_, EdgeOfFunc<'id, Self>>,
        rhs: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_bin::<_, { MTBDDOp::Max as u8 }>(manager, lhs, rhs)
    }

    #[inline]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: Ref<'_, EdgeOfFunc<'id, Self>>,
        vars: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        restrict::<_, T>(manager, root, vars)
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: Ref<'_, EdgeOfFunc<'id, Self>>,
        then_edge: Ref<'_, EdgeOfFunc<'id, Self>>,
        else_edge: Ref<'_, EdgeOfFunc<'id, Self>>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        apply_ite::<_, T>(manager, if_edge, then_edge, else_edge)
    }

    #[inline]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        mut edge: Ref<'a, EdgeOfFunc<'id, Self>>,
        args: impl IntoIterator<Item = (VarNo, bool)>,
    ) -> T {
        // `choices` maps levels to the child number to choose
        let mut choices = FixedBitSet::with_capacity(manager.num_levels() as usize);
        for (var, val) in args {
            // child 0 is "then"/"true", hence the negation
            choices.set(manager.var_to_level(var) as usize, !val);
        }

        loop {
            match manager.get_node(edge) {
                Node::Inner(node) => {
                    edge = node.child(choices.contains(node.level() as usize) as usize)
                }
                Node::Terminal(t) => break t.borrow().clone(),
            }
        }
    }
}

impl<F: Function, T: Tag> DotStyle<T> for MTBDDFunction<F> {}
