//! Recursive single-threaded apply algorithms

use std::borrow::Borrow;

use fixedbitset::FixedBitSet;

use oxidd_core::function::{EdgeOfFunc, Function, INodeOfFunc, NumberBase, PseudoBooleanFunction};
use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::{ApplyCache, Edge, HasApplyCache, HasLevel, InnerNode, Manager, Node, Tag, VarNo};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{MTBDDOp, Operation, collect_children, reduce, stat};

// spell-checker:ignore fnode,gnode,flevel,glevel

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M, T, const OP: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call OP);
    let (operator, op1, op2) = match super::terminal_bin::<M, T, OP>(manager, &f, &g)? {
        Operation::Binary(o, op1, op2) => (o, op1, op2),
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

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (f0, f1) = if flevel == level {
        collect_children(fnode.unwrap_inner())
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (g0, g1) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g.borrowed(), g.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_bin::<M, T, OP>(manager, f0, g0)?);
    let e = EdgeDropGuard::new(manager, apply_bin::<M, T, OP>(manager, f1, g1)?);
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), operator)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, operator, &[op1, op2], h.borrowed());

    Ok(h)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
///
/// `f` must be a 0-1-valued MTBDD (see [`PseudoBooleanFunction::ite_edge`]).
/// As an extension of the classical restriction, terminals of `f` other than
/// `0`, `1`, and NaN are treated as "truthy" (`debug_assert`-ed against,
/// since this indicates a violation of the documented precondition), and NaN
/// propagates like in the other operators of this module.
fn apply_ite<M, T>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Ite);

    // The condition is irrelevant if both branches agree.
    if g == h {
        return Ok(manager.clone_edge(&g));
    }

    // Terminal cases for `f`. We decide as soon as `f` resolves to a
    // terminal, which is what makes this a 0-1-valued-condition restricted
    // "ite", as opposed to a fully generic ternary operator.
    if let Node::Terminal(t) = manager.get_node(&f) {
        let t = t.borrow();
        return Ok(if t.is_zero() {
            manager.clone_edge(&h)
        } else if t.is_nan() {
            manager.get_terminal(T::nan())?
        } else {
            debug_assert!(t.is_one(), "the condition of `ite` must be 0-1-valued");
            manager.clone_edge(&g)
        });
    }

    // Query apply cache
    stat!(cache_query MTBDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        MTBDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit MTBDDOp::Ite);
        return Ok(res);
    }

    // `f` is not a terminal (handled above), so it is safe to unwrap it.
    let fnode = manager.get_node(&f).unwrap_inner();
    let gnode = manager.get_node(&g);
    let hnode = manager.get_node(&h);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = flevel.min(glevel).min(hlevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (ht, he) = if hlevel == level {
        collect_children(hnode.unwrap_inner())
    } else {
        (h.borrowed(), h.borrowed())
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

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasMTBDDOpApplyCache<M: Manager>: HasApplyCache<M, MTBDDOp> {}
impl<M: Manager + HasApplyCache<M, MTBDDOp>> HasMTBDDOpApplyCache<M> for M {}

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
    for<'id> F::Manager<'id>: Manager<Terminal = T> + HasMTBDDOpApplyCache<F::Manager<'id>>,
    for<'id> INodeOfFunc<'id, F>: HasLevel,
{
    type Number = T;

    #[inline]
    fn constant_edge<'id>(
        manager: &Self::Manager<'id>,
        value: Self::Number,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        manager.get_terminal(value)
    }

    #[inline]
    fn var_edge<'id>(
        manager: &Self::Manager<'id>,
        var: VarNo,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let level = manager.var_to_level(var);
        let t = EdgeDropGuard::new(manager, manager.get_terminal(T::one())?);
        let e = EdgeDropGuard::new(manager, manager.get_terminal(T::zero())?);
        oxidd_core::LevelView::get_or_insert(
            &mut manager.level(level),
            InnerNode::new(level, [t.into_edge(), e.into_edge()]),
        )
    }

    #[inline]
    fn add_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Add as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn sub_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Sub as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn mul_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Mul as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn div_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Div as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn min_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Min as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn max_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Max as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_ite::<_, T>(
            manager,
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }

    #[inline]
    fn eval_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (VarNo, bool)>,
    ) -> T {
        // `choices` maps levels to the child number to choose
        let mut choices = FixedBitSet::with_capacity(manager.num_levels() as usize);
        for (var, val) in args {
            // child 0 is "then"/"true", hence the negation
            choices.set(manager.var_to_level(var) as usize, !val);
        }

        #[inline] // this function is tail-recursive
        fn inner<M, T: Clone>(manager: &M, edge: Borrowed<M::Edge>, choices: &FixedBitSet) -> T
        where
            M: Manager<Terminal = T>,
            M::InnerNode: HasLevel,
        {
            match manager.get_node(&edge) {
                Node::Inner(node) => {
                    let edge = node.child(choices.contains(node.level() as usize) as usize);
                    inner(manager, edge, choices)
                }
                Node::Terminal(t) => t.borrow().clone(),
            }
        }

        inner(manager, edge.borrowed(), &choices)
    }
}

impl<F: Function, T: Tag> DotStyle<T> for MTBDDFunction<F> {}
