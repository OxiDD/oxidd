//! Recursive single-threaded apply algorithms

use oxidd_core::function::Function;
use oxidd_core::function::PseudoBooleanFunction;
use oxidd_core::util::Borrowed;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::HasLevel;
use oxidd_core::Manager;
use oxidd_core::Tag;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use super::*;

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
pub(super) fn apply_bin<M, T, const OP: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, Operator = MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call OP);
    let (operator, op1, op2) = match terminal_bin::<M, T, OP>(manager, &f, &g)? {
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

// --- Function Interface ------------------------------------------------------

/// Boolean function backed by a binary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
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
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    type Number = T;

    #[inline]
    fn constant_edge<'id>(
        manager: &Self::Manager<'id>,
        value: Self::Number,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        manager.get_terminal(value)
    }

    #[inline]
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let t = EdgeDropGuard::new(manager, manager.get_terminal(T::one())?);
        let e = EdgeDropGuard::new(manager, manager.get_terminal(T::zero())?);
        let children = [t.into_edge(), e.into_edge()];
        let edge = manager.add_level(|level| InnerNode::new(level, children))?;
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn add_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, T, { MTBDDOp::Add as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn sub_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, T, { MTBDDOp::Sub as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn mul_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, T, { MTBDDOp::Mul as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn div_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, T, { MTBDDOp::Div as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn min_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, T, { MTBDDOp::Min as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn max_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, T, { MTBDDOp::Max as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
}

impl<F: Function, T: Tag> DotStyle<T> for MTBDDFunction<F> {}
