//! Recursive single-threaded apply algorithms

use std::borrow::Borrow;

use fixedbitset::FixedBitSet;

use oxidd_core::function::{EdgeOfFunc, Function, INodeOfFunc, NumberBase, PseudoBooleanFunction};
use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::{
    ApplyCache, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager, Node, Tag, VarNo,
};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{MTBDDOp, Operation, collect_children, reduce, stat};

// spell-checker:ignore fnode,gnode,vnode,flevel,glevel,vlevel

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

/// Result of [`restrict_inner()`]
enum RestrictInnerResult<'a, M: Manager> {
    Done(M::Edge),
    Rec {
        vars: Borrowed<'a, M::Edge>,
        f: Borrowed<'a, M::Edge>,
        fnode: &'a M::InnerNode,
    },
}

/// Tail-recursive part of [`restrict()`]. `f` is the function of which the
/// variables should be restricted to constant values according to `vars`.
///
/// Invariant: `f` points to `fnode` at `flevel`, `vars` points to `vnode`
#[inline]
fn restrict_inner<'a, M, T>(
    manager: &'a M,
    f: Borrowed<'a, M::Edge>,
    fnode: &'a M::InnerNode,
    flevel: LevelNo,
    vars: Borrowed<'a, M::Edge>,
    vnode: &'a M::InnerNode,
) -> RestrictInnerResult<'a, M>
where
    M: Manager<Terminal = T>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
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
            Node::Terminal(t) if t.borrow().is_one() => {
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
                matches!(manager.get_node(&vnode.child(1)), Node::Terminal(t) if t.borrow().is_zero()),
                "vars must be a conjunction of literals"
            );
            // positive literal ⇒ select then branch
            (fnode.child(0), vt, n)
        }
        Node::Terminal(t) if t.borrow().is_one() => {
            debug_assert!(
                matches!(manager.get_node(&vnode.child(1)), Node::Terminal(t) if t.borrow().is_zero()),
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

/// Recursively restrict a set of `vars` (a conjunction of literals) to
/// constant values in `f`
fn restrict<M, T>(
    manager: &M,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Restrict);

    let (Node::Inner(fnode), Node::Inner(vnode)) = (manager.get_node(&f), manager.get_node(&vars))
    else {
        return Ok(manager.clone_edge(&f));
    };

    match restrict_inner(manager, f, fnode, fnode.level(), vars, vnode) {
        RestrictInnerResult::Done(res) => Ok(res),
        RestrictInnerResult::Rec { vars, f, fnode } => {
            // f above top-most restrict variable

            // Query apply cache
            stat!(cache_query MTBDDOp::Restrict);
            if let Some(res) = manager.apply_cache().get(
                manager,
                MTBDDOp::Restrict,
                &[f.borrowed(), vars.borrowed()],
            ) {
                stat!(cache_hit MTBDDOp::Restrict);
                return Ok(res);
            }

            let (ft, fe) = collect_children(fnode);
            let t = EdgeDropGuard::new(manager, restrict(manager, ft, vars.borrowed())?);
            let e = EdgeDropGuard::new(manager, restrict(manager, fe, vars.borrowed())?);
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
    }
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
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        restrict::<_, T>(manager, root.borrowed(), vars.borrowed())
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
