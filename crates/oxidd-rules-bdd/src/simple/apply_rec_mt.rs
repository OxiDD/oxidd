//! Recursive, multi-threaded apply algorithms

use std::borrow::Borrow;

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
use oxidd_core::Tag;
use oxidd_core::WorkerManager;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use crate::stat;

use super::apply_rec_st;
use super::collect_children;
use super::reduce;
use super::BDDOp;
use super::BDDTerminal;
use super::Operation;
#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the 'not' operator to `f`
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_not_rec()`].
fn apply_not<M>(manager: &M, depth: u32, f: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::apply_not(manager, f);
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

    let d = depth - 1;
    let (t, e) = manager.join(
        || Ok(EdgeDropGuard::new(manager, apply_not(manager, d, ft)?)),
        || Ok(EdgeDropGuard::new(manager, apply_not(manager, d, fe)?)),
    );
    let (t, e) = (t?, e?);
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), BDDOp::Not)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, BDDOp::Not, &[f.borrowed()], h.borrowed());

    Ok(h)
}

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_bin_rec()`].
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M, const OP: u8>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::apply_bin::<M, OP>(manager, f, g);
    }
    stat!(call OP);
    let (operator, op1, op2) = match super::terminal_bin::<M, OP>(manager, &f, &g) {
        Operation::Binary(o, op1, op2) => (o, op1, op2),
        Operation::Not(f) => {
            return apply_not(manager, depth - 1, f);
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

    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = apply_bin::<M, OP>(manager, d, ft, gt)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = apply_bin::<M, OP>(manager, d, fe, ge)?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), operator)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, operator, &[op1, op2], h.borrowed());

    Ok(h)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_ite_rec()`].
fn apply_ite<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use BDDTerminal::*;
    if depth == 0 {
        return apply_rec_st::apply_ite(manager, f, g, h);
    }
    stat!(call BDDOp::Ite);

    // Terminal cases
    if g == h {
        return Ok(manager.clone_edge(&g));
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
            return if *t.borrow() == True {
                apply_bin::<M, { BDDOp::Or as u8 }>(manager, depth, f, h)
            } else {
                apply_bin::<M, { BDDOp::ImpStrict as u8 }>(manager, depth, f, h)
            };
        }
        (Node::Inner(_), Node::Terminal(t)) => {
            return if *t.borrow() == True {
                apply_bin::<M, { BDDOp::Imp as u8 }>(manager, depth, f, g)
            } else {
                apply_bin::<M, { BDDOp::And as u8 }>(manager, depth, f, g)
            };
        }
        (Node::Terminal(gt), Node::Terminal(ht)) => {
            return match (*gt.borrow(), *ht.borrow()) {
                (False, False) => Ok(manager.get_terminal(False).unwrap()),
                (False, True) => apply_not(manager, depth, f),
                (True, False) => Ok(manager.clone_edge(&f)),
                (True, True) => Ok(manager.get_terminal(True).unwrap()),
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

    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = apply_ite(manager, d, ft, gt, ht)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = apply_ite(manager, d, fe, ge, he)?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), BDDOp::Ite)?;

    manager
        .apply_cache()
        .add(manager, BDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

fn restrict<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::restrict(manager, f, vars);
    }
    stat!(call BDDOp::Restrict);

    let (Node::Inner(fnode), Node::Inner(vnode)) = (manager.get_node(&f), manager.get_node(&vars))
    else {
        return Ok(manager.clone_edge(&f));
    };

    match apply_rec_st::restrict_inner(manager, f, fnode, fnode.level(), vars, vnode) {
        apply_rec_st::RestrictInnerResult::Done(res) => Ok(res),
        apply_rec_st::RestrictInnerResult::Rec { vars, f, fnode } => {
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
            let d = depth - 1;
            let (t, e) = manager.join(
                || {
                    let t = restrict(manager, d, ft, vars.borrowed())?;
                    Ok(EdgeDropGuard::new(manager, t))
                },
                || {
                    let e = restrict(manager, d, fe, vars.borrowed())?;
                    Ok(EdgeDropGuard::new(manager, e))
                },
            );
            let (t, e) = (t?, e?);

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

fn quant<M, const Q: u8>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::quant::<M, Q>(manager, f, vars);
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

    let d = depth - 1;
    let (ft, fe) = collect_children(fnode);
    let vt = if vlevel == flevel {
        vnode.child(0)
    } else {
        vars.borrowed()
    };
    let (t, e) = manager.join(
        || {
            let t = quant::<M, Q>(manager, d, ft, vt.borrowed())?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = quant::<M, Q>(manager, d, fe, vt.borrowed())?;
            Ok(EdgeDropGuard::new(manager, e))
        },
    );
    let (t, e) = (t?, e?);

    let res = if flevel == vlevel {
        apply_bin::<M, Q>(manager, d, t.borrowed(), e.borrowed())
    } else {
        reduce(manager, flevel, t.into_edge(), e.into_edge(), operator)
    }?;

    manager
        .apply_cache()
        .add(manager, operator, &[f, vars], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

/// Boolean function backed by a binary decision diagram, multi-threaded version
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
    for<'id> F::Manager<'id>: WorkerManager,
{
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }

    fn init_depth(manager: &F::Manager<'_>) -> u32 {
        let n = manager.current_num_threads();
        if n > 1 {
            (4096 * n).ilog2()
        } else {
            0
        }
    }
}

impl<F: Function> BooleanFunction for BDDFunctionMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BDDTerminal>
        + super::HasBDDOpApplyCache<F::Manager<'id>>
        + WorkerManager,
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
        apply_not(manager, Self::init_depth(manager), edge.borrowed())
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::And as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::Or as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::Nand as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::Nor as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::Xor as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::Equiv as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::Imp as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { BDDOp::ImpStrict as u8 }>(
            manager,
            Self::init_depth(manager),
            lhs.borrowed(),
            rhs.borrowed(),
        )
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
            Self::init_depth(manager),
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }

    #[inline]
    fn sat_count_edge<'id, N: SatCountNumber, S: std::hash::BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        apply_rec_st::BDDFunction::<F>::sat_count_edge(manager, edge, vars, cache)
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
        apply_rec_st::BDDFunction::<F>::pick_cube_edge(manager, edge, order, choice)
    }

    #[inline]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
    ) -> bool {
        apply_rec_st::BDDFunction::<F>::eval_edge(manager, edge, args)
    }
}

impl<F: Function> BooleanFunctionQuant for BDDFunctionMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = BDDTerminal>
        + super::HasBDDOpApplyCache<F::Manager<'id>>
        + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    #[inline]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        restrict(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }

    #[inline]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        quant::<_, { BDDOp::And as u8 }>(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }

    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        quant::<_, { BDDOp::Or as u8 }>(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }

    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        quant::<_, { BDDOp::Xor as u8 }>(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }
}

impl<F: Function, T: Tag> DotStyle<T> for BDDFunctionMT<F> {}
