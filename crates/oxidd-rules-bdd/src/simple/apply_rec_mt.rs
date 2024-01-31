//! Recursive, multi-threaded apply algorithms

use std::collections::HashMap;

use oxidd_core::function::BooleanFunction;
use oxidd_core::function::BooleanFunctionQuant;
use oxidd_core::function::Function;
use oxidd_core::util::Borrowed;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::util::OptBool;
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
use oxidd_core::WorkerManager;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use super::*;

// spell-checker:ignore fnode,gnode,hnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the 'not' operator to `f`
///
/// `depth` is decremented for each recursive call. If it reaches 0, this
/// function simply calls [`apply_not_rec()`].
fn apply_not<M>(manager: &M, depth: u32, f: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, Operator = BDDOp> + WorkerManager,
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

    let (f0, f1) = collect_children(node);
    let level = node.level();

    let d = depth - 1;
    let (t, e) = manager.join(
        || Ok(EdgeDropGuard::new(manager, apply_not(manager, d, f0)?)),
        || Ok(EdgeDropGuard::new(manager, apply_not(manager, d, f1)?)),
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
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, Operator = BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::apply_bin::<M, OP>(manager, f, g);
    }
    stat!(call OP);
    let (operator, op1, op2) = match terminal_bin::<M, OP>(manager, &f, &g) {
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
    let (f0, f1) = if flevel == level {
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (g0, g1) = if glevel == level {
        collect_children(gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };

    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = apply_bin::<M, OP>(manager, d, f0, g0)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = apply_bin::<M, OP>(manager, d, f1, g1)?;
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
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, Operator = BDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use BDDTerminal::*;
    if depth == 0 {
        return apply_rec_st::apply_ite_rec(manager, f, g, h);
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
    let (f0, f1) = if flevel == level {
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (g0, g1) = if glevel == level {
        collect_children(gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (h0, h1) = if hlevel == level {
        collect_children(hnode)
    } else {
        (h.borrowed(), h.borrowed())
    };

    let d = depth - 1;
    let (t, e) = manager.join(
        || {
            let t = apply_ite(manager, d, f0, g0, h0)?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = apply_ite(manager, d, f1, g1, h1)?;
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

fn quant_rec<M, const Q: u8>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal> + HasApplyCache<M, Operator = BDDOp> + WorkerManager,
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
        Node::Terminal(_) => return Ok(manager.clone_edge(&f)),
    };
    let flevel = fnode.level();

    // We can ignore all variables above the top-most variable. Removing them
    // before querying the apply cache should increase the hit ratio by a lot.
    let vars = crate::set_pop(manager, vars, flevel);
    let vlevel = match manager.get_node(&vars) {
        Node::Inner(n) => n.level(),
        Node::Terminal(_) => return Ok(manager.clone_edge(&f)),
    };
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

    let d = depth - 1;
    let (f0, f1) = collect_children(fnode);
    let (t, e) = manager.join(
        || {
            let t = quant_rec::<M, Q>(manager, d, f0, vars.borrowed())?;
            Ok(EdgeDropGuard::new(manager, t))
        },
        || {
            let e = quant_rec::<M, Q>(manager, d, f1, vars.borrowed())?;
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
    for<'id> F::Manager<'id>:
        Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>> + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    #[inline]
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let f0 = manager.get_terminal(BDDTerminal::True).unwrap();
        let f1 = manager.get_terminal(BDDTerminal::False).unwrap();
        let edge = manager.add_level(|level| InnerNode::new(level, [f0, f1]))?;
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        manager.get_terminal(BDDTerminal::False).unwrap()
    }
    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        manager.get_terminal(BDDTerminal::True).unwrap()
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_not(manager, Self::init_depth(manager), edge.borrowed())
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        if_edge: &<Self::Manager<'id> as Manager>::Edge,
        then_edge: &<Self::Manager<'id> as Manager>::Edge,
        else_edge: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
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
        edge: &<Self::Manager<'id> as Manager>::Edge,
        vars: LevelNo,
        cache: &mut HashMap<NodeID, N, S>,
    ) -> N {
        BDDFunction::sat_count_edge(manager, edge, vars, cache)
    }

    #[inline]
    fn pick_cube_edge<'id, 'a, I>(
        manager: &'a Self::Manager<'id>,
        edge: &'a <Self::Manager<'id> as Manager>::Edge,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &<Self::Manager<'id> as Manager>::Edge) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a <Self::Manager<'id> as Manager>::Edge>,
    {
        BDDFunction::pick_cube_edge(manager, edge, order, choice)
    }

    #[inline]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a <Self::Manager<'id> as Manager>::Edge,
        env: impl IntoIterator<Item = (&'a <Self::Manager<'id> as Manager>::Edge, bool)>,
    ) -> bool {
        BDDFunction::eval_edge(manager, edge, env)
    }
}

impl<F: Function> BooleanFunctionQuant for BDDFunctionMT<F>
where
    for<'id> F::Manager<'id>:
        Manager<Terminal = BDDTerminal> + HasBDDOpApplyCache<F::Manager<'id>> + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    #[inline]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &<Self::Manager<'id> as Manager>::Edge,
        vars: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        quant_rec::<_, { BDDOp::And as u8 }>(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }

    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &<Self::Manager<'id> as Manager>::Edge,
        vars: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        quant_rec::<_, { BDDOp::Or as u8 }>(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }

    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &<Self::Manager<'id> as Manager>::Edge,
        vars: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        quant_rec::<_, { BDDOp::Xor as u8 }>(
            manager,
            Self::init_depth(manager),
            root.borrowed(),
            vars.borrowed(),
        )
    }
}

impl<F: Function, T: Tag> DotStyle<T> for BDDFunctionMT<F> {}
