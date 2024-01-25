//! Recursive single-threaded apply algorithms

use std::collections::HashMap;
use std::hash::BuildHasher;

use bitvec::vec::BitVec;

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
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use super::*;

// spell-checker:ignore fnode,gnode,hnode,flevel,glevel,hlevel,vlevel

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
    M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal> + HasApplyCache<M, Operator = CBDDOp>,
    M::InnerNode: HasLevel,
{
    stat!(call OP);
    let (op, f, fnode, g, gnode) = if OP == CBDDOp::And as u8 {
        match terminal_and(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => {
                (CBDDOp::And, f.borrowed(), fnode, g.borrowed(), gnode)
            }
            // `And` is commutative, hence we swap `f` and `g` in the apply
            // cache key if `f > g` to have a unique representation of the set
            // `{f, g}`.
            NodesOrDone::Nodes(fnode, gnode) => {
                (CBDDOp::And, g.borrowed(), gnode, f.borrowed(), fnode)
            }
            NodesOrDone::Done(h) => return Ok(h),
        }
    } else {
        assert_eq!(OP, CBDDOp::Xor as u8);
        match terminal_xor(manager, &f, &g) {
            NodesOrDone::Nodes(fnode, gnode) if f < g => {
                (CBDDOp::Xor, f.borrowed(), fnode, g.borrowed(), gnode)
            }
            NodesOrDone::Nodes(fnode, gnode) => {
                (CBDDOp::Xor, g.borrowed(), gnode, f.borrowed(), fnode)
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
    let (f0, f1) = if flevel == level {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (g0, g1) = if glevel == level {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f0, g0)?);
    let e = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f1, g1)?);

    let h = reduce(manager, level, t.into_edge(), e.into_edge(), op)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, op, &[f, g], h.borrowed());

    Ok(h)
}

/// Shorthand for `apply_bin_rec::<M, { CBDDOp::And as u8 }>(manager, f, g)`
#[inline(always)]
fn apply_and<M>(manager: &M, f: Borrowed<M::Edge>, g: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal> + HasApplyCache<M, Operator = CBDDOp>,
    M::InnerNode: HasLevel,
{
    apply_bin::<M, { CBDDOp::And as u8 }>(manager, f, g)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
pub(super) fn apply_ite<M>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal> + HasApplyCache<M, Operator = CBDDOp>,
    M::InnerNode: HasLevel,
{
    stat!(call CBDDOp::Ite);

    // Terminal cases
    let gu = g.with_tag(EdgeTag::None); // untagged
    let hu = h.with_tag(EdgeTag::None);
    if gu == hu {
        return Ok(if g.tag() == h.tag() {
            manager.clone_edge(&g)
        } else {
            not_owned(apply_bin::<M, { CBDDOp::Xor as u8 }>(manager, f, g)?) // f ↔ g
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
            Ok(not_owned(apply_and(manager, not(&f), g)?)) // f → g
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
    stat!(cache_query CBDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        CBDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit CBDDOp::Ite);
        return Ok(res);
    }

    // Get the top-most level of the three
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = std::cmp::min(std::cmp::min(flevel, glevel), hlevel);

    // Collect cofactors of all top-most nodes
    let (f0, f1) = if flevel == level {
        collect_cofactors(f.tag(), fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (g0, g1) = if glevel == level {
        collect_cofactors(g.tag(), gnode)
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (h0, h1) = if hlevel == level {
        collect_cofactors(h.tag(), hnode)
    } else {
        (h.borrowed(), h.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_ite(manager, f0, g0, h0)?);
    let e = EdgeDropGuard::new(manager, apply_ite(manager, f1, g1, h1)?);
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), CBDDOp::Ite)?;

    manager
        .apply_cache()
        .add(manager, CBDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

/// Compute the quantification `Q` over `vars`
///
/// `Q` is one of `CBDDOp::Forall`, `CBDDOp::Exist`, and `CBDDOp::Forall` as
/// `u8`.
pub(super) fn quant<M, const Q: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = CBDDTerminal, EdgeTag = EdgeTag> + HasApplyCache<M, Operator = CBDDOp>,
    M::InnerNode: HasLevel,
{
    let operator = match () {
        _ if Q == CBDDOp::Forall as u8 => CBDDOp::Forall,
        _ if Q == CBDDOp::Exist as u8 => CBDDOp::Exist,
        _ if Q == CBDDOp::Unique as u8 => CBDDOp::Unique,
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

    let (f0, f1) = collect_cofactors(f.tag(), fnode);
    let t = EdgeDropGuard::new(manager, quant::<M, Q>(manager, f0, vars.borrowed())?);
    let e = EdgeDropGuard::new(manager, quant::<M, Q>(manager, f1, vars.borrowed())?);

    let res = if flevel == vlevel {
        match operator {
            CBDDOp::Forall => apply_and(manager, t.borrowed(), e.borrowed())?,
            CBDDOp::Exist => not_owned(apply_and(manager, not(&t), not(&e))?),
            CBDDOp::Unique => {
                apply_bin::<M, { CBDDOp::Xor as u8 }>(manager, t.borrowed(), e.borrowed())?
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
pub struct CBDDFunction<F: Function>(F);

impl<F: Function> From<F> for CBDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        CBDDFunction(value)
    }
}

impl<F: Function> CBDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function> BooleanFunction for CBDDFunction<F>
where
    for<'id> F::Manager<'id>:
        Manager<Terminal = CBDDTerminal, EdgeTag = EdgeTag> + HasCBDDOpApplyCache<F::Manager<'id>>,
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
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        get_terminal(manager, false)
    }
    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        get_terminal(manager, true)
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Ok(not_owned(manager.clone_edge(edge)))
    }
    #[inline]
    fn not_edge_owned<'id>(
        _manager: &Self::Manager<'id>,
        edge: <Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Ok(not_owned(edge))
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_and(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Ok(not_owned(apply_and(manager, not(lhs), not(rhs))?))
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Ok(not_owned(Self::and_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_and(manager, not(lhs), not(rhs))
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_bin::<_, { CBDDOp::Xor as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Ok(not_owned(Self::xor_edge(manager, lhs, rhs)?))
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Ok(not_owned(apply_and(manager, lhs.borrowed(), not(rhs))?))
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        apply_and(manager, not(lhs), rhs.borrowed())
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
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }

    #[inline]
    fn sat_count_edge<'id, N: SatCountNumber, S: BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &<Self::Manager<'id> as Manager>::Edge,
        vars: LevelNo,
        cache: &mut HashMap<NodeID, N, S>,
    ) -> N {
        fn inner<M, N: SatCountNumber, S: BuildHasher>(
            manager: &M,
            e: Borrowed<M::Edge>,
            terminal_val: &N,
            cache: &mut HashMap<NodeID, N, S>,
        ) -> N
        where
            M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal>,
        {
            let node = match manager.get_node(&e) {
                Node::Inner(node) => node,
                Node::Terminal(_) => return terminal_val.clone(),
            };

            // query cache
            let node_id = e.node_id();
            if let Some(n) = cache.get(&node_id) {
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
            cache.insert(node_id, n.clone());
            n
        }

        // This function does not use the identity `|f| = num_vars - |f'|` to
        // avoid rounding issues
        fn inner_floating<M, N, S>(
            manager: &M,
            e: Borrowed<M::Edge>,
            terminal_val: &N,
            cache: &mut HashMap<NodeID, N, S>,
        ) -> N
        where
            M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal>,
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
            if let Some(n) = cache.get(&node_id) {
                return n.clone();
            }
            let (e0, e1) = collect_cofactors(tag, node);
            let mut n = inner_floating(manager, e0, terminal_val, cache);
            n += &inner_floating(manager, e1, terminal_val, cache);
            n >>= 1u32;
            cache.insert(node_id, n.clone());
            n
        }

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
        edge: &'a <Self::Manager<'id> as Manager>::Edge,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &<Self::Manager<'id> as Manager>::Edge) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a <Self::Manager<'id> as Manager>::Edge>,
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
            let (t, e) = collect_cofactors(edge.tag(), node);
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
        edge: &'a <Self::Manager<'id> as Manager>::Edge,
        env: impl IntoIterator<Item = (&'a <Self::Manager<'id> as Manager>::Edge, bool)>,
    ) -> bool {
        let mut values = BitVec::new();
        values.resize(manager.num_levels() as usize, false);
        for (edge, val) in env {
            let node = manager
                .get_node(edge)
                .expect_inner("edges in `env` must refer to inner nodes");
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

impl<F: Function> BooleanFunctionQuant for CBDDFunction<F>
where
    for<'id> F::Manager<'id>:
        Manager<Terminal = CBDDTerminal, EdgeTag = EdgeTag> + HasCBDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    #[inline]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &<Self::Manager<'id> as Manager>::Edge,
        vars: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        quant::<_, { CBDDOp::Forall as u8 }>(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &<Self::Manager<'id> as Manager>::Edge,
        vars: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        quant::<_, { CBDDOp::Exist as u8 }>(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &<Self::Manager<'id> as Manager>::Edge,
        vars: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        quant::<_, { CBDDOp::Unique as u8 }>(manager, root.borrowed(), vars.borrowed())
    }
}

impl<F: Function, T: Tag> DotStyle<T> for CBDDFunction<F> {}
