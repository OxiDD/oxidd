//! Recursive single-threaded apply algorithms

use std::borrow::Borrow;

use oxidd_core::function::{EdgeOfFunc, Function, TVLFunction};
use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::Manager;
use oxidd_core::Node;
use oxidd_core::Tag;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{collect_children, reduce, stat, terminal_bin, Operation, TDDOp, TDDTerminal};

// spell-checker:ignore fnode,gnode,hnode,flevel,glevel,hlevel,ghlevel

/// Recursively apply the 'not' operator to `f`
fn apply_not<M>(manager: &M, f: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = TDDTerminal> + HasApplyCache<M, TDDOp>,
    M::InnerNode: HasLevel,
{
    stat!(call TDDOp::Not);
    let node = match manager.get_node(&f) {
        Node::Inner(node) => node,
        Node::Terminal(t) => return Ok(manager.get_terminal(!*t.borrow()).unwrap()),
    };

    // Query apply cache
    stat!(cache_query TDDOp::Not);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, TDDOp::Not, &[f.borrowed()])
    {
        stat!(cache_hit TDDOp::Not);
        return Ok(h);
    }

    let (f0, f1, f2) = collect_children(node);
    let level = node.level();

    let t = EdgeDropGuard::new(manager, apply_not(manager, f0)?);
    let u = EdgeDropGuard::new(manager, apply_not(manager, f1)?);
    let e = EdgeDropGuard::new(manager, apply_not(manager, f2)?);
    let h = reduce(
        manager,
        level,
        t.into_edge(),
        u.into_edge(),
        e.into_edge(),
        TDDOp::Not,
    )?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, TDDOp::Not, &[f.borrowed()], h.borrowed());

    Ok(h)
}

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M, const OP: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = TDDTerminal> + HasApplyCache<M, TDDOp>,
    M::InnerNode: HasLevel,
{
    stat!(call OP);
    let (operator, op1, op2) = match terminal_bin::<M, OP>(manager, &f, &g) {
        Operation::Binary(o, op1, op2) => (o, op1, op2),
        Operation::Not(f) => {
            return apply_not(manager, f);
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

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (f0, f1, f2) = if flevel == level {
        collect_children(fnode.unwrap_inner())
    } else {
        (f.borrowed(), f.borrowed(), f.borrowed())
    };
    let (g0, g1, g2) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g.borrowed(), g.borrowed(), g.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f0, g0)?);
    let u = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f1, g1)?);
    let e = EdgeDropGuard::new(manager, apply_bin::<M, OP>(manager, f2, g2)?);
    let h = reduce(
        manager,
        level,
        t.into_edge(),
        u.into_edge(),
        e.into_edge(),
        operator,
    )?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, operator, &[op1, op2], h.borrowed());

    Ok(h)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
fn apply_ite_rec<M>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = TDDTerminal> + HasApplyCache<M, TDDOp>,
    M::InnerNode: HasLevel,
{
    use TDDTerminal::*;
    stat!(call TDDOp::Ite);

    // Terminal cases
    if g == h {
        return Ok(manager.clone_edge(&g));
    }
    if f == g {
        return apply_bin::<M, { TDDOp::Or as u8 }>(manager, f, h);
    }
    if f == h {
        return apply_bin::<M, { TDDOp::And as u8 }>(manager, f, g);
    }
    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let hnode = manager.get_node(&h);
    if let Node::Terminal(t) = fnode {
        let t = *t.borrow();
        if t != Unknown {
            return Ok(manager.clone_edge(&*if t == True { g } else { h }));
        } else if gnode.is_any_terminal() && hnode.is_any_terminal() {
            return Ok(manager.get_terminal(Unknown).unwrap());
        }
    }
    match (manager.get_node(&g), manager.get_node(&h)) {
        (Node::Terminal(t), Node::Inner(_)) => match *t.borrow() {
            True => return apply_bin::<M, { TDDOp::Or as u8 }>(manager, f, h),
            Unknown => {}
            False => return apply_bin::<M, { TDDOp::ImpStrict as u8 }>(manager, f, h),
        },
        (Node::Inner(_), Node::Terminal(t)) => match *t.borrow() {
            True => return apply_bin::<M, { TDDOp::Imp as u8 }>(manager, f, g),
            Unknown => {}
            False => return apply_bin::<M, { TDDOp::And as u8 }>(manager, f, g),
        },
        (Node::Terminal(gt), Node::Terminal(ht)) => {
            match (*gt.borrow(), *ht.borrow()) {
                (False, True) => return apply_not(manager, f),
                (True, False) => return Ok(manager.clone_edge(&f)),
                _ => {}
            };
        }
        _ => {}
    };

    // Query apply cache
    stat!(cache_query TDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        TDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit TDDOp::Ite);
        return Ok(res);
    }

    // Get the top-most level of the three
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = std::cmp::min(std::cmp::min(flevel, glevel), hlevel);

    // Collect cofactors of all top-most nodes
    let (f0, f1, f2) = if flevel == level {
        collect_children(fnode.unwrap_inner())
    } else {
        (f.borrowed(), f.borrowed(), f.borrowed())
    };
    let (g0, g1, g2) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g.borrowed(), g.borrowed(), g.borrowed())
    };
    let (h0, h1, h2) = if hlevel == level {
        collect_children(hnode.unwrap_inner())
    } else {
        (h.borrowed(), h.borrowed(), h.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_ite_rec(manager, f0, g0, h0)?);
    let u = EdgeDropGuard::new(manager, apply_ite_rec(manager, f1, g1, h1)?);
    let e = EdgeDropGuard::new(manager, apply_ite_rec(manager, f2, g2, h2)?);
    let res = reduce(
        manager,
        level,
        t.into_edge(),
        u.into_edge(),
        e.into_edge(),
        TDDOp::Ite,
    )?;

    manager
        .apply_cache()
        .add(manager, TDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasTDDOpApplyCache<M: Manager>: HasApplyCache<M, TDDOp> {}
impl<M: Manager + HasApplyCache<M, TDDOp>> HasTDDOpApplyCache<M> for M {}

/// Three value logic function backed by a ternary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr(transparent)]
pub struct TDDFunction<F: Function>(F);

impl<F: Function> From<F> for TDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        TDDFunction(value)
    }
}

impl<F: Function> TDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function> TVLFunction for TDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = TDDTerminal> + HasTDDOpApplyCache<F::Manager<'id>>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    #[inline]
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let f0 = manager.get_terminal(TDDTerminal::True).unwrap();
        let f1 = manager.get_terminal(TDDTerminal::Unknown).unwrap();
        let f2 = manager.get_terminal(TDDTerminal::False).unwrap();
        let edge = manager.add_level(|level| InnerNode::new(level, [f0, f1, f2]))?;
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(TDDTerminal::False).unwrap()
    }
    #[inline]
    fn u_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(TDDTerminal::Unknown).unwrap()
    }
    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(TDDTerminal::True).unwrap()
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_not(manager, edge.borrowed())
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::And as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::Or as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::Nand as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::Nor as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::Xor as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::Equiv as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::Imp as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, { TDDOp::ImpStrict as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_ite_rec(
            manager,
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }
}

impl<F: Function, T: Tag> DotStyle<T> for TDDFunction<F> {
    fn edge_style(
        no: usize,
        _tag: T,
    ) -> (oxidd_dump::dot::EdgeStyle, bool, oxidd_dump::dot::Color) {
        (
            if no == 0 {
                oxidd_dump::dot::EdgeStyle::Solid
            } else if no == 1 {
                oxidd_dump::dot::EdgeStyle::Dotted
            } else {
                oxidd_dump::dot::EdgeStyle::Dashed
            },
            false,
            oxidd_dump::dot::Color::BLACK,
        )
    }
}
