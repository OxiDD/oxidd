//! List decision diagrams (LDDs) for OxiDD.
//!
//!

use std::{borrow::Borrow, cmp::Ordering, hash::Hash};

use oxidd_core::{
    DiagramRules, Edge, HasApplyCache, InnerNode, LevelNo, Manager, Node, ReducedOrNew, function::Function, util::{AllocResult, Borrowed}
};
use oxidd_derive::{Countable, Function};

/// Terminal nodes in simple binary decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
pub enum LDDTerminal {
    /// This represents the empty set, also denoted by `false`.
    Empty,
    /// This represents the set containing only the empty list, also denoted by `true`.
    True,
}

/// Native operators of this LDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Ord, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum LDDOp {
    Union,
}

/// For LDDs it is essential that values are ordered.
trait LDDValue: Ord + Eq + Hash {}

trait LDDManager: Manager<Terminal = LDDTerminal> + HasApplyCache<Self, LDDOp> {
    type InnerNodeValue: LDDValue;
}
impl<M> LDDManager for M
where
    M: Manager<Terminal = LDDTerminal> + HasApplyCache<Self, LDDOp>,
    M::InnerNodeValue: LDDValue,
{
    type InnerNodeValue = M::InnerNodeValue;
}


// --- Function Interface ------------------------------------------------------

/// Boolean function backed by a list decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr_id = "LDD"]
#[repr(transparent)]
pub struct LDDFunction<F: Function>(F);

impl<F: Function> From<F> for LDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        LDDFunction(value)
    }
}

impl<F: Function> LDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

/// [`DiagramRules`] for list decision diagrams
pub struct LDDRules;

impl<E: Edge, N: InnerNode<E, Value = ()>> DiagramRules<E, N, LDDTerminal> for LDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline(always)]
    fn reduce<M: Manager<Edge = E, InnerNode = N>>(
        _manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let f_then = it.next().unwrap();
        let f_else = it.next().unwrap();
        debug_assert!(it.next().is_none());

        ReducedOrNew::New(N::new(level, [f_then, f_else], ()), Default::default())
    }

    #[inline(always)]
    fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        node.children()
    }

    #[inline(always)]
    fn cofactor(_tag: E::Tag, node: &N, n: usize) -> Borrowed<'_, E> {
        node.child(n)
    }
}

/// Recursively apply the 'union' operator to `f` and `g`
fn apply_union<M: LDDManager>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    <M as Manager>::InnerNodeValue: LDDValue,
{
    if f == g {
        return Ok(manager.clone_edge(&f));
    }

    let f_node = match manager.get_node(&f) {
        Node::Terminal(t) => {
            if *t.borrow() == LDDTerminal::Empty {
                return Ok(manager.clone_edge(&g));
            }

            unreachable!("Invalid terminal");
        }
        Node::Inner(f_node) => f_node.borrow(),
    };

    let g_node = match manager.get_node(&g) {
        Node::Terminal(t) => {
            if *t.borrow() == LDDTerminal::Empty {
                return Ok(manager.clone_edge(&f));
            }

            unreachable!("Invalid terminal");
        }
        Node::Inner(g_node) => g_node.borrow(),
    };

    let result = match f_node.get_value().cmp(g_node.get_value()) {
        Ordering::Less => {
            let (f_down, f_right) = collect_children(f_node);

            let right = apply_union(manager, f_right, g)?;
            make_node(manager, f_node.get_value(), f_down, right.borrowed())
        }
        Ordering::Greater => {
            let (f_down, f_right) = collect_children(f_node);
            let (g_down, g_right) = collect_children(g_node);

            let low = apply_union(manager, f_down, g_down)?;
            let high = apply_union(manager, f_right, g_right)?;
            make_node(manager, g_node.get_value(), low.borrowed(), high.borrowed())
        }
        Ordering::Equal => {
            let (g_down, g_right) = collect_children(g_node);

            let right = apply_union(manager, f, g_right)?;
            make_node(manager, g_node.get_value(), g_down, right.borrowed())
        }
    };

    result
}

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<'_, E>, Borrowed<'_, E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = node.children();
    let f_down = it.next().unwrap();
    let f_right = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (f_down, f_right)
}

/// Create a node in `manager` if necessary
#[inline(always)]
fn make_node<M: LDDManager>(
    manager: &M,
    value: &<M as Manager>::InnerNodeValue,
    down: Borrowed<M::Edge>,
    right: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    unreachable!("Not implemented yet");
    // oxidd_core::LevelView::get_or_insert(&mut manager.level(0), InnerNode::new(0, [*down, *right], value.clone()))
}
