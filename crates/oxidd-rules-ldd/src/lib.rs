//! List decision diagrams (LDDs) for OxiDD.
//!
//!

use std::{borrow::Borrow, cmp::Ordering, hash::Hash};

use oxidd_core::{
    ApplyCache, DiagramRules, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager, Node, ReducedOrNew, function::{EdgeOfFunc, Function}, util::{AllocResult, Borrowed}
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

    Project,
}

/// For LDDs it is essential that values are ordered and cloneable.
trait LDDValue: Ord + Eq + Hash {}

trait LDDManager: Manager<Terminal = LDDTerminal, InnerNodeValue: LDDValue, InnerNode: HasLevel> + HasApplyCache<Self, LDDOp>
{}
impl<M> LDDManager for M
where
    M: Manager<Terminal = LDDTerminal> + HasApplyCache<M, LDDOp>,
    M::InnerNodeValue: LDDValue,
    M::InnerNode: HasLevel,
{}


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

impl<F: Function> LDDFunction<F>
where
    for<'id> F::Manager<'id>: LDDManager,
{
    #[inline]
    fn union_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        f: EdgeOfFunc<'id, Self>,
        g: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_union(manager, f.borrowed(), g.borrowed())
    }
}

/// [`DiagramRules`] for list decision diagrams
pub struct LDDRules;

impl<E: Edge, N: InnerNode<E, Value: LDDValue>> DiagramRules<E, N, LDDTerminal> for LDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline(always)]
    fn reduce<M: Manager<Edge = E, InnerNode = N>>(
        _manager: &M,
        _level: LevelNo,
        _children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        unimplemented!("Missing the value to construct a new node");
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

/// Return a singleton LDD that contains only the list represented by `vector`.
fn singleton<M: LDDManager>(manager: &M, vector: &[M::InnerNodeValue]) -> AllocResult<M::Edge> {
    let mut root =  manager.get_terminal(LDDTerminal::True).unwrap();

    for val in vector.iter().rev() {
        root = make_node(manager, val, root, manager.get_terminal(LDDTerminal::Empty).unwrap())?;
    }

    Ok(root)
}

/// Computes a meta LDD that is suitable for the [project] function from the
/// given projection indices.
///
/// This function is useful to be able to cache the projection LDD instead of
/// computing it from the projection array every time.
// pub fn compute_proj<M: LDDManager>(storage: &mut M, proj: &[M::InnerNodeValue]) -> AllocResult<M::Edge> {
//     // Compute length of proj.
//     let length = match proj.iter().max() {
//         Some(x) => *x + 1,
//         None => 0,
//     };

//     // Convert projection vectors to meta information.
//     let mut result: Vec<M::InnerNodeValue> = Vec::new();
//     for i in 0..length {
//         let included = proj.contains(&i);

//         if included {
//             result.push(1);
//         } else {
//             result.push(0);
//         }
//     }

//     singleton(storage, &result)
// }

/// Recursively apply the 'union' operator to `f` and `g`
fn apply_union<M: LDDManager>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
{
    if f == g {
        return Ok(manager.clone_edge(&f));
    }

    // Query apply cache
    stat!(cache_query LDDOp::Union);
    if let Some(res) = manager.apply_cache().get(
        manager,
        LDDOp::Union,
        &[f.borrowed(), g.borrowed()],
    ) {
        stat!(cache_hit LDDOp::Union);
        return Ok(res);
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
            make_node(manager, f_node.get_value(), manager.clone_edge(&f_down), right)
        }
        Ordering::Greater => {
            let (f_down, f_right) = collect_children(f_node);
            let (g_down, g_right) = collect_children(g_node);

            let low = apply_union(manager, f_down, g_down)?;
            let high = apply_union(manager, f_right, g_right)?;
            make_node(manager, g_node.get_value(), low, high)
        }
        Ordering::Equal => {
            let (g_down, g_right) = collect_children(g_node);

            let right = apply_union(manager, f, g_right)?;
            make_node(manager, g_node.get_value(), manager.clone_edge(&g_down), right)
        }
    }?;

    manager
        .apply_cache()
        .add(manager, LDDOp::Union, &[f, g], result.borrowed());

    Ok(result)
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
    down: M::Edge,
    right: M::Edge,
) -> AllocResult<M::Edge> {
    oxidd_core::LevelView::get_or_insert(
        &mut manager.level(0),
        InnerNode::new(0, [down, right], value.clone()),
    )
}

macro_rules! stat {
    (call $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .calls
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (cache_query $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .cache_queries
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (cache_hit $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .cache_hits
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (reduced $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .reduced
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
}

pub(crate) use stat;