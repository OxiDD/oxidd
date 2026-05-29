//! List decision diagrams (LDDs) for OxiDD.
//!
//!

use std::{borrow::Borrow, cmp::Ordering, hash::Hash};

use oxidd_core::{
    function::{EdgeOfFunc, Function},
    util::{AllocResult, Borrowed, EdgeDropGuard},
    ApplyCache, DiagramRules, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager,
    ManagerRef, Node, ReducedOrNew,
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
trait LDDValue: Clone + Ord + Eq + Hash {
    /// The value used to indicate "included" in a projection meta-LDD.
    fn true_value() -> Self;

    /// The value used to indicate "excluded" in a projection meta-LDD, and
    /// "skip" (neither read nor write) in a relation-product meta-LDD.
    fn false_value() -> Self;

    /// Value encoding a read-only position in a relation-product meta-LDD.
    fn read_only_value() -> Self;

    /// Value encoding a write-only position in a relation-product meta-LDD.
    fn write_only_value() -> Self;

    /// Value encoding the *read* half of a read+write position pair in a
    /// relation-product meta-LDD.
    fn read_of_pair_value() -> Self;

    /// Value encoding the *write* half of a read+write position pair in a
    /// relation-product meta-LDD.
    fn write_of_pair_value() -> Self;
}

/// The default LDDValue to be used.
impl LDDValue for u32 {
    #[inline(always)]
    fn true_value() -> Self {
        1
    }

    #[inline(always)]
    fn false_value() -> Self {
        0
    }

    #[inline(always)]
    fn read_only_value() -> Self {
        1
    }

    #[inline(always)]
    fn write_only_value() -> Self {
        2
    }

    #[inline(always)]
    fn read_of_pair_value() -> Self {
        3
    }

    #[inline(always)]
    fn write_of_pair_value() -> Self {
        4
    }
}

trait LDDManager:
    Manager<Terminal = LDDTerminal, InnerNodeValue: LDDValue, InnerNode: HasLevel>
    + HasApplyCache<Self, LDDOp>
{
}
impl<M> LDDManager for M
where
    M: Manager<Terminal = LDDTerminal> + HasApplyCache<M, LDDOp>,
    M::InnerNodeValue: LDDValue,
    M::InnerNode: HasLevel,
{
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

impl<F: Function> LDDFunction<F>
where
    for<'id> F::Manager<'id>: LDDManager,
{
    pub fn union(&self, other: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            Ok(Self::from_edge(
                manager,
                Self::union_edge(
                    manager,
                    manager.clone_edge(self.as_edge(manager)),
                    manager.clone_edge(other.as_edge(manager)),
                )?,
            ))
        })
    }

    pub fn relation_product_meta<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        read_proj: &[u32],
        write_proj: &[u32],
    ) -> AllocResult<(
        <<LDDFunction<F> as Function>::Manager<'id> as Manager>::Edge,
        Vec<usize>,
        Vec<usize>,
    )> {
        // Compute length of meta.
        let length = std::cmp::max(
            match read_proj.iter().max() {
                Some(x) => *x + 1,
                None => 0,
            },
            match write_proj.iter().max() {
                Some(x) => *x + 1,
                None => 0,
            },
        );

        // Convert projection vectors to meta.
        let mut read_positions = Vec::new();
        let mut write_positions = Vec::new();

        let mut meta: Vec<<<LDDFunction<F> as Function>::Manager<'id> as Manager>::InnerNodeValue> =
            Vec::new();

        let mut offset: usize = 0;
        for i in 0..length {
            let read = read_proj.contains(&i);
            let write = write_proj.contains(&i);
            if read && write {
                meta.push(<_ as LDDValue>::read_of_pair_value());
                meta.push(<_ as LDDValue>::write_of_pair_value());
                read_positions.push(offset);
                write_positions.push(offset + 1);
                offset += 2;
            } else if read {
                meta.push(<_ as LDDValue>::read_only_value());
                read_positions.push(offset);
                offset += 1;
            } else if write {
                meta.push(<_ as LDDValue>::write_only_value());
                write_positions.push(offset);
                offset += 1;
            } else {
                meta.push(<_ as LDDValue>::false_value());
            }
        }

        Ok((singleton(manager, &meta)?, read_positions, write_positions))
    }

    #[inline]
    pub fn empty_set<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
    ) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            manager.get_terminal(LDDTerminal::Empty)?,
        ))
    }

    #[inline]
    pub fn empty_vector<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
    ) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            manager.get_terminal(LDDTerminal::True)?,
        ))
    }

    /// Returns an LDD containing only the given vector, i.e., { vector }.
    #[inline]
    pub fn singleton<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        vector: &[<<LDDFunction<F> as Function>::Manager<'id> as Manager>::InnerNodeValue],
    ) -> AllocResult<LDDFunction<F>> {
        Ok(Self::from_edge(manager, singleton(manager, vector)?))
    }

    #[inline]
    pub fn singleton_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        vector: &[<<LDDFunction<F> as Function>::Manager<'id> as Manager>::InnerNodeValue],
    ) -> AllocResult<<<LDDFunction<F> as Function>::Manager<'id> as Manager>::Edge> {
        singleton(manager, vector)
    }

    #[inline]
    pub fn union_edge<'id>(
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
    let mut root = manager.get_terminal(LDDTerminal::True)?;

    for val in vector.iter().rev() {
        root = make_node(
            manager,
            val,
            root,
            manager.get_terminal(LDDTerminal::Empty)?,
        )?;
    }

    Ok(root)
}

/// Computes a meta LDD that is suitable for the [project] function from the
/// given projection indices.
///
/// This function is useful to be able to cache the projection LDD instead of
/// computing it from the projection array every time.
fn compute_proj<M: LDDManager>(storage: &mut M, proj: &[u32]) -> AllocResult<M::Edge> {
    // Compute length of proj.
    let length = match proj.iter().max() {
        Some(x) => *x + 1,
        None => 0,
    };

    // Convert projection vectors to meta information.
    let mut result: Vec<M::InnerNodeValue> = Vec::new();
    for i in 0..length {
        let included = proj.contains(&i);

        if included {
            result.push(M::InnerNodeValue::true_value());
        } else {
            result.push(M::InnerNodeValue::false_value());
        }
    }

    singleton(storage, &result)
}

/// Computes the set of vectors projected onto the given indices, where proj is equal to compute_proj([i_0, ..., i_k]).
///
/// Formally, for a single vector <x_0, ..., x_n> we have that:
///     - project(<x_0, ..., x_n>, i_0 < ... < i_k) = <x_(i_0), ..., x_(i_k)>
///     - project(X, i_0 < ... < i_k) = { project(x, i_0 < ... < i_k) | x in X }.
///
/// Note that the indices are sorted in the definition, but compute_proj
/// can take any array and ignores both duplicates and order. Also, it
/// follows that i_k must be smaller than or equal to n as x_(i_k) is not
/// defined otherwise.
fn project<M: LDDManager>(
    manager: &M,
    set: Borrowed<M::Edge>,
    proj: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    // Base case: if proj has reached the True terminal, the projection is
    // fully consumed — return the True terminal (empty vector).
    let proj_node = match manager.get_node(&proj) {
        Node::Terminal(terminal) => {
            if *terminal.borrow() == LDDTerminal::True {
                return manager.get_terminal(LDDTerminal::True);
            }
            unreachable!("proj should never be the Empty terminal");
        }
        Node::Inner(proj_node) => proj_node,
    };

    // Base case: if set is the Empty terminal, the result is the empty set.
    let set_node = match manager.get_node(&set) {
        Node::Terminal(terminal) => {
            if *terminal.borrow() == LDDTerminal::Empty {
                return manager.get_terminal(LDDTerminal::Empty);
            }
            // set is the True terminal (empty vector) but proj still has
            // levels — proj extends beyond the length of set.
            debug_assert!(false, "proj can be at most as high as set");
            unreachable!();
        }
        Node::Inner(set_node) => set_node,
    };

    let proj_value = proj_node.get_value();
    let (proj_down, _proj_right) = collect_children(proj_node);
    let set_value = set_node.get_value();
    let (set_down, set_right) = collect_children(set_node);

    if *proj_value == M::InnerNodeValue::false_value() {
        // This position is not in the projection: skip it by unioning the
        // projected right branch (same proj level) with the projected down
        // branch (advance proj level).
        let right_result =
            EdgeDropGuard::new(manager, project(manager, set_right, proj.borrowed())?);
        let down_result = EdgeDropGuard::new(manager, project(manager, set_down, proj_down)?);
        apply_union(manager, right_result.borrowed(), down_result.borrowed())
    } else if *proj_value == M::InnerNodeValue::true_value() {
        // This position is in the projection: keep the current value and
        // recurse on both branches.
        let right_result =
            EdgeDropGuard::new(manager, project(manager, set_right, proj.borrowed())?);
        let down_result = EdgeDropGuard::new(manager, project(manager, set_down, proj_down)?);

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            // The down sub-result is empty — nothing to insert, return right only.
            Ok(right_result.into_edge())
        } else {
            make_node(
                manager,
                set_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )
        }
    } else {
        panic!("proj has unexpected value");
    }
}

/// Recursively apply the 'union' operator to `f` and `g`
fn apply_union<M: LDDManager>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    if f == g {
        return Ok(manager.clone_edge(&f));
    }

    // Query apply cache
    stat!(cache_query LDDOp::Union);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Union, &[f.borrowed(), g.borrowed()])
    {
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

            let right = apply_union(manager, f_right, g.borrowed())?;
            make_node(
                manager,
                f_node.get_value(),
                manager.clone_edge(&f_down),
                right,
            )
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

            let right = apply_union(manager, f.borrowed(), g_right)?;
            make_node(
                manager,
                g_node.get_value(),
                manager.clone_edge(&g_down),
                right,
            )
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
