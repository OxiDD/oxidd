//! List decision diagrams (LDDs) for OxiDD.
//!
//!

use std::{borrow::Borrow, cmp::Ordering, collections::HashMap, hash::Hash};

use oxidd_core::{
    function::{EdgeOfFunc, Function},
    util::{AllocResult, Borrowed, EdgeDropGuard},
    ApplyCache, DiagramRules, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager,
    ManagerRef, Node, NodeID, ReducedOrNew,
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

    RelationalProduct,

    Minus,
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

#[allow(private_bounds)]
impl<F: Function> LDDFunction<F>
where
    for<'id> F::Manager<'id>: LDDManager,
{

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

    /// Returns the largest subset of `a` that does not contain any element of
    /// `b` (set difference `a \ b`).
    #[inline]
    pub fn minus_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        a: EdgeOfFunc<'id, Self>,
        b: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_minus(manager, a.borrowed(), b.borrowed())
    }

    /// Computes the set of vectors reachable in one step from `set` via the
    /// sparse relation `rel`.  `meta` must be produced by
    /// [`relation_product_meta`][Self::relation_product_meta].
    #[inline]
    pub fn relational_product_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        rel: EdgeOfFunc<'id, Self>,
        meta: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_relational_product(manager, set.borrowed(), rel.borrowed(), meta.borrowed())
    }

    /// Returns the number of vectors (lists) contained in the set rooted at
    /// `set`.
    #[inline]
    pub fn len_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
    ) -> usize {
        len(manager, set.borrowed(), &mut HashMap::new())
    }

    /// Computes a meta-LDD encoding the projection onto the indices in `proj`,
    /// suitable as the second argument of [`project_edge`][Self::project_edge].
    #[inline]
    pub fn projection_meta<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        proj: &[u32],
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        compute_proj(manager, proj)
    }

    /// Projects the vectors in `set` onto the indices encoded by `proj` (built
    /// via [`projection_meta`][Self::projection_meta]).
    #[inline]
    pub fn project_edge<'id>(
        manager: &<LDDFunction<F> as Function>::Manager<'id>,
        set: EdgeOfFunc<'id, Self>,
        proj: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        project(manager, set.borrowed(), proj.borrowed())
    }

    /// Computes the union `self ∪ other`.
    pub fn union(&self, other: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::union_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(other.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set difference `self \ other`.
    pub fn minus(&self, other: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::minus_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(other.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Computes the set of vectors reachable in one step from `self` via the
    /// sparse relation `rel`, guided by `meta` (produced by
    /// [`relation_product_meta`][Self::relation_product_meta]).
    pub fn relational_product(&self, rel: &Self, meta: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::relational_product_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(rel.as_edge(manager)),
                manager.clone_edge(meta.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Projects the vectors in `self` onto the indices encoded by `proj`
    /// (produced by [`projection_meta`][Self::projection_meta]).
    pub fn project(&self, proj: &Self) -> AllocResult<Self> {
        self.manager_ref().with_manager_shared(|manager| {
            let edge = Self::project_edge(
                manager,
                manager.clone_edge(self.as_edge(manager)),
                manager.clone_edge(proj.as_edge(manager)),
            )?;
            Ok(Self::from_edge(manager, edge))
        })
    }

    /// Returns the number of vectors (lists) contained in `self`.
    pub fn len(&self) -> usize {
        self.manager_ref().with_manager_shared(|manager| {
            Self::len_edge(manager, manager.clone_edge(self.as_edge(manager)))
        })
    }

    /// Returns `true` if `self` is the empty set `∅`, i.e. contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.manager_ref().with_manager_shared(|manager| {
            manager
                .get_node(self.as_edge(manager))
                .is_terminal(&LDDTerminal::Empty)
        })
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
fn compute_proj<M: LDDManager>(manager: &M, proj: &[u32]) -> AllocResult<M::Edge> {
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

    singleton(manager, &result)
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

    stat!(cache_query LDDOp::Project);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Project, &[set.borrowed(), proj.borrowed()])
    {
        stat!(cache_hit LDDOp::Project);
        return Ok(res);
    }

    let proj_value = proj_node.get_value();
    let (proj_down, _proj_right) = collect_children(proj_node);
    let set_value = set_node.get_value();
    let (set_down, set_right) = collect_children(set_node);

    let result = if *proj_value == M::InnerNodeValue::false_value() {
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
    }?;

    manager
        .apply_cache()
        .add(manager, LDDOp::Project, &[set, proj], result.borrowed());

    Ok(result)
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
            // g carries the smaller value, which f lacks: keep g's value and
            // down-branch, and union f with g's remaining (right) values.
            let (g_down, g_right) = collect_children(g_node);

            let right = apply_union(manager, f.borrowed(), g_right)?;
            make_node(
                manager,
                g_node.get_value(),
                manager.clone_edge(&g_down),
                right,
            )
        }
        Ordering::Equal => {
            // Both carry the same value: union the down-branches (the
            // continuations) and the right-branches (the further values).
            let (f_down, f_right) = collect_children(f_node);
            let (g_down, g_right) = collect_children(g_node);

            let low = apply_union(manager, f_down, g_down)?;
            let high = apply_union(manager, f_right, g_right)?;
            make_node(manager, g_node.get_value(), low, high)
        }
    }?;

    manager
        .apply_cache()
        .add(manager, LDDOp::Union, &[f, g], result.borrowed());

    Ok(result)
}

/// Returns the largest subset of `a` that contains no element of `b`
/// (set difference `a \ b`).
fn apply_minus<M: LDDManager>(
    manager: &M,
    a: Borrowed<M::Edge>,
    b: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    // a \ a == ∅  and  ∅ \ b == ∅
    if a == b || manager.get_node(&a).is_terminal(&LDDTerminal::Empty) {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    // a \ ∅ == a
    if manager.get_node(&b).is_terminal(&LDDTerminal::Empty) {
        return Ok(manager.clone_edge(&a));
    }

    stat!(cache_query LDDOp::Minus);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, LDDOp::Minus, &[a.borrowed(), b.borrowed()])
    {
        stat!(cache_hit LDDOp::Minus);
        return Ok(res);
    }

    let a_node = match manager.get_node(&a) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!("a is not empty"),
    };
    let b_node = match manager.get_node(&b) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!("b is not empty"),
    };
    let a_value = a_node.get_value();
    let (a_down, a_right) = collect_children(a_node);
    let b_value = b_node.get_value();
    let (b_down, b_right) = collect_children(b_node);

    let result = match a_value.cmp(b_value) {
        Ordering::Less => {
            // b has no entry for this value — keep a's entry, advance a_right.
            let right_result =
                EdgeDropGuard::new(manager, apply_minus(manager, a_right, b.borrowed())?);
            make_node(
                manager,
                a_value,
                manager.clone_edge(&a_down),
                right_result.into_edge(),
            )?
        }
        Ordering::Equal => {
            // Subtract b_down from a_down; subtract b_right from a_right.
            let down_result =
                EdgeDropGuard::new(manager, apply_minus(manager, a_down, b_down)?);
            let right_result =
                EdgeDropGuard::new(manager, apply_minus(manager, a_right, b_right)?);
            if manager
                .get_node(&down_result)
                .is_terminal(&LDDTerminal::Empty)
            {
                right_result.into_edge()
            } else {
                make_node(
                    manager,
                    a_value,
                    down_result.into_edge(),
                    right_result.into_edge(),
                )?
            }
        }
        Ordering::Greater => {
            // a has no entry for b's value — skip b's entry.
            apply_minus(manager, a.borrowed(), b_right)?
        }
    };

    manager
        .apply_cache()
        .add(manager, LDDOp::Minus, &[a, b], result.borrowed());

    Ok(result)
}

/// Computes the set of vectors reachable in one step from `set` via the sparse
/// relation `rel`, guided by `meta` (produced by
/// [`LDDFunction::relation_product_meta`]).
///
/// Meta values at each level:
/// - 0 – position not in the relation: keep set values, advance meta only.
/// - 1 – read-only: match set and rel values; keep matched values in output.
/// - 2 – write-only: union all set values, then write each rel value.
/// - 3 – read half of a read+write pair: match values (not emitted); meta_down is 4.
/// - 4 – write half of a read+write pair: write rel values.
fn apply_relational_product<M: LDDManager>(
    manager: &M,
    set: Borrowed<M::Edge>,
    rel: Borrowed<M::Edge>,
    meta: Borrowed<M::Edge>,
) -> AllocResult<M::Edge> {
    // meta == True means all meta levels consumed; return set unchanged.
    match manager.get_node(&meta) {
        Node::Terminal(t) => {
            debug_assert_eq!(
                *t.borrow(),
                LDDTerminal::True,
                "meta should never reach the Empty terminal"
            );
            return Ok(manager.clone_edge(&set));
        }
        Node::Inner(_) => {}
    }

    // Empty set or empty relation → empty result.
    if manager
        .get_node(&set)
        .is_terminal(&LDDTerminal::Empty)
    {
        return manager.get_terminal(LDDTerminal::Empty);
    }
    if manager
        .get_node(&rel)
        .is_terminal(&LDDTerminal::Empty)
    {
        return manager.get_terminal(LDDTerminal::Empty);
    }

    stat!(cache_query LDDOp::RelationalProduct);
    if let Some(res) = manager.apply_cache().get(
        manager,
        LDDOp::RelationalProduct,
        &[set.borrowed(), rel.borrowed(), meta.borrowed()],
    ) {
        stat!(cache_hit LDDOp::RelationalProduct);
        return Ok(res);
    }

    let meta_node = match manager.get_node(&meta) {
        Node::Inner(n) => n.borrow(),
        _ => unreachable!(),
    };
    let meta_value = meta_node.get_value();
    let (meta_down, _meta_right) = collect_children(meta_node);

    let result = if *meta_value == M::InnerNodeValue::false_value() {
        // 0: not in relation — keep all set values, advance meta into next level.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);

        let right_result = EdgeDropGuard::new(
            manager,
            apply_relational_product(manager, set_right, rel.borrowed(), meta.borrowed())?,
        );
        let down_result = EdgeDropGuard::new(
            manager,
            apply_relational_product(manager, set_down, rel.borrowed(), meta_down)?,
        );

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                set_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else if *meta_value == M::InnerNodeValue::read_only_value() {
        // 1: read only — match set and rel values; keep matched values in output.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                // No rel entry for this set value; skip it.
                apply_relational_product(manager, set_right, rel.borrowed(), meta.borrowed())?
            }
            Ordering::Equal => {
                let down_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_product(manager, set_down, rel_down, meta_down)?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_product(manager, set_right, rel_right, meta.borrowed())?,
                );
                if manager
                    .get_node(&down_result)
                    .is_terminal(&LDDTerminal::Empty)
                {
                    right_result.into_edge()
                } else {
                    make_node(
                        manager,
                        set_value,
                        down_result.into_edge(),
                        right_result.into_edge(),
                    )?
                }
            }
            Ordering::Greater => {
                // No set entry for this rel value; skip the rel value.
                apply_relational_product(manager, set.borrowed(), rel_right, meta.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_only_value() {
        // 2: write only — union all set down-branches (ignoring their values),
        // then write each rel value with that combined continuation.
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        // Collect union of all down-branches at this set level.
        let combined = {
            let mut acc = manager.get_terminal(LDDTerminal::Empty)?;
            let mut cur = manager.clone_edge(&set);
            loop {
                let (down_owned, right_owned, right_empty) = {
                    let cur_node = match manager.get_node(&cur) {
                        Node::Inner(n) => n.borrow(),
                        _ => unreachable!(),
                    };
                    let (cur_down, cur_right) = collect_children(cur_node);
                    let empty = manager
                        .get_node(&cur_right)
                        .is_terminal(&LDDTerminal::Empty);
                    (
                        manager.clone_edge(&cur_down),
                        manager.clone_edge(&cur_right),
                        empty,
                    )
                };
                let new_acc = apply_union(manager, acc.borrowed(), down_owned.borrowed())?;
                // drop both after the union completes
                drop(down_owned);
                drop(acc);
                acc = new_acc;
                if right_empty {
                    drop(right_owned);
                    break;
                }
                drop(cur);
                cur = right_owned;
            }
            drop(cur);
            acc
        };

        let combined_guard = EdgeDropGuard::new(manager, combined);
        let down_result = EdgeDropGuard::new(
            manager,
            apply_relational_product(manager, combined_guard.borrowed(), rel_down, meta_down)?,
        );
        let right_result = EdgeDropGuard::new(
            manager,
            apply_relational_product(manager, set.borrowed(), rel_right, meta.borrowed())?,
        );

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                rel_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else if *meta_value == M::InnerNodeValue::read_of_pair_value() {
        // 3: read half of a read+write pair — match values (not emitted);
        // union the matched continuation with the remaining-pairs result.
        let set_node = match manager.get_node(&set) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("set must have as many levels as meta"),
        };
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let set_value = set_node.get_value();
        let (set_down, set_right) = collect_children(set_node);
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        match set_value.cmp(rel_value) {
            Ordering::Less => {
                apply_relational_product(manager, set_right, rel.borrowed(), meta.borrowed())?
            }
            Ordering::Equal => {
                // meta_down should be write_of_pair_value (4); it introduces
                // the new values.  Union with the right-siblings result so all
                // (read, write) pairs at this level are considered.
                let down_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_product(manager, set_down, rel_down, meta_down)?,
                );
                let right_result = EdgeDropGuard::new(
                    manager,
                    apply_relational_product(manager, set_right, rel_right, meta.borrowed())?,
                );
                apply_union(manager, down_result.borrowed(), right_result.borrowed())?
            }
            Ordering::Greater => {
                apply_relational_product(manager, set.borrowed(), rel_right, meta.borrowed())?
            }
        }
    } else if *meta_value == M::InnerNodeValue::write_of_pair_value() {
        // 4: write half of a read+write pair — emit rel values.
        let rel_node = match manager.get_node(&rel) {
            Node::Inner(n) => n.borrow(),
            _ => unreachable!("rel must have as many levels as meta"),
        };
        let rel_value = rel_node.get_value();
        let (rel_down, rel_right) = collect_children(rel_node);

        let down_result = EdgeDropGuard::new(
            manager,
            apply_relational_product(manager, set.borrowed(), rel_down, meta_down)?,
        );
        let right_result = EdgeDropGuard::new(
            manager,
            apply_relational_product(manager, set.borrowed(), rel_right, meta.borrowed())?,
        );

        if manager
            .get_node(&down_result)
            .is_terminal(&LDDTerminal::Empty)
        {
            right_result.into_edge()
        } else {
            make_node(
                manager,
                rel_value,
                down_result.into_edge(),
                right_result.into_edge(),
            )?
        }
    } else {
        panic!("meta has unexpected value");
    };

    manager.apply_cache().add(
        manager,
        LDDOp::RelationalProduct,
        &[set, rel, meta],
        result.borrowed(),
    );

    Ok(result)
}

/// Returns the number of vectors (lists) contained in the set `set`.
///
/// The empty set (the `Empty` terminal) contains no vectors; the set
/// containing only the empty vector (the `True` terminal) contains exactly
/// one. The counts of inner nodes are memoized in `cache` (keyed by
/// [`NodeID`]) so that shared sub-diagrams are only counted once.
fn len<M: LDDManager>(
    manager: &M,
    set: Borrowed<M::Edge>,
    cache: &mut HashMap<NodeID, usize>,
) -> usize {
    match manager.get_node(&set) {
        Node::Terminal(t) => {
            return if *t.borrow() == LDDTerminal::True { 1 } else { 0 };
        }
        Node::Inner(_) => {}
    }

    let node_id = set.node_id();
    if let Some(n) = cache.get(&node_id) {
        return *n;
    }

    // Walk the right spine, summing the sizes of all down-branches. The right
    // spine is terminated by the Empty terminal.
    let mut result = 0;
    let mut current = EdgeDropGuard::new(manager, manager.clone_edge(&set));
    loop {
        let (down, right) = {
            let node = match manager.get_node(&current) {
                Node::Inner(n) => n.borrow(),
                _ => unreachable!("the right spine ends at the Empty terminal"),
            };
            let (down, right) = collect_children(node);
            (
                EdgeDropGuard::new(manager, manager.clone_edge(&down)),
                EdgeDropGuard::new(manager, manager.clone_edge(&right)),
            )
        };
        result += len(manager, down.borrowed(), cache);
        if manager.get_node(&right).is_terminal(&LDDTerminal::Empty) {
            break;
        }
        current = right;
    }

    cache.insert(node_id, result);
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
