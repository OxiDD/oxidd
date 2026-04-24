//! Simple dummy edge implementation based on [`Arc`]
//!
//! The implementation is very limited but perfectly fine to test e.g. an apply
//! cache.

use std::collections::HashSet;
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::Arc;

use oxidd_core::error::DuplicateVarName;
use oxidd_core::util::{AllocResult, DropWith, Own, Ref};
use oxidd_core::{
    DiagramRules, Edge, HasWorkers, InnerNode, LevelNo, LevelView, Manager, Node, NodeID,
    ReducedOrNew, VarNo,
};

/// Simple dummy edge implementation based on [`Arc`]
///
/// The implementation is very limited but perfectly fine to test e.g. an apply
/// cache.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DummyEdge(*const ());

impl DummyEdge {
    /// Create a new `DummyEdge`
    pub fn new() -> Own<Self> {
        let e = DummyEdge(Arc::into_raw(Arc::new(())));
        unsafe { Own::from_raw(e) }
    }

    /// Get the node's reference count (note: `Node::ref_count()` is
    /// unimplemented)
    pub fn ref_count(this: Ref<'_, Self>) -> usize {
        let arc = ManuallyDrop::new(unsafe { Arc::from_raw(this.raw().0) });
        Arc::strong_count(&arc)
    }
}

impl Edge for DummyEdge {
    type Tag = ();

    fn with_tag(self, _tag: ()) -> Self {
        self
    }
    fn tag(self) -> Self::Tag {}

    fn node_id(self) -> NodeID {
        self.0.addr()
    }
}

/// Dummy manager that does not actually manage anything. It is only useful to
/// clone and drop edges.
pub struct DummyManager;

/// Dummy diagram rules
pub struct DummyRules;
impl DiagramRules<DummyEdge, DummyNode, ()> for DummyRules {
    type Cofactors<'a> = std::iter::Empty<Ref<'a, DummyEdge>>;

    fn reduce<M>(
        _manager: &M,
        _level: LevelNo,
        _children: impl IntoIterator<Item = Own<DummyEdge>>,
    ) -> ReducedOrNew<DummyEdge, DummyNode>
    where
        M: Manager<Edge = DummyEdge, InnerNode = DummyNode>,
    {
        ReducedOrNew::New(DummyNode, ())
    }

    fn cofactors(_tag: (), _node: &DummyNode) -> Self::Cofactors<'_> {
        std::iter::empty()
    }
}

unsafe impl Manager for DummyManager {
    type Edge = DummyEdge;
    type EdgeTag = ();
    type InnerNode = DummyNode;
    type InnerNodeValue = ();
    type Terminal = ();
    type TerminalRef<'a> = &'a ();
    type Rules = DummyRules;
    type TerminalIterator<'a>
        = std::iter::Empty<Own<DummyEdge>>
    where
        Self: 'a;
    type NodeSet = HashSet<NodeID>;
    type LevelView<'a>
        = DummyLevelView
    where
        Self: 'a;
    type LevelIterator<'a>
        = std::iter::Empty<DummyLevelView>
    where
        Self: 'a;

    fn get_node<'a>(&'a self, _edge: Ref<'a, Self::Edge>) -> Node<'a, Self> {
        Node::Inner(&DummyNode)
    }

    fn clone_edge(&self, edge: Ref<'_, Self::Edge>) -> Own<Self::Edge> {
        let raw = edge.raw();
        let _ = ManuallyDrop::new(unsafe { Arc::from_raw(raw.0) }).clone();
        unsafe { Own::from_raw(raw) }
    }

    fn drop_edge(&self, edge: Own<Self::Edge>) {
        let raw = edge.into_raw();
        drop(unsafe { Arc::from_raw(raw.0) });
    }

    fn try_remove_node(&self, edge: Own<Self::Edge>, _level: LevelNo) -> bool {
        let raw = edge.into_raw();
        Arc::into_inner(unsafe { Arc::from_raw(raw.0) }).is_some()
    }

    fn num_inner_nodes(&self) -> usize {
        0
    }

    fn num_levels(&self) -> LevelNo {
        0
    }

    fn num_named_vars(&self) -> VarNo {
        0
    }

    fn add_vars(&mut self, _additional: VarNo) -> Range<VarNo> {
        unimplemented!()
    }

    fn add_named_vars<S: Into<String>>(
        &mut self,
        _names: impl IntoIterator<Item = S>,
    ) -> Result<Range<VarNo>, DuplicateVarName> {
        unimplemented!()
    }

    fn var_name(&self, _var: VarNo) -> &str {
        panic!("out of range")
    }

    fn set_var_name(
        &mut self,
        _var: VarNo,
        _name: impl Into<String>,
    ) -> Result<(), DuplicateVarName> {
        panic!("out of range")
    }

    fn name_to_var(&self, _name: impl AsRef<str>) -> Option<VarNo> {
        None
    }

    fn var_to_level(&self, _var: VarNo) -> LevelNo {
        panic!("out of range")
    }

    fn level_to_var(&self, _level: LevelNo) -> VarNo {
        panic!("out of range")
    }

    fn level(&self, _no: LevelNo) -> Self::LevelView<'_> {
        panic!("out of range")
    }

    unsafe fn level_unchecked(&self, _no: LevelNo) -> Self::LevelView<'_> {
        panic!("out of range")
    }

    fn levels(&self) -> Self::LevelIterator<'_> {
        std::iter::empty()
    }

    fn get_terminal(&self, _terminal: Self::Terminal) -> AllocResult<Own<Self::Edge>> {
        unimplemented!()
    }

    fn num_terminals(&self) -> usize {
        0
    }

    fn terminals(&self) -> Self::TerminalIterator<'_> {
        std::iter::empty()
    }

    fn gc(&self) -> usize {
        0
    }

    fn reorder<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        f(self)
    }

    fn gc_count(&self) -> u64 {
        0
    }

    fn reorder_count(&self) -> u64 {
        0
    }
}

impl HasWorkers for DummyManager {
    type WorkerPool = crate::Workers;

    fn workers(&self) -> &Self::WorkerPool {
        &crate::Workers
    }
}

/// Dummy level view (not constructible)
pub struct DummyLevelView;

unsafe impl LevelView<DummyEdge, DummyNode> for DummyLevelView {
    type Iterator<'a>
        = std::iter::Empty<Ref<'a, DummyEdge>>
    where
        Self: 'a;

    type Taken = Self;

    fn len(&self) -> usize {
        unreachable!()
    }

    fn level_no(&self) -> LevelNo {
        unreachable!()
    }

    fn reserve(&mut self, _additional: usize) {
        unreachable!()
    }

    fn get(&self, _node: &DummyNode) -> Option<Ref<'_, DummyEdge>> {
        unreachable!()
    }

    fn insert(&mut self, _edge: Own<DummyEdge>) -> bool {
        unreachable!()
    }

    unsafe fn insert_unchecked(&mut self, _edge: Own<DummyEdge>) -> bool {
        unreachable!()
    }

    fn get_or_insert(&mut self, _node: DummyNode) -> AllocResult<Own<DummyEdge>> {
        unreachable!()
    }

    unsafe fn get_or_insert_unchecked(&mut self, _node: DummyNode) -> AllocResult<Own<DummyEdge>> {
        unreachable!()
    }

    fn gc(&mut self) {
        unreachable!()
    }

    fn try_remove(&mut self, _edge: Own<DummyEdge>) -> bool {
        unreachable!()
    }

    unsafe fn swap(&mut self, _other: &mut Self) {
        unreachable!()
    }

    fn iter(&self) -> Self::Iterator<'_> {
        unreachable!()
    }

    fn take(&mut self) -> Option<Self::Taken> {
        unreachable!()
    }
}

/// Dummy node
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct DummyNode;

impl DropWith<DummyEdge> for DummyNode {
    fn drop_with(self, _drop_edge: impl Fn(Own<DummyEdge>)) {
        unimplemented!()
    }
}

impl InnerNode<DummyEdge> for DummyNode {
    const ARITY: usize = 0;

    type Value = ();

    type ChildrenIter<'a>
        = std::iter::Empty<Ref<'a, DummyEdge>>
    where
        Self: 'a;

    fn new(
        _level: LevelNo,
        _children: impl IntoIterator<Item = Own<DummyEdge>>,
        _value: (),
    ) -> Self {
        unimplemented!()
    }

    fn check_level(&self, _check: impl FnOnce(LevelNo) -> bool) -> bool {
        true
    }
    fn assert_level_matches(&self, _level: LevelNo) {}

    fn children(&self) -> Self::ChildrenIter<'_> {
        std::iter::empty()
    }

    fn child(&self, _n: usize) -> Ref<'_, DummyEdge> {
        unimplemented!()
    }

    unsafe fn set_child(&self, _n: usize, _child: Own<DummyEdge>) -> Own<DummyEdge> {
        unimplemented!()
    }

    fn ref_count(&self) -> usize {
        unimplemented!()
    }

    fn get_value(&self) -> &() {
        &()
    }
}

/// Assert that the reference counts of edges match
///
/// # Example
///
/// ```
/// # use oxidd_core::{Edge, Manager};
/// # use oxidd_test_utils::assert_ref_counts;
/// # use oxidd_test_utils::edge::{DummyEdge, DummyManager};
/// let e1 = DummyEdge::new();
/// let e2 = DummyManager.clone_edge(e1.borrowed());
/// let e3 = DummyEdge::new();
/// assert_ref_counts!(e1, e2 = 2; e3 = 1);
/// # DummyManager.drop_edge(e1);
/// # DummyManager.drop_edge(e2);
/// # DummyManager.drop_edge(e3);
/// ```
#[macro_export]
macro_rules! assert_ref_counts {
    ($edge:ident = $count:literal) => {
        assert_eq!($crate::edge::DummyEdge::ref_count($edge.borrowed()), $count);
    };
    ($edge:ident, $($edges:ident),+ = $count:literal) => {
        assert_ref_counts!($edge = $count);
        assert_ref_counts!($($edges),+ = $count);
    };
    // spell-checker:ignore edgess
    ($($edges:ident),+ = $count:literal; $($($edgess:ident),+ = $counts:literal);+) => {
        assert_ref_counts!($($edges),+ = $count);
        assert_ref_counts!($($($edgess),+ = $counts);+);
    };
}
