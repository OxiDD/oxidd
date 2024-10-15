//! Simple dummy edge implementation based on [`Arc`]
//!
//! The implementation is very limited but perfectly fine to test e.g. an apply
//! cache.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use oxidd_core::util::{AllocResult, Borrowed, DropWith};
use oxidd_core::{
    DiagramRules, Edge, HasWorkers, InnerNode, LevelNo, LevelView, Manager, Node, NodeID,
    ReducedOrNew,
};

/// Simple dummy edge implementation based on [`Arc`]
///
/// The implementation is very limited but perfectly fine to test e.g. an apply
/// cache.
#[derive(Debug)]
pub struct DummyEdge(Arc<()>);

impl PartialEq for DummyEdge {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for DummyEdge {}
impl PartialOrd for DummyEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DummyEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::as_ptr(&self.0).cmp(&Arc::as_ptr(&other.0))
    }
}
impl Hash for DummyEdge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Drop for DummyEdge {
    fn drop(&mut self) {
        eprintln!(
            "Edges must not be dropped. Use Manager::drop_edge(). Backtrace:\n{}",
            std::backtrace::Backtrace::capture()
        );
    }
}

impl DummyEdge {
    /// Create a new `DummyEdge`
    pub fn new() -> Self {
        DummyEdge(Arc::new(()))
    }

    /// Get the node's reference count (note: `Node::ref_count()` is
    /// unimplemented)
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}
impl Default for DummyEdge {
    fn default() -> Self {
        Self::new()
    }
}

impl Edge for DummyEdge {
    type Tag = ();

    fn borrowed(&self) -> Borrowed<'_, Self> {
        let ptr = Arc::as_ptr(&self.0);
        Borrowed::new(DummyEdge(unsafe { Arc::from_raw(ptr) }))
    }
    fn with_tag(&self, _tag: ()) -> Borrowed<'_, Self> {
        let ptr = Arc::as_ptr(&self.0);
        Borrowed::new(DummyEdge(unsafe { Arc::from_raw(ptr) }))
    }
    fn with_tag_owned(self, _tag: ()) -> Self {
        self
    }
    fn tag(&self) -> Self::Tag {}

    fn node_id(&self) -> NodeID {
        Arc::as_ptr(&self.0) as usize
    }
}

/// Dummy manager that does not actually manage anything. It is only useful to
/// clone and drop edges.
pub struct DummyManager;

/// Dummy diagram rules
pub struct DummyRules;
impl DiagramRules<DummyEdge, DummyNode, ()> for DummyRules {
    type Cofactors<'a> = std::iter::Empty<Borrowed<'a, DummyEdge>>;

    fn reduce<M>(
        _manager: &M,
        _level: LevelNo,
        _children: impl IntoIterator<Item = DummyEdge>,
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
    type Terminal = ();
    type TerminalRef<'a> = &'a ();
    type TerminalIterator<'a>
        = std::iter::Empty<DummyEdge>
    where
        Self: 'a;
    type Rules = DummyRules;
    type NodeSet = HashSet<NodeID>;
    type LevelView<'a>
        = DummyLevelView
    where
        Self: 'a;
    type LevelIterator<'a>
        = std::iter::Empty<DummyLevelView>
    where
        Self: 'a;

    fn get_node(&self, _edge: &Self::Edge) -> Node<Self> {
        Node::Inner(&DummyNode)
    }

    fn clone_edge(&self, edge: &Self::Edge) -> Self::Edge {
        DummyEdge(edge.0.clone())
    }

    fn drop_edge(&self, edge: Self::Edge) {
        // Move the inner arc out. We need to use `std::ptr::read` since
        // `DummyEdge` implements `Drop` (to print an error).
        let inner = unsafe { std::ptr::read(&edge.0) };
        std::mem::forget(edge);
        drop(inner);
    }

    fn num_inner_nodes(&self) -> usize {
        0
    }

    fn num_levels(&self) -> LevelNo {
        0
    }

    fn add_level(
        &mut self,
        _f: impl FnOnce(LevelNo) -> Self::InnerNode,
    ) -> AllocResult<Self::Edge> {
        unimplemented!()
    }

    fn level(&self, _no: LevelNo) -> Self::LevelView<'_> {
        panic!("out of range")
    }

    fn levels(&self) -> Self::LevelIterator<'_> {
        std::iter::empty()
    }

    fn get_terminal(&self, _terminal: Self::Terminal) -> AllocResult<Self::Edge> {
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
        = std::iter::Empty<&'a DummyEdge>
    where
        Self: 'a,
        DummyEdge: 'a;

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

    fn get(&self, _node: &DummyNode) -> Option<&DummyEdge> {
        unreachable!()
    }

    fn insert(&mut self, _edge: DummyEdge) -> bool {
        unreachable!()
    }

    fn get_or_insert(&mut self, _node: DummyNode) -> AllocResult<DummyEdge> {
        unreachable!()
    }

    unsafe fn gc(&mut self) {
        unreachable!()
    }

    unsafe fn remove(&mut self, _node: &DummyNode) -> bool {
        unreachable!()
    }

    unsafe fn swap(&mut self, _other: &mut Self) {
        unreachable!()
    }

    fn iter(&self) -> Self::Iterator<'_> {
        unreachable!()
    }

    fn take(&mut self) -> Self::Taken {
        unreachable!()
    }
}

/// Dummy node
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct DummyNode;

impl DropWith<DummyEdge> for DummyNode {
    fn drop_with(self, _drop_edge: impl Fn(DummyEdge)) {
        unimplemented!()
    }
}

impl InnerNode<DummyEdge> for DummyNode {
    const ARITY: usize = 0;

    type ChildrenIter<'a>
        = std::iter::Empty<Borrowed<'a, DummyEdge>>
    where
        Self: 'a;

    fn new(_level: LevelNo, _children: impl IntoIterator<Item = DummyEdge>) -> Self {
        unimplemented!()
    }

    fn check_level(&self, _check: impl FnOnce(LevelNo) -> bool) -> bool {
        true
    }
    fn assert_level_matches(&self, _level: LevelNo) {}

    fn children(&self) -> Self::ChildrenIter<'_> {
        std::iter::empty()
    }

    fn child(&self, _n: usize) -> Borrowed<DummyEdge> {
        unimplemented!()
    }

    unsafe fn set_child(&self, _n: usize, _child: DummyEdge) -> DummyEdge {
        unimplemented!()
    }

    fn ref_count(&self) -> usize {
        unimplemented!()
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
/// let e2 = DummyManager.clone_edge(&e1);
/// let e3 = DummyEdge::new();
/// assert_ref_counts!(e1, e2 = 2; e3 = 1);
/// # DummyManager.drop_edge(e1);
/// # DummyManager.drop_edge(e2);
/// # DummyManager.drop_edge(e3);
/// ```
#[macro_export]
macro_rules! assert_ref_counts {
    ($edge:ident = $count:literal) => {
        assert_eq!($edge.ref_count(), $count);
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
