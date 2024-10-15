//! Collection of fundamental traits and types to represent decision diagrams
//!
//! # Overview
//!
//! One of the most central traits is [`Manager`]. The manager is responsible
//! for storing the nodes of a decision diagram ([`InnerNode`]s and terminal
//! nodes) and provides [`Edge`]s to identify them.
//!
//! From the user's perspective, [`Function`][function::Function] is very
//! important. A function is some kind of external reference to a node and is
//! the basis for assigning semantics to nodes and providing operations such as
//! applying connectives of boolean logic.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::double_must_use)]
// Explicitly writing out `'id` lifetimes possibly makes some code easier to
// read.
#![allow(clippy::needless_lifetimes)]
// `match` syntax may be easier to read
#![allow(clippy::manual_map)]

use std::borrow::Borrow;
use std::hash::Hash;

use util::AllocResult;
use util::Borrowed;
use util::DropWith;
use util::NodeSet;

pub mod function;
pub mod util;

/// Manager reference
///
/// The methods of this trait synchronize accesses to the manager: In a
/// concurrent setting, a manager has some kind of read/write lock, and
/// [`Self::with_manager_shared()`] / [`Self::with_manager_exclusive()`] acquire
/// this lock accordingly. In a sequential implementation, a
/// [`RefCell`][std::cell::RefCell] or the like may be used instead of lock.
pub trait ManagerRef: Clone + Eq + Hash + for<'a, 'id> From<&'a Self::Manager<'id>> {
    /// Type of the associated manager
    ///
    /// For more details on why this type is generic over `'id`, see the
    /// documentation of [`Function::Manager`][function::Function::Manager].
    type Manager<'id>: Manager;

    /// Obtain a shared manager reference
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn with_manager_shared<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&Self::Manager<'id>) -> T;

    /// Obtain an exclusive manager reference
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>) -> T;
}

/// Reduction rules for decision diagrams
///
/// This trait is intended to be implemented on a zero-sized type. Refer to
/// [`DiagramRules::reduce()`] for an example implementation.
pub trait DiagramRules<E: Edge, N: InnerNode<E>, T> {
    /// Iterator created by [`DiagramRules::cofactors()`]
    type Cofactors<'a>: Iterator<Item = Borrowed<'a, E>>
    where
        E: 'a,
        N: 'a;

    /// Apply the reduction rule(s)
    ///
    /// Besides uniqueness of nodes (there are no two nodes with the same
    /// children at the same level), decision diagrams typically impose some
    /// other reduction rules. The former is provided by the [`Manager`] /
    /// [`LevelView`]s, the latter is implemented here.
    ///
    /// The implementation is responsible for consuming the entire `children`
    /// iterator and dropping unused edges.
    ///
    /// # Example implementation
    ///
    /// In binary decision diagrams (BDDs), there are no nodes with equal
    /// children. An implementation might look like this:
    ///
    /// ```
    /// # use oxidd_core::{*, util::BorrowedEdgeIter};
    /// struct BDDRules;
    /// impl<E: Edge, N: InnerNode<E>, T> DiagramRules<E, N, T> for BDDRules {
    ///     type Cofactors<'a> = N::ChildrenIter<'a> where N: 'a, E: 'a;
    ///
    ///     fn reduce<M: Manager<Edge = E, InnerNode = N, Terminal = T>>(
    ///         manager: &M,
    ///         level: LevelNo,
    ///         children: impl IntoIterator<Item = E>,
    ///     ) -> ReducedOrNew<E, N> {
    ///         let mut it = children.into_iter();
    ///         let f0 = it.next().unwrap();
    ///         let f1 = it.next().unwrap();
    ///         debug_assert!(it.next().is_none());
    ///
    ///         if f0 == f1 {
    ///             manager.drop_edge(f1);
    ///             ReducedOrNew::Reduced(f0)
    ///         } else {
    ///             ReducedOrNew::New(InnerNode::new(level, [f0, f1]), Default::default())
    ///         }
    ///     }
    ///
    ///     fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
    ///         node.children()
    ///     }
    /// }
    /// ```
    ///
    /// Note that we assume no complemented edges, hence the cofactor iterator
    /// is just `node.children()`.
    ///
    /// The implementation assumes that there are no panics, otherwise it would
    /// leak some edges. It might be a bit better to use
    /// [`EdgeDropGuard`][util::EdgeDropGuard]s, but this would make the code
    /// more clumsy, and our assumption is usually fair enough.
    fn reduce<M: Manager<Edge = E, InnerNode = N, Terminal = T>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N>;

    /// Get the cofactors of `node` assuming an incoming edge with `tag`
    ///
    /// In some diagram types, this is the same as [`InnerNode::children()`].
    /// However, in a binary decision diagram with complement edges, we need to
    /// respect the tag of the incoming edge: If the incoming edge is
    /// complemented, then we need to complement the outgoing edges as well.
    fn cofactors(tag: E::Tag, node: &N) -> Self::Cofactors<'_>;

    /// Get the `n`-th cofactor of `node` assuming an incoming edge with `tag`
    ///
    /// This is equivalent to `Self::cofactors(tag, node).nth(n).unwrap()`.
    #[inline]
    fn cofactor(tag: E::Tag, node: &N, n: usize) -> Borrowed<E> {
        Self::cofactors(tag, node).nth(n).expect("out of range")
    }
}

/// Result of the attempt to create a new node
///
/// Before actually creating a new node, reduction rules should be applied
/// (see [`DiagramRules::reduce()`]). If a reduction was applied, then
/// [`DiagramRules::reduce()`] returns the `Reduced` variant, otherwise the
/// `New` variant.
pub enum ReducedOrNew<E: Edge, N: InnerNode<E>> {
    /// A reduction rule was applied
    Reduced(E),
    /// The node is new. After inserting it into the manager, the edge should be
    /// tagged with the given tag.
    New(N, E::Tag),
}

impl<E: Edge, N: InnerNode<E>> ReducedOrNew<E, N> {
    /// Insert `self` into `manager` and `unique_table` at the given `level` if
    /// it is `New`, otherwise return the `Reduced` edge.
    ///
    /// `level` must agree with the level used for creating the node, and must
    /// be strictly above (i.e. less than) the children's levels.
    #[must_use]
    #[inline(always)]
    pub fn then_insert<M>(self, manager: &M, level: LevelNo) -> AllocResult<E>
    where
        M: Manager<InnerNode = N, Edge = E>,
    {
        match self {
            ReducedOrNew::Reduced(e) => Ok(e),
            ReducedOrNew::New(node, tag) => {
                debug_assert_ne!(level, LevelNo::MAX);
                debug_assert!(node.check_level(|l| l == level));
                debug_assert!(node.children().all(|c| {
                    if let Node::Inner(node) = manager.get_node(&*c) {
                        node.check_level(|l| level < l)
                    } else {
                        true
                    }
                }));

                let edge = manager.level(level).get_or_insert(node)?;
                Ok(edge.with_tag_owned(tag))
            }
        }
    }
}

/// Node in a decision diagram
///
/// [`Eq`] and [`Hash`] should consider the children only, in particular no
/// level information. This means that if `Self` implements [`HasLevel`],
/// [`HasLevel::set_level()`] may be called while the node is present in a
/// [`LevelView`]. This is not the case for [`InnerNode::set_child()`]: The user
/// must remove the node from the [`LevelView`] before setting the children (and
/// re-insert it afterwards).
#[must_use]
pub trait InnerNode<E: Edge>: Sized + Eq + Hash + DropWith<E> {
    /// The node's arity (upper bound)
    const ARITY: usize;

    /// Iterator over children of an inner node
    type ChildrenIter<'a>: ExactSizeIterator<Item = Borrowed<'a, E>>
    where
        Self: 'a,
        E: 'a;

    /// Create a new node
    ///
    /// Note that this does not apply any reduction rules. A node type that does
    /// not store levels internally (does not implement [`HasLevel`]) may simply
    /// ignore the `level` parameter.
    ///
    /// Panics if `children`'s length does not match the node's requirements
    /// (typically, the length should be [`Self::ARITY`], but some node types
    /// may deviate from that).
    #[must_use]
    fn new(level: LevelNo, children: impl IntoIterator<Item = E>) -> Self;

    /// Returns the result of `check` applied to the node's level in case this
    /// node type stores levels, otherwise returns `true`.
    ///
    /// Use [`HasLevel::level()`] if you require your nodes to store the level
    /// number and want to get the level number.
    fn check_level(&self, check: impl FnOnce(LevelNo) -> bool) -> bool;

    /// Panics if the node types stores a level and the node's level is not
    /// `level`
    fn assert_level_matches(&self, level: LevelNo);

    /// Get the children of this node as an iterator
    #[must_use]
    fn children(&self) -> Self::ChildrenIter<'_>;

    /// Get the `n`-th child of this node
    fn child(&self, n: usize) -> Borrowed<E>;

    /// Set the `n`-th child of this node
    ///
    /// Returns the previous `n`-th child.
    ///
    /// Note that this function may also change the node's hash value etc., so
    /// in case the node is stored in a hash table ([`LevelView`]), it needs to
    /// be removed before calling this method.
    ///
    /// Panics if the node does not have an `n`-th child.
    ///
    /// # Safety
    ///
    /// The caller must have exclusive access to the node. In the first place,
    /// this is granted by acquiring an exclusive manager lock
    /// ([`Function::with_manager_exclusive()`][function::Function::with_manager_exclusive] or
    /// [`ManagerRef::with_manager_exclusive()`]). However, exclusive access to
    /// some nodes may be delegated to other threads (which is the reason why we
    /// only require a shared and not a mutable manager reference). Furthermore,
    /// there must not be a borrowed child (obtained via
    /// [`InnerNode::children()`]).
    #[must_use = "call `Manager::drop_edge()` if you don't need the previous edge"]
    unsafe fn set_child(&self, n: usize, child: E) -> E;

    /// Get the node's reference count
    ///
    /// This ignores all internal references used by the manager
    /// implementation.
    #[must_use]
    fn ref_count(&self) -> usize;
}

/// Level number type
///
/// Levels with lower numbers are located at the top of the diagram, higher
/// numbers at the bottom. [`LevelNo::MAX`] is reserved for terminal nodes.
/// Adjacent levels have adjacent level numbers.
pub type LevelNo = u32;

/// Atomic version of [`LevelNo`]
pub type AtomicLevelNo = std::sync::atomic::AtomicU32;

/// Trait for nodes that have a level
///
/// Quasi-reduced BDDs, for instance, do not need the level information stored
/// in their nodes, so there is no need to implement this trait.
///
/// Implementors should also implement [`InnerNode`]. If `Self` is [`Sync`],
/// then the level number should be implemented using [`AtomicLevelNo`]. In
/// particular, concurrent calls to [`Self::level()`] and [`Self::set_level()`]
/// must not lead to data races.
///
/// # Safety
///
/// 1. A node in a [`LevelView`] with level number L has level number L (i.e.
///    `self.level()` returns L).
/// 2. [`InnerNode::check_level()`] with a check `c` must return
///    `c(self.level())`. Similarly, [`InnerNode::assert_level_matches()`] must
///    panic if the level does not match.
///
/// These conditions are crucial to enable concurrent level swaps as part of
/// reordering (see the `oxidd-reorder` crate): The algorithm iterates over the
/// nodes at the upper level and needs to know whether a node is part of the
/// level directly below it. The procedure only has access to nodes at these two
/// levels, hence it must rely on the level information for SAFETY.
///
/// Note that invariant 1 may be broken by [`HasLevel::set_level()`] and
/// [`LevelView::swap()`]; the caller of these functions is responsible to
/// re-establish the invariant.
pub unsafe trait HasLevel {
    /// Get the node's level
    #[must_use]
    fn level(&self) -> LevelNo;

    /// Set the node's level
    ///
    /// # Safety
    ///
    /// This method may break SAFETY invariant 1 of the [`HasLevel`] trait: A
    /// node in a [`LevelView`] with level number L has level number L (i.e.
    /// `self.level()` returns L). The caller is responsible to re-establish the
    /// invariant. (Make sure that the calling code is exception-safe!)
    unsafe fn set_level(&self, level: LevelNo);
}

/// Node identifier returned by [`Edge::node_id()`]
///
/// The most significant bit is reserved, i.e. `NodeID`s must (normally) be less
/// than `1 << (NodeID::BITS - 1)`
pub type NodeID = usize;

/// Edge in a decision diagram
///
/// Generally speaking, an edge is the directed connection between two nodes,
/// with some kind of annotation. In a binary decision diagram with complement
/// edges, there are the "true" edges as well as the normal and the complemented
/// "false" edges. When considering a single edge, it usually is not so
/// important whether this edge is a "true" or "false" edge; we can simply have
/// distinguishable "slots" in the source node. In contrast, whether an edge is
/// complemented or not has a much greater influence on the meaning of an edge.
///
/// In a decision diagram, obtaining the source of an edge is usually not so
/// important, hence this trait does not provide such functionality. This means
/// that an edge can (more or less) be implemented as a (tagged) pointer to the
/// target node.
///
/// This trait requires implementors to also implement [`Ord`]. Edges should be
/// considered equal if and only if they point to the same node with the same
/// tag. Besides that, there are no further restrictions. The order implemented
/// for [`Ord`] can be an arbitrary, fixed order (e.g. using addresses of the
/// nodes). The main idea of this is to give the set `{f, g}` of two edges `f`
/// and `g` a unique tuple/array representation.
#[must_use]
pub trait Edge: Sized + Ord + Hash {
    /// Edge tag
    ///
    /// For instance, an edge tag can be used to mark an edge as complemented.
    ///
    /// If the decision diagram does not need any special edge tags, this can
    /// simply be `()`.
    type Tag: Tag;

    /// Turn a reference into a borrowed handle
    fn borrowed(&self) -> Borrowed<Self>;
    /// Get a version of this [`Edge`] with the given tag
    ///
    /// Refer to [`Borrowed::edge_with_tag()`] for cases in which this method
    /// cannot be used due to lifetime restrictions.
    fn with_tag(&self, tag: Self::Tag) -> Borrowed<Self>;
    /// Get a version of this [`Edge`] with the given tag
    fn with_tag_owned(self, tag: Self::Tag) -> Self;

    /// Get the [`Tag`] of this [`Edge`]
    fn tag(&self) -> Self::Tag;

    /// Returns some unique identifier for the node, e.g. for I/O purposes
    fn node_id(&self) -> NodeID;
}

/// Trait for tags that can be attached to pointers (e.g. edges, see
/// [`Edge::Tag`])
///
/// This trait is automatically implemented for types that implement [`Eq`],
/// [`Default`], and [`Countable`].
pub trait Tag: Sized + Copy + Eq + Default + Countable {}
impl<T: Eq + Default + Countable> Tag for T {}

/// Types whose values can be counted, i.e. there is a bijection between the
/// values of the type and the range `0..=MAX_VALUE`.
///
/// This is mainly intended to be implemented on `enum`s. In most cases, you can
/// simply derive it.
///
/// # Safety
///
/// [`Countable::as_usize()`] and [`Countable::from_usize()`] must form a
/// bijection between the values of type `Self` and `0..=MAX_VALUE`, more
/// formally: For all `t: Self` it must hold that
/// `Self::from_usize(t.as_usize()) == t`.
/// For all `u: usize` such that `t.as_usize() == u` for some `t: Self`,
/// `Self::from_usize(u).as_usize() == u` must be true. Furthermore,
/// `t.as_usize() <= Self::MAX_VALUE` must hold.
///
/// This trait is marked unsafe because violating any invariant of the above may
/// e.g. result in out-of-bounds accesses.
pub unsafe trait Countable: Sized + Copy {
    /// Maximum value returned by `self.as_usize()`.
    const MAX_VALUE: usize;

    /// Convert `self` into a `usize`.
    #[must_use]
    fn as_usize(self) -> usize;
    /// Convert the given `value` into an instance of `Self`.
    ///
    /// May panic if an invalid value is passed, or return some default value.
    #[must_use]
    fn from_usize(value: usize) -> Self;
}

// SAFETY: There is a bijection for all values of type `()` and `0..=MAX_VALUE`.
unsafe impl Countable for () {
    const MAX_VALUE: usize = 0;

    #[inline]
    fn as_usize(self) -> usize {
        0
    }

    #[inline]
    fn from_usize(_value: usize) -> Self {}
}

// SAFETY: There is a bijection for all values of type `bool` and `0..=1`.
unsafe impl Countable for bool {
    const MAX_VALUE: usize = 1;

    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn from_usize(value: usize) -> Self {
        value != 0
    }
}

/// Either an inner or a terminal node
pub enum Node<'a, M: Manager + 'a> {
    #[allow(missing_docs)]
    Inner(&'a M::InnerNode),
    #[allow(missing_docs)]
    Terminal(M::TerminalRef<'a>),
}

impl<'a, M: Manager + 'a> Clone for Node<'a, M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, M: Manager + 'a> Copy for Node<'a, M> {}

impl<'a, M: Manager> Node<'a, M> {
    /// Get the reference count of the underlying node
    #[must_use]
    #[inline]
    pub fn ref_count(self) -> usize {
        match self {
            Node::Inner(node) => node.ref_count(),
            Node::Terminal(_) => usize::MAX, /* TODO: should we support concrete reference counts
                                              * for terminals? */
        }
    }
}

impl<'a, M: Manager> Node<'a, M>
where
    M::InnerNode: HasLevel,
{
    /// Get the level of the underlying node (`LevelNo::MAX` for terminals)
    #[must_use]
    #[inline]
    pub fn level(self) -> LevelNo {
        match self {
            Node::Inner(node) => node.level(),
            Node::Terminal(_) => LevelNo::MAX,
        }
    }
}

impl<'a, M: Manager> Node<'a, M> {
    /// Unwrap the inner node
    ///
    /// Panics if `self` is a terminal.
    #[must_use]
    #[track_caller]
    #[inline]
    pub fn unwrap_inner(self) -> &'a M::InnerNode {
        match self {
            Node::Inner(node) => node,
            Node::Terminal(_) => panic!("expected an inner node, but this is a terminal"),
        }
    }

    /// Unwrap the inner node
    ///
    /// Panics with `msg` if `self` is a terminal.
    #[must_use]
    #[track_caller]
    #[inline]
    pub fn expect_inner(self, msg: &str) -> &'a M::InnerNode {
        match self {
            Node::Inner(node) => node,
            Node::Terminal(_) => panic!("{}", msg),
        }
    }

    /// Returns `true` if this is an inner node
    #[inline]
    pub fn is_inner(self) -> bool {
        match self {
            Node::Inner(_) => true,
            Node::Terminal(_) => false,
        }
    }

    /// Unwrap the terminal
    ///
    /// Panics if `self` is an inner node
    #[must_use]
    #[track_caller]
    #[inline]
    pub fn unwrap_terminal(&self) -> &M::Terminal {
        match self {
            Node::Inner(_) => panic!("expected a terminal node, but this is an inner node"),
            Node::Terminal(ref t) => t.borrow(),
        }
    }

    /// Unwrap the terminal
    ///
    /// Panics with `msg` if `self` is an inner node
    #[must_use]
    #[track_caller]
    #[inline]
    pub fn expect_terminal(&self, msg: &str) -> &M::Terminal {
        match self {
            Node::Inner(_) => panic!("{}", msg),
            Node::Terminal(ref t) => t.borrow(),
        }
    }

    /// Returns `true` if this is any terminal node
    #[inline]
    pub fn is_any_terminal(self) -> bool {
        match self {
            Node::Inner(_) => false,
            Node::Terminal(_) => true,
        }
    }

    /// Returns `true` if this is the given `terminal`
    #[inline]
    pub fn is_terminal(self, terminal: &M::Terminal) -> bool {
        match self {
            Node::Inner(_) => false,
            Node::Terminal(t) => t.borrow() == terminal,
        }
    }
}

/// Manager for nodes in a decision diagram
///
/// In the basic formulation, a decision diagram is a directed acyclic graph
/// where all inner nodes are associated with a level, and each level in turn is
/// associated with a variable. A decision diagram can represent functions
/// Dⁿ → T, where n is the number of variables. Every inner node has |D|
/// outgoing edges, pointing to child nodes. The semantics of an inner node
/// depends on the value of its associated variable. If the variable has
/// value d ∈ D, then the node's semantics is the semantics of the child
/// referenced by the edge corresponding to d. Every terminal node is associated
/// with a value t ∈ T, and the node's semantics is just that value t. Some
/// kinds of decision diagrams deviate from this formulation in one way or the
/// other, but still fit into the framework.
///
/// The manager's responsibility is to store nodes and provide edges to identify
/// them. Internally, it has a unique table to ensure uniqueness of nodes
/// (typically, there should be no two nodes with the same children at the same
/// level). The manager supports some kind of garbage collection: If there are
/// no more edges pointing to a node, the node does not necessarily have to be
/// deleted immediately. It may well be that the node revives. But from time to
/// time it may make sense to remove all currently dead nodes. The manager also
/// supports a levelized view on the diagram (via [`LevelView`]s).
///
/// Note that we are way more concerned about levels than about variables here.
/// This is because variables can easily be handled externally. For many kinds
/// of diagrams, we can just use the function representation of a variable and
/// get the mapping between variables and levels "for free". This is especially
/// nice as there are reordering operations which change the mapping between
/// levels and variables but implicitly update the function representations
/// accordingly.
///
/// # Safety
///
/// The implementation must ensure that every inner node only refers to nodes
/// stored in this manager.
///
/// Every level view is associated with a manager and a level number.
/// [`Manager::level()`] must always return the level view associated to this
/// manager with the given level number.
///
/// If [`Manager::InnerNode`] implements [`HasLevel`], then the implementation
/// must ensure that [`HasLevel::level()`] returns level number L for all nodes
/// at the level view for L. Specifically this means that
/// [`Manager::add_level()`] must check the newly created node. The invariant
/// may only be broken by unsafe code (e.g. via [`HasLevel::set_level()`] and
/// [`LevelView::swap()`]) and must be re-established when leaving the unsafe
/// scope (be aware of panics!).
pub unsafe trait Manager: Sized {
    /// Type of edge
    type Edge: Edge<Tag = Self::EdgeTag>;
    /// Type of edge tags
    type EdgeTag: Tag;
    /// Type of inner nodes
    type InnerNode: InnerNode<Self::Edge>;
    /// Type of terminals
    type Terminal: Eq + Hash;
    /// References to [`Self::Terminal`]s
    ///
    /// Should either be a `&'a Self::Terminal` or just `Self::Terminal`. The
    /// latter is useful for "static terminal managers" which don't actually
    /// store terminal nodes but can map between edges and terminal nodes on the
    /// fly. In this case, it would be hard to hand out node references.
    type TerminalRef<'a>: Borrow<Self::Terminal> + Copy
    where
        Self: 'a;
    /// Diagram rules, see [`DiagramRules`] for more details
    type Rules: DiagramRules<Self::Edge, Self::InnerNode, Self::Terminal>;

    /// Iterator over all terminals
    ///
    /// The actual items are edges pointing to terminals since this allows us to
    /// get a [`NodeID`].
    type TerminalIterator<'a>: Iterator<Item = Self::Edge>
    where
        Self: 'a;

    /// Node set type, possibly more efficient than a `HashSet<NodeID>`
    type NodeSet: NodeSet<Self::Edge>;

    /// A view on a single level of the unique table.
    type LevelView<'a>: LevelView<Self::Edge, Self::InnerNode>
    where
        Self: 'a;

    /// Iterator over levels
    type LevelIterator<'a>: DoubleEndedIterator<Item = Self::LevelView<'a>> + ExactSizeIterator
    where
        Self: 'a;

    /// Get a reference to the node to which `edge` points
    #[must_use]
    fn get_node(&self, edge: &Self::Edge) -> Node<Self>;

    /// Clone `edge`
    #[must_use]
    fn clone_edge(&self, edge: &Self::Edge) -> Self::Edge;

    /// Drop `edge`
    fn drop_edge(&self, edge: Self::Edge);

    /// Get the count of inner nodes
    #[must_use]
    fn num_inner_nodes(&self) -> usize;

    /// Get an approximate count of inner nodes
    ///
    /// For concurrent implementations, it may be much less costly to determine
    /// an approximation of the inner node count than an accurate count
    /// ([`Self::num_inner_nodes()`]).
    #[must_use]
    fn approx_num_inner_nodes(&self) -> usize {
        self.num_inner_nodes()
    }

    /// Get the number of levels
    #[must_use]
    fn num_levels(&self) -> LevelNo;

    /// Add a level with the given node to the unique table.
    ///
    /// To avoid unnecessary (un-)locking, this function takes a closure `f`
    /// that creates a first node for the new level.
    ///
    /// Returns an edge for the newly created node.
    ///
    /// Panics if the new node's level does not match the provided level.
    #[must_use]
    fn add_level(&mut self, f: impl FnOnce(LevelNo) -> Self::InnerNode) -> AllocResult<Self::Edge>;

    /// Get the level given by `no`
    ///
    /// Implementations may or may not acquire a lock here.
    ///
    /// Panics if `no >= self.num_levels()`.
    #[must_use]
    fn level(&self, no: LevelNo) -> Self::LevelView<'_>;

    /// Iterate over the levels from top to bottom
    #[must_use]
    fn levels(&self) -> Self::LevelIterator<'_>;

    /// Get an edge for the given terminal
    ///
    /// Locking behavior: May acquire a lock for the internal terminal unique
    /// table. In particular, this means that calling this function while
    /// holding a terminal iterator ([`Manager::terminals()`]) may cause a
    /// deadlock.
    #[must_use]
    fn get_terminal(&self, terminal: Self::Terminal) -> AllocResult<Self::Edge>;

    /// Get the number of terminals
    ///
    /// Should agree with the length of the iterator returned by
    /// [`Manager::terminals()`].
    ///
    /// Locking behavior: May acquire a lock for the internal terminal unique
    /// table. In particular, this means that calling this function while
    /// holding a terminal iterator ([`Manager::terminals()`]) may cause a
    /// deadlock.
    #[must_use]
    fn num_terminals(&self) -> usize;

    /// Iterator over all terminals
    ///
    /// Locking behavior: May acquire a lock for the internal terminal unique
    /// table.
    #[must_use]
    fn terminals(&self) -> Self::TerminalIterator<'_>;

    /// Perform garbage collection
    ///
    /// This method looks for nodes that are neither referenced by a
    /// [`Function`][function::Function] nor another node and removes them. The
    /// method works from top to bottom, so if a node is only referenced by
    /// nodes that can be removed, this node will be removed as well.
    ///
    /// Returns the number of nodes removed.
    fn gc(&self) -> usize;

    /// Prepare and postprocess a reordering operation. The reordering itself is
    /// performed in `f`.
    ///
    /// Returns the value returned by `f`.
    fn reorder<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T;

    /// Get the count of reordering operations
    ///
    /// This counter should monotonically increase to ensure that caches are
    /// invalidated accordingly.
    fn reorder_count(&self) -> u64;
}

/// View of a single level in the manager
///
/// # Safety
///
/// Each level view is associated with a [`Manager`] M and a [`LevelNo`] L. The
/// level view must ensure that all contained nodes and their descendants are
/// stored in M. The edges returned by [`LevelView::get()`] and
/// [`LevelView::get_or_insert()`] must reference nodes at this level.
/// [`LevelView::iter()`] must yield all edges to nodes at this level (it must
/// not hide some away).
///
/// [`LevelView::swap()`] conceptually removes all nodes from this level and
/// inserts them at the other level and vice versa.
pub unsafe trait LevelView<E: Edge, N: InnerNode<E>> {
    /// Iterator over [`Edge`]s pointing to nodes at this level
    type Iterator<'a>: Iterator<Item = &'a E>
    where
        Self: 'a,
        E: 'a;

    /// Taken level view, see [`LevelView::take()`]
    type Taken: LevelView<E, N>;

    /// Get the number of nodes on this level
    #[must_use]
    fn len(&self) -> usize;

    /// Returns `true` iff this level contains nodes
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the level number of this level
    #[must_use]
    fn level_no(&self) -> LevelNo;

    /// Reserve space for `additional` nodes on this level
    fn reserve(&mut self, additional: usize);

    /// Get the edge corresponding to the given node (if present)
    #[must_use]
    fn get(&self, node: &N) -> Option<&E>;

    /// Insert the given edge into the unique table at this level, assuming that
    /// the referenced node is already stored in the associated manager.
    ///
    /// Returns `true` if the edge was inserted, `false` if it was already
    /// present.
    ///
    /// Panics if `edge`
    /// - points to a terminal node,
    /// - references a node from a different manager, or
    /// - `N` implements [`HasLevel`] and `edge` references a node for which
    ///   [`HasLevel::level()`] returns a different level.
    ///
    /// Furthermore, this function should panic if `edge` is tagged, but the
    /// caller must not rely on that. An implementation may simply remove the
    /// tag for optimization purposes.
    fn insert(&mut self, edge: E) -> bool;

    /// Get the edge corresponding to `level` and `node` if present, or insert
    /// it.
    ///
    /// Panics if
    /// - the children of `node` are stored in a different manager, or
    /// - `N` implements [`HasLevel`] and [`HasLevel::level(node)`] returns a
    ///   different level.
    #[must_use]
    fn get_or_insert(&mut self, node: N) -> AllocResult<E>;

    /// Perform garbage collection on this level
    ///
    /// # Safety
    ///
    /// Must be called from inside the closure passed to [`Manager::reorder()`].
    unsafe fn gc(&mut self);

    /// Remove `node` from (this level of) the manager
    ///
    /// Returns whether the value was present at this level.
    ///
    /// # Safety
    ///
    /// Must be called from inside the closure passed to [`Manager::reorder()`].
    unsafe fn remove(&mut self, node: &N) -> bool;

    /// Move all nodes from this level to the other level and vice versa.
    ///
    /// # Safety
    ///
    /// This method does not necessarily change the level returned by
    /// [`HasLevel::level()`] for the nodes at this or the `other` level. The
    /// caller must ensure a consistent state using calls to
    /// [`HasLevel::set_level()`]. (Be aware of exception safety!)
    unsafe fn swap(&mut self, other: &mut Self);

    /// Iterate over all the edges at this level
    #[must_use]
    fn iter(&self) -> Self::Iterator<'_>;

    /// Clear this level, returning a level view containing all the previous
    /// edges.
    #[must_use]
    fn take(&mut self) -> Self::Taken;
}

/// Cache for the result of apply operations
///
/// This trait provides methods to add computation results to the apply cache
/// and to query the cache for these results. Just as every cache, the
/// implementation may decide to evict results from the cache.
pub trait ApplyCache<M: Manager, O: Copy>: DropWith<M::Edge> {
    /// Get the result of `operation`, if cached
    #[must_use]
    fn get_with_numeric(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        numeric_operands: &[u32],
    ) -> Option<M::Edge>;

    /// Add the result of `operation` to this cache
    ///
    /// An implementation is free to not cache any result. (This is why we use
    /// `Borrowed<M::Edge>`, which in this case elides a few clone and drop
    /// operations.) If the cache already contains a key equal to `operation`,
    /// there is no need to update its value. (Again, we can elide clone and
    /// drop operations.)
    fn add_with_numeric(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        numeric_operands: &[u32],
        value: Borrowed<M::Edge>,
    );

    /// Shorthand for [`Self::get_with_numeric()`] without numeric operands
    #[inline(always)]
    #[must_use]
    fn get(&self, manager: &M, operator: O, operands: &[Borrowed<M::Edge>]) -> Option<M::Edge> {
        self.get_with_numeric(manager, operator, operands, &[])
    }

    /// Shorthand for [`Self::add_with_numeric()`] without numeric operands
    #[inline(always)]
    fn add(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        value: Borrowed<M::Edge>,
    ) {
        self.add_with_numeric(manager, operator, operands, &[], value)
    }

    /// Remove all entries from the cache
    fn clear(&self, manager: &M);
}

/// Apply cache container
///
/// Intended to be implemented by [`Manager`]s such that generic implementations
/// of the recursive apply algorithm can simply require
/// `M: Manager + HasApplyCache<M, O>`, where `O` is the operator type.
pub trait HasApplyCache<M: Manager, O: Copy> {
    /// The concrete apply cache type
    type ApplyCache: ApplyCache<M, O>;

    /// Get a shared reference to the contained apply cache
    #[must_use]
    fn apply_cache(&self) -> &Self::ApplyCache;

    /// Get a mutable reference to the contained apply cache
    #[must_use]
    fn apply_cache_mut(&mut self) -> &mut Self::ApplyCache;
}

/// Worker thread pool associated with a [`Manager`]
///
/// A manager having its own thread pool has the advantage that it may use
/// thread-local storage for its workers to pre-allocate some resources (e.g.,
/// slots for nodes) and thereby reduce lock contention.
pub trait WorkerPool: Sync {
    /// Get the current number of threads
    fn current_num_threads(&self) -> usize;

    /// Get the recursion depth up to which operations are split
    fn split_depth(&self) -> u32;

    /// Set the recursion depth up to which operations are split
    ///
    /// `None` means that the implementation should automatically choose the
    /// depth. `Some(0)` means that no operations are split.
    fn set_split_depth(&self, depth: Option<u32>);

    /// Execute `op` within the thread pool
    ///
    /// If this method is called from another thread pool, it may cooperatively
    /// yield execution to that pool until `op` has finished.
    fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R;

    /// Execute `op_a` and `op_b` in parallel within the thread pool
    ///
    /// Note that the split depth has no influence on this method. Checking
    /// whether to split an operation must be done externally.
    fn join<RA: Send, RB: Send>(
        &self,
        op_a: impl FnOnce() -> RA + Send,
        op_b: impl FnOnce() -> RB + Send,
    ) -> (RA, RB);

    /// Execute `op` on every worker in the thread pool
    fn broadcast<R: Send>(&self, op: impl Fn(BroadcastContext) -> R + Sync) -> Vec<R>;
}

/// Context provided to workers by [`WorkerPool::broadcast()`]
#[derive(Clone, Copy, Debug)]
pub struct BroadcastContext {
    /// Index of this worker (in range `0..num_threads`)
    pub index: u32,

    /// Number of threads receiving the broadcast
    pub num_threads: u32,
}

/// Helper trait to be implemented by [`Manager`] and [`ManagerRef`] if they
/// feature a [`WorkerPool`].
pub trait HasWorkers: Sync {
    /// Type of the worker pool
    type WorkerPool: WorkerPool;

    /// Get the worker pool
    fn workers(&self) -> &Self::WorkerPool;
}
