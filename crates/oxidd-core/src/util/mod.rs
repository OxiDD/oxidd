//! Various utilities

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasher;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ops::DerefMut;

use crate::Edge;
use crate::LevelNo;
use crate::Manager;
use crate::NodeID;

pub mod edge_hash_map;
pub use edge_hash_map::EdgeHashMap;
pub mod num;
mod substitution;
pub use substitution::*;

pub use nanorand::WyRand as Rng;

/// Borrowed version of some handle
///
/// A handle is typically just a pointer, but we cannot copy it as we need to
/// update the reference counters accordingly. However, if we want to have
/// multiple instances of the same handle without changing the reference
/// counters, we can use shared references. This works as long as one does not
/// attempt to change the handle's tag. In this case, we need ownership of the
/// handle with the different tag, but need to restrict its lifetime to the one
/// of the original handle. Furthermore, we *must not* drop the new handle. This
/// is exactly, what this data structure provides.
///
/// `Borrowed<'a, H>` always has the same representation as `H`.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Borrowed<'a, H>(ManuallyDrop<H>, PhantomData<&'a H>);

impl<'a, H> Borrowed<'a, H> {
    /// Create a new borrowed handle
    ///
    /// While the type of `handle` suggests that the handle is owned, it should
    /// be just the owned representation of a borrowed handle, e.g. no reference
    /// counters should be increased when creating `handle`.
    #[must_use]
    #[inline]
    pub fn new(handle: H) -> Self {
        Self(ManuallyDrop::new(handle), PhantomData)
    }

    /// Convert a borrowed handle into the underlying [`ManuallyDrop`] handle
    ///
    /// # Safety
    ///
    /// The caller must ensure that the resources referenced by the handle
    /// remain valid during its usage. Furthermore the returned handle must not
    /// be dropped.
    #[inline]
    pub unsafe fn into_inner(this: Self) -> ManuallyDrop<H> {
        this.0
    }
}

impl<'a, H> Deref for Borrowed<'a, H> {
    type Target = H;

    #[inline]
    fn deref(&self) -> &H {
        self.0.deref()
    }
}

impl<'a, E: Edge> Borrowed<'a, E> {
    /// Change the tag of a borrowed [`Edge`]
    ///
    /// This is equivalent to [`Edge::with_tag()`], but can be used in some
    /// situations where [`Edge::with_tag()`] can't due to lifetime
    /// restrictions.
    #[inline]
    pub fn edge_with_tag(self, tag: E::Tag) -> Self {
        // We must not drop `edge`. This would happen if `Edge::with_tag_owned`
        // panicked (which it definitely shouldn't, but …). Dropping the edge
        // itself shouldn't immediately lead to undefined behavior, but we need
        // to make sure that the owned edge is not dropped as well, so we abort
        // in this case.
        let guard = AbortOnDrop("`Edge::with_tag_owned` panicked.");
        let edge = ManuallyDrop::into_inner(self.0);
        let res = Self(ManuallyDrop::new(edge.with_tag_owned(tag)), PhantomData);
        guard.defuse();
        res
    }
}

/// Drop functionality for containers of [`Edge`]s
///
/// An edge on its own cannot be dropped: It may well be just an integer index
/// for an array. In this case we are lacking the base pointer, or more
/// abstractly the [`Manager`]. Most edge containers cannot store a manager
/// reference, be it for memory consumption or lifetime restrictions. This trait
/// provides methods to drop such a container with an externally supplied
/// function to drop edges.
pub trait DropWith<E: Edge>: Sized {
    /// Drop `self`
    ///
    /// Among dropping other parts, this calls `drop_edge` for all children.
    ///
    /// Having [`Self::drop_with_manager()`] only is not enough: To drop a
    /// [`Function`][crate::function::Function], we should not require a manager
    /// lock, otherwise we might end up in a dead-lock (if we are in a
    /// [`.with_manager_exclusive()`][crate::function::Function::with_manager_exclusive]
    /// block). So we cannot provide a `&Manager` reference. Furthermore, when
    /// all [`Function`][crate::function::Function]s and
    /// [`ManagerRef`][crate::ManagerRef]s referencing a manager are gone and
    /// the manager needs to drop e.g. the apply cache, it may also provide a
    /// function that only forgets edges rather than actually dropping them,
    /// saving the (in this case) unnecessary work of changing reference
    /// counters etc.
    fn drop_with(self, drop_edge: impl Fn(E));

    /// Drop `self`
    ///
    /// This is equivalent to `self.drop_with(|e| manager.drop_edge(e))`.
    ///
    /// Among dropping other parts, this calls
    /// [`manager.drop_edge()`][Manager::drop_edge] for all children.
    #[inline]
    fn drop_with_manager<M: Manager<Edge = E>>(self, manager: &M) {
        self.drop_with(|e| manager.drop_edge(e));
    }
}

/// A container that may hold "weak" references to nodes and needs to be
/// informed about operations potentially removing nodes
///
/// The main motivation behind this trait is the following observation: When
/// using reference counting to implement garbage collection of dead nodes,
/// cloning and dropping edges when inserting entries into the apply cache may
/// cause many CPU cache misses. To circumvent this performance issue, the apply
/// cache may store [`Borrowed<M::Edge>`]s (e.g., using the unsafe
/// [`Borrowed::into_inner()`]). Now the apply cache implementation has to
/// guarantee that every edge returned by the [`get()`][crate::ApplyCache::get]
/// method still points to a valid node. To that end, the cache may, e.g., clear
/// itself when [`Self::pre_gc()`] is called and reject any insertion of new
/// entries until [`Self::post_gc()`].
pub trait GCContainer<M: Manager> {
    /// Prepare for garbage collection
    ///
    /// The implementing container data structure may rely on that this method
    /// is called before the `Manager` performs garbage collection or any
    /// other operation that possibly removes nodes from the manager. (If
    /// this is required for SAFETY, however, creation of a `GCContainer`
    /// instance must be marked unsafe).
    ///
    /// This method may lock (parts of) `self`. Unlocking is then done in
    /// [`Self::post_gc()`].
    fn pre_gc(&self, manager: &M);

    /// Post-process a garbage collection
    ///
    /// # Safety
    ///
    /// Each call to this method must be paired with a distinct preceding
    /// [`Self::pre_gc()`] call. All operations potentially removing nodes must
    /// happen between [`Self::pre_gc()`] and the call to this method.
    unsafe fn post_gc(&self, manager: &M);
}

/// Drop guard for edges to ensure that they are not leaked
pub struct EdgeDropGuard<'a, M: Manager> {
    manager: &'a M,
    edge: ManuallyDrop<M::Edge>,
}

impl<'a, M: Manager> EdgeDropGuard<'a, M> {
    /// Create a new drop guard
    #[inline]
    pub fn new(manager: &'a M, edge: M::Edge) -> Self {
        Self {
            manager,
            edge: ManuallyDrop::new(edge),
        }
    }

    /// Convert `this` into the contained edge
    #[inline]
    pub fn into_edge(mut self) -> M::Edge {
        // SAFETY: `this.edge` is never used again, we drop `this` below
        let edge = unsafe { ManuallyDrop::take(&mut self.edge) };
        std::mem::forget(self);
        edge
    }
}

impl<'a, M: Manager> Drop for EdgeDropGuard<'a, M> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self.edge` is never used again.
        self.manager
            .drop_edge(unsafe { ManuallyDrop::take(&mut self.edge) });
    }
}

impl<'a, M: Manager> Deref for EdgeDropGuard<'a, M> {
    type Target = M::Edge;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.edge
    }
}
impl<'a, M: Manager> DerefMut for EdgeDropGuard<'a, M> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.edge
    }
}

/// Drop guard for vectors of edges to ensure that they are not leaked
pub struct EdgeVecDropGuard<'a, M: Manager> {
    manager: &'a M,
    vec: Vec<M::Edge>,
}

impl<'a, M: Manager> EdgeVecDropGuard<'a, M> {
    /// Create a new drop guard
    #[inline]
    pub fn new(manager: &'a M, vec: Vec<M::Edge>) -> Self {
        Self { manager, vec }
    }

    /// Convert `this` into the contained edge
    #[inline]
    pub fn into_vec(mut self) -> Vec<M::Edge> {
        std::mem::take(&mut self.vec)
    }
}

impl<'a, M: Manager> Drop for EdgeVecDropGuard<'a, M> {
    #[inline]
    fn drop(&mut self) {
        for e in std::mem::take(&mut self.vec) {
            self.manager.drop_edge(e);
        }
    }
}

impl<'a, M: Manager> Deref for EdgeVecDropGuard<'a, M> {
    type Target = Vec<M::Edge>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}
impl<'a, M: Manager> DerefMut for EdgeVecDropGuard<'a, M> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

/// Drop guard for inner nodes to ensure that they are not leaked
pub struct InnerNodeDropGuard<'a, M: Manager> {
    manager: &'a M,
    node: ManuallyDrop<M::InnerNode>,
}

impl<'a, M: Manager> InnerNodeDropGuard<'a, M> {
    /// Create a new drop guard
    #[inline]
    pub fn new(manager: &'a M, node: M::InnerNode) -> Self {
        Self {
            manager,
            node: ManuallyDrop::new(node),
        }
    }

    /// Convert `this` into the contained node
    #[inline]
    pub fn into_node(mut this: Self) -> M::InnerNode {
        // SAFETY: `this.edge` is never used again, we drop `this` below
        let node = unsafe { ManuallyDrop::take(&mut this.node) };
        std::mem::forget(this);
        node
    }
}

impl<'a, M: Manager> Drop for InnerNodeDropGuard<'a, M> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self.node` is never used again.
        unsafe { ManuallyDrop::take(&mut self.node) }.drop_with(|e| self.manager.drop_edge(e));
    }
}

impl<'a, M: Manager> Deref for InnerNodeDropGuard<'a, M> {
    type Target = M::InnerNode;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.node
    }
}
impl<'a, M: Manager> DerefMut for InnerNodeDropGuard<'a, M> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

/// Iterator that yields borrowed edges ([`Borrowed<'a, E>`][Borrowed]) provided
/// that `I` is an iterator that yields `&'a E`.
pub struct BorrowedEdgeIter<'a, E, I>(I, PhantomData<Borrowed<'a, E>>);

impl<'a, E: Edge, I: Iterator<Item = &'a E>> From<I> for BorrowedEdgeIter<'a, E, I> {
    fn from(it: I) -> Self {
        Self(it, PhantomData)
    }
}

impl<'a, E: Edge, I: Iterator<Item = &'a E>> Iterator for BorrowedEdgeIter<'a, E, I> {
    type Item = Borrowed<'a, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(edge) => Some(edge.borrowed()),
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, E: Edge, I: FusedIterator<Item = &'a E>> FusedIterator for BorrowedEdgeIter<'a, E, I> {}

impl<'a, E: Edge, I: ExactSizeIterator<Item = &'a E>> ExactSizeIterator
    for BorrowedEdgeIter<'a, E, I>
{
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Set of nodes
pub trait NodeSet<E>: Clone + Default + Eq {
    /// Get the number of nodes in the set
    #[must_use]
    fn len(&self) -> usize;

    /// Returns `true` iff there are no nodes in the set
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a node (the node to which edge points) to the set
    ///
    /// Returns `true` if the element was added (i.e. not previously present).
    fn insert(&mut self, edge: &E) -> bool;

    /// Return `true` if the set contains the given node
    #[must_use]
    fn contains(&self, edge: &E) -> bool;

    /// Remove a node from the set
    ///
    /// Returns `true` if the node was present in the set.
    fn remove(&mut self, edge: &E) -> bool;
}
impl<E: Edge, S: Clone + Default + BuildHasher> NodeSet<E> for HashSet<NodeID, S> {
    #[inline]
    fn len(&self) -> usize {
        HashSet::len(self)
    }
    #[inline]
    fn insert(&mut self, edge: &E) -> bool {
        self.insert(edge.node_id())
    }
    #[inline]
    fn contains(&self, edge: &E) -> bool {
        self.contains(&edge.node_id())
    }
    #[inline]
    fn remove(&mut self, edge: &E) -> bool {
        self.remove(&edge.node_id())
    }
}
impl<E: Edge> NodeSet<E> for BTreeSet<NodeID> {
    #[inline]
    fn len(&self) -> usize {
        BTreeSet::len(self)
    }
    #[inline]
    fn insert(&mut self, edge: &E) -> bool {
        self.insert(edge.node_id())
    }
    #[inline]
    fn contains(&self, edge: &E) -> bool {
        self.contains(&edge.node_id())
    }
    #[inline]
    fn remove(&mut self, edge: &E) -> bool {
        self.remove(&edge.node_id())
    }
}

/// Optional Boolean with `repr(i8)`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(i8)]
pub enum OptBool {
    /// Don't care
    None = -1,
    #[allow(missing_docs)]
    False = 0,
    #[allow(missing_docs)]
    True = 1,
}

impl From<bool> for OptBool {
    fn from(value: bool) -> Self {
        if value {
            Self::True
        } else {
            Self::False
        }
    }
}

/// Zero-sized struct that calls [`std::process::abort()`] if dropped
///
/// This is useful to make code exception safe. If there is a region that must
/// not panic for safety reasons, you can use this to prevent further unwinding,
/// at least.
///
/// Before aborting, the provided string is printed. If the guarded code in the
/// example panics, `FATAL: Foo panicked. Aborting.` will be printed to stderr.
///
/// ## Example
///
/// ```
/// # use oxidd_core::util::AbortOnDrop;
/// let panic_guard = AbortOnDrop("Foo panicked.");
/// // ... code that might panic ...
/// panic_guard.defuse();
/// ```
pub struct AbortOnDrop(pub &'static str);

impl AbortOnDrop {
    /// Consume `self` without aborting the process.
    ///
    /// Equivalent to `std::mem::forget(self)`.
    #[inline(always)]
    pub fn defuse(self) {
        std::mem::forget(self);
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        eprintln!("FATAL: {} Aborting.", self.0);
        std::process::abort();
    }
}

/// Out of memory error type
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct OutOfMemory;

/// Result type with [`OutOfMemory`] error
pub type AllocResult<T> = Result<T, OutOfMemory>;

/// Is the underlying type a floating point number?
pub trait IsFloatingPoint {
    /// `true` iff the underlying type is a floating point number
    const FLOATING_POINT: bool;

    /// One greater than the minimum possible normal power of 2 exponent, see
    /// [`f64::MIN_EXP`] for instance. `0` for integers
    const MIN_EXP: i32;
}

// dirty hack until we have specialization
/// cbindgen:ignore
impl<T: std::ops::ShlAssign<i32>> IsFloatingPoint for T {
    const FLOATING_POINT: bool = false;
    const MIN_EXP: i32 = 0;
}

/// A number type suitable for counting satisfying assignments
pub trait SatCountNumber:
    Clone
    + From<u32>
    + for<'a> std::ops::AddAssign<&'a Self>
    + for<'a> std::ops::SubAssign<&'a Self>
    + std::ops::ShlAssign<u32>
    + std::ops::ShrAssign<u32>
    + IsFloatingPoint
{
}

impl<
        T: Clone
            + From<u32>
            + for<'a> std::ops::AddAssign<&'a T>
            + for<'a> std::ops::SubAssign<&'a T>
            + std::ops::ShlAssign<u32>
            + std::ops::ShrAssign<u32>
            + IsFloatingPoint,
    > SatCountNumber for T
{
}

/// Cache for counting satisfying assignments
pub struct SatCountCache<N: SatCountNumber, S: BuildHasher> {
    /// Main map from [`NodeID`]s to their model count
    pub map: HashMap<NodeID, N, S>,

    /// Number of variables in the domain
    vars: LevelNo,

    /// Epoch to indicate if the cache is still valid.
    ///
    /// While reordering preserves semantics (and therefore also the count of
    /// satisfying assignments), nodes may be deleted and their [`NodeID`]s may
    /// get reused afterwards. The `map` should only be considered valid if
    /// `epoch` is [`Manager::reorder_count()`].
    epoch: u64,
}

impl<N: SatCountNumber, S: BuildHasher + Default> Default for SatCountCache<N, S> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            vars: 0,
            epoch: 0,
        }
    }
}

impl<N: SatCountNumber, S: BuildHasher> SatCountCache<N, S> {
    /// Create a new satisfiability counting cache
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            map: HashMap::with_hasher(hash_builder),
            vars: 0,
            epoch: 0,
        }
    }

    /// Clear the cache if it has become invalid due to reordering or a change
    /// in the number of variables
    pub fn clear_if_invalid<M: Manager>(&mut self, manager: &M, vars: LevelNo) {
        let epoch = manager.reorder_count();
        if epoch != self.epoch || vars != self.vars {
            self.epoch = epoch;
            self.vars = vars;
            self.map.clear();
        }
    }
}
