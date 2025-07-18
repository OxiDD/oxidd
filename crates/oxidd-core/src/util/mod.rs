//! Various utilities

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::BuildHasher;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;

use crate::{Edge, LevelNo, Manager, NodeID};

mod on_drop;
mod substitution;

pub mod edge_hash_map;
pub use edge_hash_map::EdgeHashMap;
pub mod num;
pub use on_drop::*;
pub use substitution::*;
pub mod var_name_map;
pub use var_name_map::VarNameMap;

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
        Some(self.0.next()?.borrowed())
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

/// Deprecated alias for [`crate::error::OutOfMemory`]
#[deprecated = "use oxidd_core::error::OutOfMemory instead"]
pub type OutOfMemory = crate::error::OutOfMemory;

/// Result type with [`OutOfMemory`][crate::error::OutOfMemory] error
pub type AllocResult<T> = Result<T, crate::error::OutOfMemory>;

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

impl<T> SatCountNumber for T where
    T: Clone
        + From<u32>
        + for<'a> std::ops::AddAssign<&'a T>
        + for<'a> std::ops::SubAssign<&'a T>
        + std::ops::ShlAssign<u32>
        + std::ops::ShrAssign<u32>
        + IsFloatingPoint
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
    /// If we cached the number of satisfying assignments of a function that has
    /// been dropped and garbage collected in the meantime, the [`NodeID`]s may
    /// have been re-used for semantically different functions. The `map` should
    /// only be considered valid if `epoch` is [`Manager::gc_count()`].
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

    /// Clear the cache if it has become invalid due to garbage collections or a
    /// change in the number of variables
    pub fn clear_if_invalid<M: Manager>(&mut self, manager: &M, vars: LevelNo) {
        let epoch = manager.gc_count();
        if epoch != self.epoch || vars != self.vars {
            self.epoch = epoch;
            self.vars = vars;
            self.map.clear();
        }
    }
}
