//! Various utilities

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::BuildHasher;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;

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

/// Owned version of some handle
///
/// This is the owned version, [`Ref<'a, H>`] is the borrowed, and `H` itself is
/// the raw variant. See [`Ref<'a, H>`] for more details.
///
/// An owned handle cannot simply be dropped on its own but must be moved to
/// functions that handle resource deallocation properly.
///
/// `Own<H>` always has the same representation as `H` and `Ref<'a, H>`.
#[repr(transparent)]
#[must_use = "Own<H> cannot be dropped on its own"]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Own<H: Copy>(H);

impl<H: Copy> Own<H> {
    /// Create a new borrowed handle
    ///
    /// # Safety
    ///
    /// The caller must ensure that the handle provides access to the referenced
    /// resource as long as the returned value lives.
    #[inline(always)]
    pub unsafe fn from_raw(raw: H) -> Self {
        Self(raw)
    }

    /// Get the underlying raw handle
    #[inline(always)]
    pub fn raw(&self) -> H {
        self.0
    }
    /// Convert an owned handle into the underlying raw handle
    #[inline]
    pub fn into_raw(self) -> H {
        let this = ManuallyDrop::new(self);
        this.0
    }

    /// Borrow this handle
    #[inline(always)]
    pub fn borrowed(&self) -> Ref<'_, H> {
        Ref(self.0, PhantomData)
    }
}

impl<H: Copy> Drop for Own<H> {
    #[cold]
    #[inline(never)]
    fn drop(&mut self) {
        eprintln!(
            "{} must not be dropped. Backtrace:\n{}",
            std::any::type_name::<Self>(),
            std::backtrace::Backtrace::capture()
        );

        #[cfg(feature = "static_leak_check")]
        {
            extern "C" {
                #[link_name = "\n\n`Own<H>`s must not be dropped.`.\n"]
                fn trigger() -> !;
            }
            unsafe { trigger() }
        }
    }
}

impl<H: Copy + fmt::Debug> fmt::Debug for Own<H> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Borrowed version of some handle
///
/// This is the borrowed version, [`Own<H>`] is the owned, and `H` itself is the
/// raw variant. The raw variant is typically just a pointer or an integer,
/// freely copyable, and does not provide permission to access the referenced
/// value on its own. In this sense `H` is like a `*const T`, `Own<H>` like a
/// `Box<T>`, and `Ref<'a, H>` like a `&'a T`. However, a handle may be tagged,
/// which cannot really be represented in a `&T`, especially when the tag should
/// be updated on such a borrowed handle.
///
/// `Ref<'a, H>` always has the same representation as `H` and `Own<H>`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ref<'a, H: Copy>(H, PhantomData<&'a H>);

impl<'a, H: Copy> Ref<'a, H> {
    /// Create a new borrowed handle
    ///
    /// # Safety
    ///
    /// The caller must ensure that the handle provides access to the referenced
    /// resource for lifetime `'a`.
    #[must_use]
    #[inline]
    pub unsafe fn from_raw(raw: H) -> Self {
        Self(raw, PhantomData)
    }

    /// Convert a borrowed handle into the underlying raw handle
    #[inline]
    pub fn raw(self) -> H {
        self.0
    }
}

impl<'a, H: Copy> std::borrow::Borrow<Own<H>> for Ref<'a, H> {
    fn borrow(&self) -> &Own<H> {
        let ptr = self as *const Self as *const Own<H>;
        // SAFETY: `Ref<'_, H>` and `Own<H>` have the same representation as
        // `H`, we are just casting the reference. `Ref<'_, H>` provides the
        // same permissions as `&Own<H>`.
        unsafe { &*ptr }
    }
}

impl<'a, H: Copy + fmt::Debug> fmt::Debug for Ref<'a, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<E: Edge> Own<E> {
    /// Get a version of this [`Edge`] with the given tag
    #[inline]
    pub fn with_tag(self, tag: E::Tag) -> Self {
        let this = ManuallyDrop::new(self);
        Self(this.0.with_tag(tag))
    }

    /// Get the [`Tag`][crate::Tag] of this [`Edge`]
    #[inline]
    pub fn tag(&self) -> E::Tag {
        self.0.tag()
    }

    /// Returns some unique identifier for the referenced node, e.g., for I/O
    /// purposes
    #[inline]
    pub fn node_id(&self) -> usize {
        self.0.node_id()
    }
}

impl<'a, E: Edge> Ref<'a, E> {
    /// Get a version of this [`Edge`] with the given tag
    #[inline]
    pub fn with_tag(self, tag: E::Tag) -> Self {
        Self(self.0.with_tag(tag), PhantomData)
    }

    /// Get the [`Tag`][crate::Tag] of this [`Edge`]
    #[inline]
    pub fn tag(self) -> E::Tag {
        self.0.tag()
    }

    /// Returns some unique identifier for the referenced node, e.g., for I/O
    /// purposes
    #[inline]
    pub fn node_id(self) -> usize {
        self.0.node_id()
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
    fn drop_with(self, drop_edge: impl Fn(Own<E>));

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

/// Iterator that yields borrowed edges ([`Ref<'a, E>`][Ref]) provided that `I`
/// is an iterator that yields [`&'a Own<E>`][Own].
pub struct EdgeRefIter<'a, E: Edge + 'a, I>(I, PhantomData<Ref<'a, E>>);

impl<'a, E: Edge, I: Iterator<Item = &'a Own<E>>> From<I> for EdgeRefIter<'a, E, I> {
    fn from(it: I) -> Self {
        Self(it, PhantomData)
    }
}

impl<'a, E: Edge, I: Iterator<Item = &'a Own<E>>> Iterator for EdgeRefIter<'a, E, I> {
    type Item = Ref<'a, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.borrowed())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, E: Edge, I: FusedIterator<Item = &'a Own<E>>> FusedIterator for EdgeRefIter<'a, E, I> {}

impl<'a, E: Edge, I: ExactSizeIterator<Item = &'a Own<E>>> ExactSizeIterator
    for EdgeRefIter<'a, E, I>
{
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Set of nodes
pub trait NodeSet<E: Copy>: Clone + Default + Eq {
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
    fn insert(&mut self, edge: Ref<'_, E>) -> bool;

    /// Return `true` if the set contains the given node
    #[must_use]
    fn contains(&self, edge: Ref<'_, E>) -> bool;

    /// Remove a node from the set
    ///
    /// Returns `true` if the node was present in the set.
    fn remove(&mut self, edge: Ref<'_, E>) -> bool;
}
impl<E: Edge, S: Clone + Default + BuildHasher> NodeSet<E> for HashSet<NodeID, S> {
    #[inline]
    fn len(&self) -> usize {
        HashSet::len(self)
    }
    #[inline]
    fn insert(&mut self, edge: Ref<'_, E>) -> bool {
        self.insert(edge.node_id())
    }
    #[inline]
    fn contains(&self, edge: Ref<'_, E>) -> bool {
        self.contains(&edge.node_id())
    }
    #[inline]
    fn remove(&mut self, edge: Ref<'_, E>) -> bool {
        self.remove(&edge.node_id())
    }
}
impl<E: Edge> NodeSet<E> for BTreeSet<NodeID> {
    #[inline]
    fn len(&self) -> usize {
        BTreeSet::len(self)
    }
    #[inline]
    fn insert(&mut self, edge: Ref<'_, E>) -> bool {
        self.insert(edge.node_id())
    }
    #[inline]
    fn contains(&self, edge: Ref<'_, E>) -> bool {
        self.contains(&edge.node_id())
    }
    #[inline]
    fn remove(&mut self, edge: Ref<'_, E>) -> bool {
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
        if value { Self::True } else { Self::False }
    }
}

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
    + std::ops::Add<Self, Output = Self>
    + std::ops::Shl<u32, Output = Self>
    + std::ops::Shr<u32, Output = Self>
    + IsFloatingPoint
{
}

impl<T> SatCountNumber for T where
    T: Clone
        + From<u32>
        + std::ops::Add<Self, Output = Self>
        + std::ops::Shl<u32, Output = Self>
        + std::ops::Shr<u32, Output = Self>
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

    /// Whether to cache the SAT counts even for all nodes, including those that
    /// have only a single incoming edge.
    pub cache_all: bool,
}

impl<N: SatCountNumber, S: BuildHasher + Default> Default for SatCountCache<N, S> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            vars: 0,
            epoch: 0,
            cache_all: false,
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
            cache_all: false,
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
