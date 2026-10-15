use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};

use crate::Manager;

use super::{DropWith, Own};

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
#[derive(Debug)]
pub struct AbortOnDrop<'a>(pub &'a str);

impl AbortOnDrop<'_> {
    /// Consume `self` without aborting the process.
    ///
    /// Equivalent to `std::mem::forget(self)`.
    #[inline(always)]
    pub fn defuse(self) {
        std::mem::forget(self);
    }
}

impl Drop for AbortOnDrop<'_> {
    fn drop(&mut self) {
        eprintln!("FATAL: {} Aborting.", self.0);
        std::process::abort();
    }
}

/// Drop guard for owned edges to ensure that they are not leaked
pub struct EdgeDropGuard<'a, M: Manager> {
    /// Manager containing the edge
    pub manager: &'a M,
    edge: ManuallyDrop<Own<M::Edge>>,
}

impl<'a, M: Manager> EdgeDropGuard<'a, M> {
    /// Create a new drop guard
    #[inline]
    pub fn new(manager: &'a M, edge: Own<M::Edge>) -> Self {
        Self {
            manager,
            edge: ManuallyDrop::new(edge),
        }
    }

    /// Convert `self` into the contained edge
    #[inline]
    pub fn into_edge(self) -> Own<M::Edge> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this.edge` is never used again, we forget `this`
        unsafe { ManuallyDrop::take(&mut this.edge) }
    }

    /// Convert this drop guard into one where the edge is tagged with `tag`
    #[inline]
    pub fn with_tag(self, tag: M::EdgeTag) -> Self {
        Self::new(self.manager, self.into_edge().with_tag(tag))
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
    type Target = Own<M::Edge>;

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
    /// Manager containing the edges
    pub manager: &'a M,
    vec: Vec<Own<M::Edge>>,
}

impl<'a, M: Manager> EdgeVecDropGuard<'a, M> {
    /// Create a new drop guard
    #[inline]
    pub fn new(manager: &'a M, vec: Vec<Own<M::Edge>>) -> Self {
        Self { manager, vec }
    }

    /// Convert `this` into the contained edge
    #[inline]
    pub fn into_vec(mut self) -> Vec<Own<M::Edge>> {
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
    type Target = Vec<Own<M::Edge>>;

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
    /// Manager that is used to drop the node in the drop handler
    pub manager: &'a M,
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

    /// Convert `self` into the contained node
    #[inline]
    pub fn into_node(self) -> M::InnerNode {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this.node` is never used again, we forget `this`
        unsafe { ManuallyDrop::take(&mut this.node) }
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

/// Collect exactly `N` elements from an iterator
///
/// Panics if the iterator does not provide exactly `N` elements. Calls the
/// custom `drop` function for all elements on error.
#[inline]
pub fn collect_exact<T, const N: usize>(
    it: impl IntoIterator<Item = T>,
    drop: impl FnMut(T),
) -> [T; N] {
    /// SAFETY invariant: `collected <= N` and the first `collected` elements of
    /// `edges` are initialized
    struct Collector<T, D: FnMut(T), const N: usize> {
        array: MaybeUninit<[T; N]>,
        collected: usize,
        drop: D,
    }

    impl<T, D: FnMut(T), const N: usize> Drop for Collector<T, D, N> {
        fn drop(&mut self) {
            // SAFETY: essentially the `AsRef` impl stabilized in Rust 1.95
            let edges = unsafe { &*(&raw const self.array as *const [MaybeUninit<T>; N]) };
            // SAFETY: `self.collected <= N` is part of the type invariant
            for v in unsafe { edges.get_unchecked(..self.collected) } {
                // SAFETY: The first `self.collected` elements are initialized.
                // No elements are duplicated because we forget `self.array`.
                (self.drop)(unsafe { v.assume_init_read() })
            }
        }
    }

    let mut collector = Collector {
        array: MaybeUninit::uninit(),
        collected: 0,
        drop,
    };

    // SAFETY: essentially the `AsMut` impl stabilized in Rust 1.95
    let array = unsafe { &mut *(&raw mut collector.array as *mut [MaybeUninit<T>; N]) };
    let mut it = it.into_iter();
    for (dst, src) in array.iter_mut().zip(it.by_ref()) {
        dst.write(src);
        collector.collected += 1;
    }
    let mut additional = false;
    for v in it {
        (collector.drop)(v);
        additional = true;
    }
    assert!(!additional, "iterator provided too many elements");
    assert_eq!(
        collector.collected, N,
        "iterator did not provide enough elements"
    );
    let collector = ManuallyDrop::new(collector);
    // SAFETY: `collector.collected` is `N`, so all array elements are
    // initialized. No elements are duplicated because we forget `collector`.
    unsafe { collector.array.assume_init_read() }
}

/// Collect `N` edges from an iterator
///
/// Panics if the iterator does not provide exactly `N` elements.
#[inline]
pub fn collect_edges_exact<M: Manager, const N: usize>(
    manager: &M,
    it: impl IntoIterator<Item = Own<M::Edge>>,
) -> [Own<M::Edge>; N] {
    collect_exact(it, |edge| manager.drop_edge(edge))
}
