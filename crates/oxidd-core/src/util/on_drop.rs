use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

use crate::Manager;

use crate::util::DropWith;

/// `OnDrop::new(data, f)` executes `f(data)` when dropped
///
/// This simplifies writing actions to be executed when unwinding.
#[derive(Debug)]
pub struct OnDrop<'a, T, D: FnOnce(&mut T)>(ManuallyDrop<(&'a mut T, D)>);

impl<'a, T, D: FnOnce(&mut T)> OnDrop<'a, T, D> {
    /// Create a new `OnDrop` handler
    #[inline(always)]
    pub fn new(data: &'a mut T, drop_handler: D) -> Self {
        Self(ManuallyDrop::new((data, drop_handler)))
    }

    /// Access the data
    #[inline(always)]
    pub fn data(&self) -> &T {
        self.0 .0
    }

    /// Access the data
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut T {
        self.0 .0
    }

    /// Cancel the handler
    #[inline(always)]
    pub fn cancel(mut self) -> (&'a mut T, D) {
        // SAFETY: we destruct/drop `self`, so it cannot be accessed afterwards
        unsafe { ManuallyDrop::take(&mut self.0) }
    }
}

impl<T, D: FnOnce(&mut T)> Drop for OnDrop<'_, T, D> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: we drop `self`, so it cannot be accessed afterwards
        let (data, handler) = unsafe { ManuallyDrop::take(&mut self.0) };
        handler(data)
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

/// Drop guard for edges to ensure that they are not leaked
pub struct EdgeDropGuard<'a, M: Manager> {
    /// Manager containing the edge
    pub manager: &'a M,
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
    /// Manager containing the edges
    pub manager: &'a M,
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
