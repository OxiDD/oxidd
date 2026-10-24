use std::sync::atomic::Ordering;

use oxidd_core::{AtomicLevelNo, LevelNo};

pub mod fixed_arity;

/// Base traits to be satisfied by inner nodes for oxidd-manager. This does not
/// include the [`InnerNode`][oxidd_core::InnerNode] trait.
///
/// # Safety
///
/// - The reference counter must be initialized to 2.
/// - [`Self::retain()`] increments the counter by 1
/// - [`Self::release()`] decrements the counter by 1 with (at least)
///   [`Release`][std::sync::atomic::Ordering::Release] order.
/// - [`Self::load_rc()`] loads the reference count with the specified load
///   order.
/// - An implementation must not modify the counter unless instructed externally
///   via the `retain()` or `release()` methods.
pub unsafe trait NodeBase: Eq + std::hash::Hash {
    /// Atomically increment the reference counter (with
    /// [`Relaxed`][std::sync::atomic::Ordering::Relaxed] order)
    ///
    /// This method is responsible for preventing an overflow of the reference
    /// counter.
    fn retain(&self);

    /// Atomically decrement the reference counter (with
    /// [`Release`][std::sync::atomic::Ordering::Release] order)
    ///
    /// Returns the previous reference count.
    ///
    /// A call to this function only modifies the counter value and never drops
    /// `self`.
    ///
    /// # Safety
    ///
    /// The caller must give up ownership of one reference to `self`.
    unsafe fn release(&self) -> usize;

    /// Atomically load the current reference count with the given `order`
    fn load_rc(&self, order: Ordering) -> usize;

    /// Whether this node type contains additional data that needs to be dropped
    /// to avoid memory leaks.
    ///
    /// If the node only consists of [`Edge`][crate::manager::Edge]s and types
    /// that implement [`Copy`], an implementation may return `false`. This may
    /// speed up dropping a diagram a lot.
    #[inline(always)]
    fn needs_drop() -> bool {
        true
    }
}

trait LevelInfo: From<LevelNo> {
    fn check(&self, check: impl FnOnce(LevelNo) -> bool) -> bool;
    fn assert_matches(&self, other: LevelNo);
}

impl LevelInfo for AtomicLevelNo {
    #[inline(always)]
    fn check(&self, check: impl FnOnce(LevelNo) -> bool) -> bool {
        check(self.load(Ordering::Relaxed))
    }

    #[inline(always)]
    #[track_caller]
    fn assert_matches(&self, other: LevelNo) {
        assert_eq!(
            self.load(Ordering::Relaxed),
            other,
            "the level number does not match"
        );
    }
}

/// Zero-sized type to indicate that a node does not store level information
pub struct NoLevel;
impl From<LevelNo> for NoLevel {
    fn from(_: LevelNo) -> Self {
        NoLevel
    }
}
impl LevelInfo for NoLevel {
    #[inline(always)]
    fn check(&self, _check: impl FnOnce(LevelNo) -> bool) -> bool {
        true
    }

    #[inline(always)]
    fn assert_matches(&self, _other: LevelNo) {}
}
