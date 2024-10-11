use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

pub mod rwlock;

/// Invariant lifetime
pub type Invariant<'id> = PhantomData<fn(&'id ()) -> &'id ()>;

/// Untyped comparison of pointers
#[inline(always)]
pub fn ptr_eq_untyped<T, U>(a: *const T, b: *const U) -> bool {
    std::ptr::eq(a as *const (), b as *const ())
}

pub struct TryLock(AtomicBool);

impl TryLock {
    /// Create a new `TryLock` in unlocked state
    #[inline(always)]
    pub const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    /// Try to lock this lock
    ///
    /// Returns true on success
    #[inline(always)]
    pub fn try_lock(&self) -> bool {
        // If we read `false`, we acquired the lock, if we read `true`, we did
        // not.
        !self.0.swap(true, Ordering::Acquire)
    }

    /// Unlock this lock
    #[inline(always)]
    pub fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}
