use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

mod var_level_map;
pub use var_level_map::VarLevelMap;
pub mod rwlock;

#[inline(always)]
pub fn hash_children<E: Hash, H: Hasher>(children: impl IntoIterator<Item = E>, state: &mut H) {
    for child in children {
        child.hash(state);
    }
}

/// Invariant lifetime
pub type Invariant<'id> = PhantomData<fn(&'id ()) -> &'id ()>;

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
