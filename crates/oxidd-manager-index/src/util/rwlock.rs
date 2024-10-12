//! A read/write lock just like [`parking_lot::RwLock`] but with `repr(C)` such
//! that we can obtain a `RwLock` pointer from a data (`T`) pointer.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;

use parking_lot::lock_api::RawRwLock;

/// A read/write lock just like [`parking_lot::RwLock`]
pub struct RwLock<T> {
    data: UnsafeCell<T>,
    lock: parking_lot::RawRwLock,
}

impl<T> RwLock<T> {
    /// Offset of the encapsulated data of type `T`
    pub const DATA_OFFSET: usize = std::mem::offset_of!(Self, data);

    /// Create a new `ManagerLock`
    #[inline]
    pub fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            lock: parking_lot::RawRwLock::INIT,
        }
    }

    /// Acquire a shared lock
    #[inline]
    pub fn shared(&self) -> RwLockSharedGuard<'_, T> {
        self.lock.lock_shared();
        RwLockSharedGuard(self, PhantomData)
    }

    /// Acquire an exclusive lock
    #[inline]
    pub fn exclusive(&self) -> RwLockExclusiveGuard<'_, T> {
        self.lock.lock_exclusive();
        RwLockExclusiveGuard(self, PhantomData)
    }

    /// Get a mutable reference to the encapsulated data
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Get a pointer to the encapsulated data
    #[inline(always)]
    pub const fn data_ptr(&self) -> *mut T {
        self.data.get()
    }
}

/// RAII structure used to release the shared read access of a lock when
/// dropped.
pub struct RwLockSharedGuard<'a, T>(&'a RwLock<T>, PhantomData<*mut ()>);
impl<T> Deref for RwLockSharedGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        let ptr = self.0.data.get();
        // SAFETY: we have a shared lock
        unsafe { &*ptr }
    }
}
impl<T> Drop for RwLockSharedGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: we have a shared lock
        unsafe { self.0.lock.unlock_shared() }
    }
}

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
pub struct RwLockExclusiveGuard<'a, T>(&'a RwLock<T>, PhantomData<*mut ()>);
impl<T> Deref for RwLockExclusiveGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        let ptr = self.0.data.get();
        // SAFETY: we have a shared lock
        unsafe { &*ptr }
    }
}
impl<T> DerefMut for RwLockExclusiveGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = self.0.data.get();
        // SAFETY: we have an exclusive lock
        unsafe { &mut *ptr }
    }
}
impl<T> Drop for RwLockExclusiveGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: we have an exclusive lock
        unsafe { self.0.lock.unlock_exclusive() }
    }
}

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}