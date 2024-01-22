#[cfg(not(feature = "parking_lot"))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "parking_lot")]
pub use parking_lot::{Mutex, RawMutex};

#[cfg(not(feature = "parking_lot"))]
pub struct RawMutex(AtomicBool);

#[cfg(not(feature = "parking_lot"))]
unsafe impl parking_lot::lock_api::RawMutex for RawMutex {
    // Regarding the lint: the intent here is not to modify the `AtomicBool` in
    // a const context but to create the `RawMutex` in a const context.
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self(AtomicBool::new(false));

    type GuardMarker = parking_lot::lock_api::GuardNoSend;

    #[inline]
    fn lock(&self) {
        loop {
            if self.0.swap(true, Ordering::Acquire) {
                // was true -> is locked
                std::hint::spin_loop();
            } else {
                // was false -> is now locked
                return;
            }
        }
    }

    #[inline(always)]
    fn try_lock(&self) -> bool {
        // If we read false, we acquired the lock, if we read true, we did not.
        !self.0.swap(true, Ordering::Acquire)
    }

    #[inline(always)]
    unsafe fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

#[cfg(not(feature = "parking_lot"))]
#[allow(unused)] // only used by `fifo` and `lfu`
pub type Mutex<T> = parking_lot::lock_api::Mutex<RawMutex, T>;
