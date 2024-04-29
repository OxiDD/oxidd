#![deny(unsafe_op_in_unsafe_fn)]
#![no_std]

extern crate alloc;

use ::alloc::alloc::alloc;
use ::alloc::alloc::alloc_zeroed;
use ::alloc::alloc::dealloc;
use ::alloc::alloc::realloc;
use ::alloc::alloc::Layout;
use ::core::ptr::NonNull;
use ::sptr::invalid_mut;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;

const LARGE: usize = 2 * 1024 * 1024; // 2 MiB

#[inline(always)]
#[allow(unused_variables)] // for `cfg(miri)`
const fn is_large(size: usize) -> bool {
    #[cfg(not(miri))]
    let res = size >= LARGE;
    #[cfg(miri)]
    let res = false;
    res
}

//const HUGE: usize = 1 * 1024 * 1024 * 1024; // 1 GiB

//#[inline(always)]
//const fn is_huge(size: usize) -> bool {
//    #[cfg(not(miri))]
//    let res = size >= HUGE;
//    #[cfg(miri)]
//    let res = false;
//    res
//}

/// The global allocator, but with target dependent features for huge pages
///
/// Large parts of this implementation are taken from
/// `allocator_api2::alloc::Global`.
#[derive(Copy, Clone, Default, Debug)]
pub struct HugeAlloc;

impl HugeAlloc {
    #[inline(always)]
    fn alloc_impl(&self, layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            // SAFETY: `layout.align()` is a power of two, hence non-zero
            return Ok(unsafe {
                NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(
                    invalid_mut(layout.align()),
                    0,
                ))
            });
        }

        let ptr = if is_large(size) {
            let Ok(layout) = layout.align_to(LARGE) else {
                return Err(AllocError);
            };

            // SAFETY: `layout` has non-zero size
            let ptr = if zeroed {
                unsafe { alloc_zeroed(layout) }
            } else {
                unsafe { alloc(layout) }
            };

            // madvise for transparent huge pages, see
            // https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html
            #[cfg(all(target_os = "linux", not(miri)))]
            unsafe {
                // spell-checker:ignore MADV
                let _ = libc::madvise(ptr as *mut core::ffi::c_void, size, libc::MADV_HUGEPAGE);
            }

            #[allow(clippy::let_and_return)] // for non-Linux platforms
            ptr
        } else {
            // SAFETY: `layout` has non-zero size
            if zeroed {
                unsafe { alloc_zeroed(layout) }
            } else {
                unsafe { alloc(layout) }
            }
        };

        if ptr.is_null() {
            Err(AllocError)
        } else {
            // SAFETY: `ptr` is non-null
            Ok(unsafe { NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(ptr, size)) })
        }
    }

    /// SAFETY: Same as [`Allocator::grow`]
    #[inline(always)]
    unsafe fn grow_impl(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        debug_assert!(
            new_size >= old_size,
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        if old_size == 0 {
            // nothing to deallocate
            return self.alloc_impl(new_layout, zeroed);
        }

        // The current `alloc::alloc::System::realloc` implementation (unix)
        // falls back to alloc, copy, and dealloc for alignments of a few bytes.
        // So we don't bother with `realloc` for hugepages.
        if !is_large(new_size) && old_layout.align() == new_layout.align() {
            // SAFETY: `new_size` is non-zero as `old_size` and greater than or
            // equal to `new_size`. The alignment stays the same. We can assume
            // that the memory was allocated by this allocator.
            let raw_ptr = unsafe { realloc(ptr.as_ptr(), old_layout, new_size) };
            let Some(ptr) = NonNull::new(raw_ptr) else {
                return Err(AllocError);
            };
            if zeroed {
                // SAFETY: The range is valid for writes, the pointer is
                // properly aligned.
                unsafe { raw_ptr.add(old_size).write_bytes(0, new_size - old_size) };
            }
            return Ok(unsafe {
                NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), new_size))
            });
        }

        let new_ptr = self.alloc_impl(new_layout, zeroed)?;
        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_size`, both the old and new memory allocation are valid for
        // reads and writes for `old_size` bytes. Also, because the old
        // allocation wasn't yet deallocated, it cannot overlap `new_ptr`. Thus,
        // the call to `copy_nonoverlapping` is safe. The safety contract
        // for `dealloc` must be upheld by the caller.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr().cast(), old_size);
            self.deallocate(ptr, old_layout);
        }
        Ok(new_ptr)
    }
}

unsafe impl Allocator for HugeAlloc {
    #[inline(always)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, false)
    }

    #[inline(always)]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, true)
    }

    #[inline(always)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        if is_large(size) {
            let layout = layout.align_to(LARGE).expect("invalid layout");

            // SAFETY: `layout` has non-zero size. If the provided `layout` is
            // the equal to the one used for allocation (which the caller must
            // ensure), then we also raised the alignment there.
            unsafe { dealloc(ptr.as_ptr(), layout) }
        } else if size != 0 {
            // SAFETY: `layout` has non-zero size, other conditions must be
            // upheld by the caller.
            unsafe { dealloc(ptr.as_ptr(), layout) }
        }
    }

    #[inline(always)]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.grow_impl(ptr, old_layout, new_layout, false) }
    }

    #[inline(always)]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.grow_impl(ptr, old_layout, new_layout, true) }
    }

    #[inline(always)]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        debug_assert!(
            new_size <= old_size,
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        if new_size == 0 {
            // SAFETY: conditions must be upheld by the caller
            unsafe { self.deallocate(ptr, old_layout) };
            // SAFETY: `layout.align()` is a power of two, hence non-zero
            return Ok(unsafe {
                NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(
                    invalid_mut(new_layout.align()),
                    0,
                ))
            });
        }

        // The current `alloc::alloc::System::realloc` implementation (unix)
        // falls back to alloc, copy, and dealloc for alignments of a few bytes.
        // So we don't bother with `realloc` for hugepages.
        if !is_large(new_size) && old_layout.align() == new_layout.align() {
            // SAFETY: `new_size` is non-zero as `old_size`. The alignment stays
            // the same. We can assume that the memory was allocated by this
            // allocator.
            let raw_ptr = unsafe { realloc(ptr.as_ptr(), old_layout, new_size) };
            let Some(ptr) = NonNull::new(raw_ptr) else {
                return Err(AllocError);
            };
            // SAFETY: `ptr` is non-null
            return Ok(unsafe {
                NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), new_size))
            });
        }

        let new_ptr = self.allocate(new_layout)?;
        // SAFETY: because `new_size` must be smaller than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid
        // for reads and writes for `new_size` bytes. Also, because the old
        // allocation wasn't yet deallocated, it cannot overlap `new_ptr`. Thus,
        // the call to `copy_nonoverlapping` is safe. The safety contract
        // for `dealloc` must be upheld by the caller.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr().cast(), new_size);
            self.deallocate(ptr, old_layout);
        }
        Ok(new_ptr)
    }
}
