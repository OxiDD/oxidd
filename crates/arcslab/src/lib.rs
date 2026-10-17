//! [`slab`][slab] but with atomically reference counted items
//!
//! [slab]: https://crates.io/crates/slab

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::alloc;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::{ManuallyDrop, MaybeUninit, size_of};
use std::ops::Deref;
use std::ptr;
use std::ptr::NonNull;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use crossbeam_utils::CachePadded;
use derive_where::derive_where;
use parking_lot::Mutex;

/// Page size information
///
/// We use types and a trait instead of generic const arguments for more
/// flexibility: Currently, associated constants cannot be used as generic const
/// arguments, but associated types can be used as generic arguments.
pub trait PageSizeInfo {
    /// Page size in bytes
    ///
    /// Must be a power of two.
    ///
    /// To efficiently allocate memory for items, they are stored on so-called
    /// pages. The larger the pages, the fewer allocations are needed. However,
    /// larger pages also potentially mean a larger waste of memory. In
    /// practice, it makes sense for pages to be at least as large as physical
    /// memory pages. To reduce cache misses in the translation lookaside buffer
    /// (TLB), one can leverage huge pages which are 2 MiB or 1 GiB large on
    /// x86_64. Hence, `2 * 1024 * 1024` is a good default.
    const PAGE_SIZE: usize;
}

/// Page size `N`
///
/// See [`PageSizeInfo`] for more details.
pub struct PageSize<const N: usize>;
impl<const N: usize> PageSizeInfo for PageSize<N> {
    const PAGE_SIZE: usize = N;
}

/// Atomically reference counted value
///
/// # Safety
///
/// The reference counter must be initialized to 1 (or a larger value,
/// be aware of potential memory leaks in this case). `retain()` must increment
/// the counter by 1 with [`Relaxed`] order, `release()` must decrement the
/// counter by 1 with [`Release`] order. An implementation must not modify the
/// counter unless instructed externally via the `retain()` or `release()`
/// methods.
pub unsafe trait AtomicRefCounted {
    /// Atomically increment the reference counter (with [`Relaxed`] order)
    ///
    /// This method is responsible for preventing an overflow of the reference
    /// counter.
    fn retain(&self);

    /// Atomically decrement the reference counter (with [`Release`] order)
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

    /// Read the current reference count (with [`Relaxed`] order)
    fn current(&self) -> usize;
}

/// Simple implementation of [`AtomicRefCounted`]: `T` plus a reference count
pub struct ArcItem<T> {
    rc: AtomicUsize,
    data: T,
}

impl<T> ArcItem<T> {
    /// Create a new `ArcItem`
    pub fn new(data: T) -> Self {
        Self {
            rc: AtomicUsize::new(1),
            data,
        }
    }
}

impl<T> Deref for ArcItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

unsafe impl<T> AtomicRefCounted for ArcItem<T> {
    fn retain(&self) {
        if self.rc.fetch_add(1, Relaxed) > (usize::MAX >> 1) {
            std::process::abort();
        }
    }

    unsafe fn release(&self) -> usize {
        self.rc.fetch_sub(1, Release)
    }

    fn current(&self) -> usize {
        self.rc.load(Relaxed)
    }
}

/// [`slab`][slab] but with atomically reference counted items
///
/// [slab]: https://crates.io/crates/slab
// SAFETY invariant: There are no `&mut ArcSlab<I, D>` references; after
// `ArcSlab::new()` until dropping it is always safe to create `&ArcSlab<I, D>`
// references.
#[repr(C)]
pub struct ArcSlab<I, D, S: PageSizeInfo> {
    /// Additional data
    data: D,

    /// Internal fields
    ///
    /// These fields are cache padded to avoid false sharing with `data`:
    /// In many use cases, adding new items and cloning or dropping external
    /// handles does not modify `data`.
    int: CachePadded<ArcSlabInt<I, D, S>>,
}

struct ArcSlabInt<I, D, S: PageSizeInfo> {
    /// Count of external references to the `ArcSlab`
    rc: AtomicUsize,

    /// Number of items
    items: AtomicUsize,

    /// Linked list of pages where the items are
    pages: Mutex<PageList<I, D, S>>,

    _pin: PhantomPinned,
}

/// List of the allocated pages
struct PageList<I, D, S: PageSizeInfo> {
    /// Pointer to the current page, which contains at least a free slot
    ///
    /// SAFETY invariant: `current_page` always points to a valid `Page`
    current_page: NonNull<Page<I, D, S>>,

    /// Pointer to the first free slot
    ///
    /// SAFETY invariant: `free_slot` always points to a valid, free `Slot`
    free_slot: NonNull<Slot<I>>,
}

/// Page for storing the items corresponding to a variable
///
/// A page must be allocated with an alignment equal to its size. Then it is
/// possible to obtain the address of the page from a pointer to one of the
/// slots.
///
/// It seems that the easiest way to perform the corresponding allocations is to
/// use [`alloc::alloc()`]. Since we cannot statically compute the slot count
/// given a size, we just rely on the `items` field to be the last (that is why
/// we need `repr(C)`) and to be properly aligned.
#[repr(C)]
struct Page<I, D, S: PageSizeInfo> {
    /// Pointer back to the [`ArcSlab`]
    ///
    /// Use `NonNull<ArcSlab<I, D>>` and not `ArcSlabRef<T>` because we don't
    /// want reference counting here.
    ///
    /// SAFETY invariant: `ArcSlab` is always a valid `ArcSlab` pointer. Only
    /// during `ArcSlab::new()`, the pointee may be uninitialized. After
    /// `Page::new()`, the value is never changed.
    slab: NonNull<ArcSlab<I, D, S>>,

    /// Pointer to the previous page in the page list (or null)
    ///
    /// SAFETY invariant: Either `prev` is null, or a valid `Page` pointer
    prev: *mut Page<I, D, S>,

    /// Actually, the element count is
    ///
    /// `P::PAGE_SIZE - (addr_of!(items) - addr_of!(slab)) /
    /// size_of::<Slot<I>>()`
    ///
    /// However, we cannot use generics in const expressions. By having an array
    /// with zero elements, we can at least ensure that `items` is properly
    /// aligned.
    items: [Slot<I>; 0],

    _pin: PhantomPinned,
}

#[repr(C)]
union Slot<I> {
    /// Pointer to the next free slot (or a null pointer)
    ///
    /// SAFETY invariant: If the pointer is non-null, it is a valid slot
    /// pointer.
    next_free: *mut Slot<I>,

    item: ManuallyDrop<I>,
}

// --- Implementations for `ArcSlab` -------------------------------------------

impl<I, D, S: PageSizeInfo> ArcSlab<I, D, S> {
    /// Create a new `ArcSlab` and initialize the `data` field using `init_data`
    ///
    /// # Safety
    ///
    /// Calling `init_data` must initialize the provided location.
    /// `init_data` may assume that `*mut D` is valid for writes and properly
    /// aligned (hence it is safe to call [`std::ptr::write()`] for that
    /// location). With respect to aliasing models such as Stacked Borrows or
    /// Tree Borrows, the pointer given to `init_data` is guaranteed to be
    /// tagged as the root of the allocation.
    pub unsafe fn new_with(init_data: impl FnOnce(*mut D)) -> ArcSlabRef<I, D, S> {
        const { assert!(S::PAGE_SIZE.is_power_of_two()) };

        // Use a `Box` here such that we do not have to implement the allocation
        // ourselves. Inspired by the `Arc` implementation in the Rustonomicon.
        let mut boxed = Box::new(MaybeUninit::<ArcSlab<I, D, S>>::uninit());
        let ptr = boxed.as_mut_ptr();
        let non_null_ptr = NonNull::new(ptr).unwrap();
        let _ = Box::into_raw(boxed); // we need `ptr: *mut ArcSlab<..>` instead of `*mut MaybeUninit<..>`

        /// Helper function to create a mutable `MaybeUninit<T>` reference from
        /// a raw `T` pointer.
        ///
        /// SAFETY: `ptr` needs to be properly aligned and valid for writes for
        /// lifetime `'a`.
        #[inline(always)]
        unsafe fn uninit<'a, T>(ptr: *mut T) -> &'a mut MaybeUninit<T> {
            unsafe { &mut *(ptr as *mut MaybeUninit<T>) }
        }

        // SAFETY: `ptr` and `(*ptr).data` are valid for writes
        init_data(unsafe { &raw mut (*ptr).data });

        // SAFETY: `ptr` and `(*ptr).int` are valid, we have exclusive access
        // (`PageList::new()` and `Page::new()` do not access the slab).
        MaybeUninit::write(
            unsafe { uninit(&raw mut (*ptr).int) },
            CachePadded::new(ArcSlabInt {
                rc: AtomicUsize::new(1),
                items: AtomicUsize::new(0),
                pages: Mutex::new(PageList::new(non_null_ptr)),
                _pin: PhantomPinned,
            }),
        );

        ArcSlabRef(non_null_ptr)
    }

    /// Create a new ArcSlab with the given data
    #[allow(clippy::new_ret_no_self)]
    pub fn new(data: D) -> ArcSlabRef<I, D, S> {
        // SAFETY: writing to `slot` is safe, we initialize the slot
        unsafe { Self::new_with(|slot| ptr::write(slot, data)) }
    }

    /// Increase the counter of external references
    pub fn retain(&self) {
        // From the Rustonomicon: "Using a relaxed ordering is alright here as
        // we don't need any atomic synchronization here as we're not modifying
        // or accessing the inner data."
        let old_rc = self.int.rc.fetch_add(1, Relaxed);
        // Validate the reference count: if the counter overflows, there might
        // be use-after-free bugs. Better abort for safety.
        if old_rc > (isize::MAX as usize) {
            std::process::abort();
        }
    }

    /// Get the number of items
    #[inline]
    pub fn num_items(&self) -> usize {
        self.int.items.load(Relaxed)
    }

    /// Get the shared data
    #[inline]
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Get the `ArcSlab` from the data pointer
    ///
    /// If `ptr` points to a `D` inside an `ArcSlab` that is referenced by at
    /// least an [`ArcSlabRef`] or an [`ExtHandle`], it is safe to dereference
    /// the returned pointer.
    #[inline(always)]
    pub fn from_data_ptr(ptr: *const D) -> *const Self {
        // We have `repr(C)` and `data` is the first element, so this is just a
        // plain cast.
        ptr.cast()
    }

    /// Decrease the counter of external references and drop the `ArcSlab` if
    /// the reference count reaches 0
    ///
    /// # Safety
    ///
    /// For every call to `release()` except one, there has to be a distinct
    /// preceding call to `retain()`. `this` needs to be valid. The `ArcSlab`
    /// must not be used after the last calls to `release()`.
    pub unsafe fn release(this: NonNull<Self>) {
        // Inspired by the `Arc` implementation from the Rustonomicon.
        //
        // We need release ordering here: Every other access to `rc` in this
        // thread, in particular increments, happen before the decrement. This
        // means that if we read `1` below, no other thread can have an
        // `ArcSlabRef` and this is the last one in the current thread.

        // SAFETY: `this` is reference counted
        if unsafe { this.as_ref() }.int.rc.fetch_sub(1, Release) != 1 {
            return;
        }

        // Ensure that dropping happens after `rc.fetch_sub()` in this thread
        atomic::fence(Acquire);

        // SAFETY: `rc` and `items` are 0, no `&ArcSlab<I, D>` anymore
        drop(unsafe { Box::from_raw(this.as_ptr()) });
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> ArcSlab<I, D, S> {
    /// Get a free slot
    #[must_use]
    #[inline]
    pub fn add_item(&self, item: I) -> IntHandle<'_, I, D, S> {
        let slot = self.int.pages.lock().get_slot();
        self.int.items.fetch_add(1, Relaxed);
        let item = Slot {
            item: ManuallyDrop::new(item),
        };
        // SAFETY: the `slot` is valid and we have exclusive ownership
        unsafe { ptr::write(slot.as_ptr(), item) };

        IntHandle::<I, D, S>(slot, PhantomData)
    }
}

// SAFETY: it is safe to send `&ArcSlab<I, D, S>` to another thread
unsafe impl<I: Send + Sync, D: Send + Sync, S: PageSizeInfo> Sync for ArcSlab<I, D, S> {}

// --- Implementations for `PageList` ------------------------------------------

impl<I, D, S: PageSizeInfo> PageList<I, D, S> {
    /// Creates a new page list with the given page
    fn new(arc_slab: NonNull<ArcSlab<I, D, S>>) -> Self {
        let page = Page::new(arc_slab, ptr::null_mut());
        // SAFETY: `page` is a valid `Page` pointer
        let free_slot = unsafe { Page::first_slot(page) };
        PageList {
            current_page: page,
            free_slot,
        }
    }

    /// Obtain a free slot
    ///
    /// The returned slot can safely be overwritten and be accessed
    /// independently of this page list.
    #[inline]
    fn get_slot(&mut self) -> NonNull<Slot<I>> {
        let slot = self.free_slot;

        // SAFETY: `self.free_slot` always points to a valid `Slot`. We have
        // exclusive access to the entire list. The slot is free, so we can
        // access the `next_free` field.
        if let Some(next) = NonNull::new(unsafe { slot.as_ref().next_free }) {
            self.free_slot = next;
            return slot;
        }

        // Create a new page
        let page = Page::new(
            // SAFETY: `current_page` always points to a valid page
            unsafe { self.current_page.as_ref().slab },
            self.current_page.as_ptr(),
        );
        // SAFETY: `page` is a valid `Page` pointer
        self.free_slot = unsafe { Page::first_slot(page) };
        self.current_page = page;

        slot
    }
}

impl<I, D, S: PageSizeInfo> Drop for PageList<I, D, S> {
    fn drop(&mut self) {
        // SAFETY: `current_page` is always a valid pointer.
        unsafe { Page::dealloc(self.current_page) };
    }
}

// --- Implementations for `Page` ----------------------------------------------

impl<I, D, S: PageSizeInfo> Page<I, D, S> {
    /// Compute the layout for (de-)allocation
    ///
    /// It is SAFE to use this layout for [`alloc::alloc()`] and
    /// [`alloc::dealloc()`].
    #[inline]
    fn layout() -> alloc::Layout {
        // Assert that a page has space for at least one element. Should
        // evaluate to true at compile time.
        assert!(
            size_of::<Page<I, D, S>>() + size_of::<Slot<I>>() <= S::PAGE_SIZE,
            "P is too small, need space for at least one item"
        );
        // Calling `alloc::alloc()` with zero-sized layout is unsafe. Should
        // also evaluate to true at compile time.
        assert!(size_of::<Page<I, D, S>>() > 0);

        alloc::Layout::from_size_align(S::PAGE_SIZE, S::PAGE_SIZE).unwrap()
    }

    /// Create a new (initialized) page.
    ///
    /// `prev` refers to the previous page in the page list.
    ///
    /// The returned `Page` pointer is always valid.
    #[cold]
    fn new(slab: NonNull<ArcSlab<I, D, S>>, prev: *mut Page<I, D, S>) -> NonNull<Self> {
        let layout = Self::layout();
        // SAFETY: The layout is safe for use with `alloc::alloc()`.
        let raw_ptr = unsafe { alloc::alloc(layout) } as *mut Self;

        let ptr = match NonNull::new(raw_ptr) {
            None => alloc::handle_alloc_error(layout),
            Some(ptr) => ptr,
        };

        // Use transparent hugepages on Linux
        #[cfg(all(not(miri), target_os = "linux"))]
        if S::PAGE_SIZE >= 2 * 1024 * 1024 {
            unsafe {
                // spell-checker:ignore MADV
                let _ = libc::madvise(
                    raw_ptr as *mut std::ffi::c_void,
                    S::PAGE_SIZE,
                    libc::MADV_HUGEPAGE,
                );
            }
        }

        // SAFETY: `raw_ptr` is valid.
        let first_slot = unsafe { (&raw mut (*raw_ptr).items) as *mut Slot<I> };

        let page = Page {
            slab,
            prev,
            items: [],
            _pin: PhantomPinned,
        };
        // SAFETY: `raw_ptr` is valid for writes and properly aligned.
        unsafe { ptr::write(raw_ptr, page) };

        // Initialize the slots on the page
        //
        // SAFETY (next five blocks): The pointers are within the valid range
        // and are properly aligned (we know that at least one element fits onto
        // a page).
        let count = unsafe {
            (raw_ptr as *mut u8)
                .add(S::PAGE_SIZE)
                .offset_from(first_slot as *mut u8)
        } / size_of::<Slot<I>>() as isize;
        let mut slot = first_slot;
        let last_slot = unsafe { slot.offset(count - 1) };
        while !ptr::eq(slot, last_slot) {
            let next = unsafe { slot.offset(1) };
            unsafe { ptr::write(&raw mut (*slot).next_free, next) };
            slot = next;
        }
        unsafe { ptr::write(&raw mut (*last_slot).next_free, ptr::null_mut()) };

        ptr
    }

    /// Obtain a pointer to the page's first slot
    ///
    /// SAFETY: `ptr` must be valid (but the pointee not necessarily
    /// initialized).
    #[inline]
    unsafe fn first_slot(ptr: NonNull<Self>) -> NonNull<Slot<I>> {
        unsafe { NonNull::new_unchecked((&raw mut (*ptr.as_ptr()).items) as *mut Slot<I>) }
    }

    /// Obtain a pointer to the `Page` from a pointer that points somewhere at
    /// the page (e.g., a `Slot<I>`)
    #[inline]
    fn page_ptr<P>(ptr: *mut P) -> *mut Self {
        ptr.map_addr(|addr| addr & !(S::PAGE_SIZE - 1)) as _
    }

    /// Obtain a the `ArcSlab` corresponding to the page
    ///
    /// SAFETY: `page` must point to a valid `Page`.
    #[inline]
    unsafe fn slab(page: *const Self) -> NonNull<ArcSlab<I, D, S>> {
        // SAFETY: `page` points to a valid `Page`. The `ArcSlab` field is kept
        // constant for the entire lifecycle of a `Page`.
        unsafe { ptr::read(&raw const (*page).slab) }
    }

    /// Free the given slot, i.e., add it to the page's free slot list and
    /// possibly park or reorder pages.
    ///
    /// SAFETY: The slot must be empty.
    unsafe fn free_slot(mut slot: NonNull<Slot<I>>) {
        // SAFETY: The slot pointer is valid, hence the page pointer is. There
        // is no exclusive reference to the ArcSlab.
        let slab = unsafe { Self::slab(Self::page_ptr(slot.as_ptr())).as_ref() };
        slab.int.items.fetch_sub(1, Relaxed);

        let mut page_list = slab.int.pages.lock();
        // SAFETY: The slot pointer is valid, `page_list.free_slot` always
        // contains a valid pointer.
        unsafe { slot.as_mut().next_free = page_list.free_slot.as_ptr() };
        page_list.free_slot = slot;
    }

    /// Deallocate a `Page` and all its predecessors
    ///
    /// SAFETY: Safe as long as `page` is the start of a valid chain (only
    /// `next` pointers matter). Every page in the chain needs to be empty.
    /// There must not be any accesses to pages of this chain.
    unsafe fn dealloc(mut page: NonNull<Page<I, D, S>>) {
        loop {
            // SAFETY: `page` is a valid `Page` pointer
            let prev = unsafe { page.as_ref().prev };
            // SAFETY: `page` is valid for dropping and properly aligned
            unsafe { ptr::drop_in_place(page.as_ptr()) };
            // SAFETY: `page` was allocated by the same allocator, using
            // `Self::layout()`.
            unsafe { alloc::dealloc(page.as_ptr() as *mut u8, Self::layout()) };
            if let Some(prev) = NonNull::new(prev) {
                page = prev;
            } else {
                return;
            }
        }
    }
}

// --- Implementations for `Slot` ----------------------------------------------

impl<I: AtomicRefCounted> Slot<I> {
    /// Increase the reference counter
    ///
    /// SAFETY: The slot must be non-empty
    #[inline]
    unsafe fn retain(this: NonNull<Self>) {
        // SAFETY: The slot is non-empty
        let item = unsafe { &this.as_ref().item };
        item.retain();
    }

    /// Decrease the reference counter
    ///
    /// SAFETY: `this` must be a valid slot pointer, the slot must be non-empty
    /// and there must not be any further accesses to `this` (if it was the last
    /// reference). For every call to this function, there must have been a
    /// preceding call to `retain()`, or an initial count.
    #[inline]
    unsafe fn release<D, S: PageSizeInfo>(this: NonNull<Self>, f: impl FnOnce(I)) {
        // SAFETY:
        // - `this.as_ref()`: `this` is valid
        // - `.item`: the slot is non-empty
        // - `.release()`: we give up ownership of one reference
        if unsafe { this.as_ref().item.release() } == 1 {
            drop_item::<I, D, S>(this, f);
        }

        #[cold]
        fn drop_item<I, D, S: PageSizeInfo>(mut this: NonNull<Slot<I>>, f: impl FnOnce(I)) {
            atomic::fence(Acquire);

            // SAFETY: We have exclusive access to the slot, we don't use the
            // item again
            f(unsafe { ManuallyDrop::take(&mut this.as_mut().item) });
            // SAFETY: The slot is empty
            unsafe { Page::<I, D, S>::free_slot(this) };
        }
    }

    /// Decrease the reference counter and return the item if the counter
    /// reaches 0.
    ///
    /// SAFETY: `this` must be a valid slot pointer, the slot must be non-empty
    /// and there must not be any further accesses to `this` (if it was the last
    /// reference). For every call to this function, there must have been a
    /// preceding call to `retain()`, or an initial count.
    unsafe fn release_move<D, S: PageSizeInfo>(mut this: NonNull<Self>) -> Option<I> {
        // SAFETY:
        // - `this.as_ref()`: `this` is valid
        // - `.item`: the slot is non-empty
        // - `.release()`: we give up ownership of one reference
        if unsafe { this.as_ref().item.release() } != 1 {
            return None;
        }

        atomic::fence(Acquire);

        // SAFETY: We have exclusive access to the slot, we don't use the item
        // again
        let res = unsafe { ManuallyDrop::take(&mut this.as_mut().item) };
        // SAFETY: The slot is empty
        unsafe { Page::<I, D, S>::free_slot(this) };

        Some(res)
    }
}

// === Smart Pointers ==========================================================

// --- `ArcSlabRef` ------------------------------------------------------------

/// Reference to the [`ArcSlab`]
///
/// In some way similar to [`Arc<ArcSlab<I, D>>`][std::sync::Arc], however, we
/// only want to drop the `ArcSlab` after all `ArcSlabRef`s *and* all
/// [`ExtHandle`]s are dropped, and we do not want to store `Arc`s along with
/// the `ExtHandle`s. This is the reason for this custom reference type.
#[repr(transparent)]
#[derive_where(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArcSlabRef<I, D, S: PageSizeInfo>(NonNull<ArcSlab<I, D, S>>);

impl<I, D, S: PageSizeInfo> ArcSlabRef<I, D, S> {
    /// Convert this `ArcSlabRef` into its underlying pointer
    ///
    /// This does not change any reference counts. To avoid a memory leak, the
    /// pointer must be converted back to an `ArcSlabRef` using
    /// [`Self::from_raw()`].
    #[inline]
    pub fn into_raw(this: Self) -> NonNull<ArcSlab<I, D, S>> {
        let ptr = this.0;
        std::mem::forget(this);
        ptr
    }

    /// Construct an `ArcSlabRef` from a raw pointer
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained from
    /// [`ArcSlabRef::<I, D, S>::into_raw()`] (`I`, `D` and `S` must match).
    /// This function does not change any reference counters, so calling this
    /// function multiple times for the same pointer may lead to use after free
    /// bugs depending on the usage of the returned `ArcSlabRef`.
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<ArcSlab<I, D, S>>) -> Self {
        Self(ptr)
    }
}

impl<I, D, S: PageSizeInfo> std::ops::Deref for ArcSlabRef<I, D, S> {
    type Target = ArcSlab<I, D, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: `ArcSlab` is reference counted
        unsafe { self.0.as_ref() }
    }
}

impl<I, D, S: PageSizeInfo> Clone for ArcSlabRef<I, D, S> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: `ArcSlab` is reference counted
        unsafe { self.0.as_ref() }.retain();
        Self(self.0)
    }
}

impl<I, D, S: PageSizeInfo> Drop for ArcSlabRef<I, D, S> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `ArcSlab` is reference counted, `self.0` is valid and there
        // are no further uses of `self.0`.
        unsafe { ArcSlab::release(self.0) };
    }
}

// SAFETY: It is safe to send `ArcSlabRef`s to another thread.
unsafe impl<I: Send + Sync, D: Send + Sync, S: PageSizeInfo> Send for ArcSlabRef<I, D, S> {}
// SAFETY: It is safe to send `&ArcSlabRef`s to another thread.
unsafe impl<I: Send + Sync, D: Send + Sync, S: PageSizeInfo> Sync for ArcSlabRef<I, D, S> {}

// --- `IntHandle` -------------------------------------------------------------

/// Item handle for "internal" usage
///
/// These item handles are constrained with a lifetime because they do not
/// contribute to the external reference counter. This means that dereferencing
/// an `IntHandle` after dropping all [`ArcSlabRef`]s and [`ExtHandle`]s would
/// lead to a use-after-free (which we avoid via the lifetime).
//
// SAFETY invariant: The `ArcSlab` outlives `'a`
#[repr(transparent)]
#[derive_where(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntHandle<'a, I: AtomicRefCounted, D, S: PageSizeInfo>(
    NonNull<Slot<I>>,
    #[derive_where(skip)] PhantomData<(D, S, &'a ())>,
);

impl<I: AtomicRefCounted, D, S: PageSizeInfo> IntHandle<'_, I, D, S> {
    /// Move the referenced item out in case `this` is the last reference,
    /// otherwise return `None`
    #[inline]
    pub fn into_inner(this: Self) -> Option<I> {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety
        unsafe { Slot::release_move::<D, S>(slot) }
    }

    /// Move the referenced item out without checking whether `this` is the last
    /// reference
    ///
    /// # Safety
    ///
    /// The caller must ensure that `this` is the last reference. Beware of
    /// relaxed memory (e.g., use [`Acquire`] ordering to check that `this` is
    /// the last reference).
    #[inline]
    pub unsafe fn force_into_inner(this: Self) -> I {
        let mut slot = this.0;
        std::mem::forget(this);

        // SAFETY: We have (exclusive) access to the non-empty slot
        debug_assert_eq!(unsafe { &slot.as_ref().item }.current(), 1);

        // SAFETY: We have exclusive access to the slot, we don't use the item
        // again
        let res = unsafe { ManuallyDrop::take(&mut slot.as_mut().item) };
        // SAFETY: The slot is empty
        unsafe { Page::<I, D, S>::free_slot(slot) };

        res
    }

    /// Drop this handle with a "custom drop implementation" `f` for the item
    #[inline]
    pub fn drop_with(this: Self, f: impl FnOnce(I)) {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety
        unsafe { Slot::release::<D, S>(slot, f) };
    }

    /// Convert this handle into its underlying pointer
    ///
    /// This does not change any reference counts. To avoid a memory leak, the
    /// pointer must be converted back to an `IntHandle` using
    /// [`IntHandle::from_raw()`].
    #[inline]
    pub fn into_raw(this: Self) -> NonNull<I> {
        // The cast is fine: `Slot<I>` is a `union` with `repr(C)` and
        // `ManuallyDrop<I>` has `repr(transparent)`
        let ptr = this.0.cast::<I>();
        std::mem::forget(this);
        ptr
    }

    /// Construct an `IntHandle` from a raw pointer
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained from
    /// [`IntHandle::<I, D, S>::into_raw()`] (`I`, `D` and `S` must match).
    /// Furthermore, the caller must ensure that the [`ArcSlab`]
    /// outlives the created `IntHandle`.
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<I>) -> Self {
        // The cast is fine: `Slot<I>` is a `union` with `repr(C)` and
        // `ManuallyDrop<I>` has `repr(transparent)`
        Self(ptr.cast(), PhantomData)
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> Clone for IntHandle<'_, I, D, S> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { Slot::retain(self.0) };
        Self(self.0, PhantomData)
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> Drop for IntHandle<'_, I, D, S> {
    #[inline]
    fn drop(&mut self) {
        unsafe { Slot::release::<D, S>(self.0, |_| {}) };
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> Deref for IntHandle<'_, I, D, S> {
    type Target = I;

    #[inline]
    fn deref(&self) -> &I {
        // SAFETY: the slot is occupied
        unsafe { &self.0.as_ref().item }
    }
}

unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, S: PageSizeInfo> Send
    for IntHandle<'_, I, D, S>
{
}
unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, S: PageSizeInfo> Sync
    for IntHandle<'_, I, D, S>
{
}

// --- `ExtHandle` -------------------------------------------------------------

/// External item handle
///
/// Unlike [`IntHandle`], this item handle also contributes to the external
/// reference counter of the [`ArcSlab`]. Therefore, we can always obtain an
/// `&ArcSlab`.
#[repr(transparent)]
#[derive_where(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExtHandle<I: AtomicRefCounted, D, S: PageSizeInfo>(
    NonNull<Slot<I>>,
    #[derive_where(skip)] PhantomData<(D, S)>,
);

impl<I: AtomicRefCounted, D, S: PageSizeInfo> ExtHandle<I, D, S> {
    /// Move the referenced item out in case `this` is the last reference,
    /// otherwise return `None`
    #[inline]
    pub fn into_inner(this: Self) -> Option<I> {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety

        // The item must be released before the `ArcSlab`: `this` might be the
        // last reference, so otherwise we might have a use after free.
        let res = unsafe { Slot::release_move::<D, S>(slot) };
        let page = Page::<I, D, S>::page_ptr(slot.as_ptr());
        unsafe { ArcSlab::release(Page::slab(page)) };
        res
    }

    /// Drop this handle with a "custom drop implementation" `f` for the item
    #[inline]
    pub fn drop_with(this: Self, f: impl FnOnce(I)) {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety

        // The item must be released before the `ArcSlab`: `this` might be the
        // last reference, so otherwise we might have a use after free.
        unsafe { Slot::release::<D, S>(slot, f) };
        let page = Page::<I, D, S>::page_ptr(slot.as_ptr());
        unsafe { ArcSlab::release(Page::slab(page)) };
    }

    /// Obtain the [`ArcSlab`] storing the referenced item
    #[inline]
    pub fn slab(this: &Self) -> &ArcSlab<I, D, S> {
        let page = Page::<I, D, S>::page_ptr(this.0.as_ptr());
        unsafe { Page::slab(page).as_ref() }
    }

    /// Convert this handle into its underlying pointer
    ///
    /// This does not change any reference counts. To avoid a memory leak, the
    /// pointer must be converted back to an `ExtHandle` using
    /// [`ExtHandle::from_raw()`].
    #[inline]
    pub fn into_raw(this: Self) -> NonNull<I> {
        // The cast is fine: `Slot<I>` is a `union` with `repr(C)` and
        // `ManuallyDrop<I>` has `repr(transparent)`
        let ptr = this.0.cast::<I>();
        std::mem::forget(this);
        ptr
    }

    /// Construct an `ExtHandle` from a raw pointer
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained from
    /// [`ExtHandle::<I, D, S>::into_raw()`] (`I`, `D` and `S` must match).
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<I>) -> Self {
        // The cast is fine: `Slot<I>` is a `union` with `repr(C)` and
        // `ManuallyDrop<I>` has `repr(transparent)`
        Self(ptr.cast(), PhantomData)
    }
}

impl<'a, I: AtomicRefCounted, D, S: PageSizeInfo> From<IntHandle<'a, I, D, S>>
    for ExtHandle<I, D, S>
{
    fn from(value: IntHandle<'a, I, D, S>) -> Self {
        let handle = Self(value.0, PhantomData);
        std::mem::forget(value);
        Self::slab(&handle).retain();
        handle
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> Clone for ExtHandle<I, D, S> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: the slot is occupied
        unsafe { Slot::retain(self.0) };
        Self::slab(self).retain();
        Self(self.0, PhantomData)
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> Drop for ExtHandle<I, D, S> {
    #[inline]
    fn drop(&mut self) {
        // The item must be released before the `ArcSlab`: `this` might be the
        // last reference, so otherwise we might have a use after free.
        unsafe { Slot::release::<D, S>(self.0, |_| {}) };
        let page = Page::<I, D, S>::page_ptr(self.0.as_ptr());
        unsafe { ArcSlab::release(Page::slab(page)) };
    }
}

impl<I: AtomicRefCounted, D, S: PageSizeInfo> Deref for ExtHandle<I, D, S> {
    type Target = I;

    #[inline]
    fn deref(&self) -> &I {
        // SAFETY: the slot is occupied
        unsafe { &self.0.as_ref().item }
    }
}

unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, S: PageSizeInfo> Send
    for ExtHandle<I, D, S>
{
}
unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, S: PageSizeInfo> Sync
    for ExtHandle<I, D, S>
{
}

// === Tests ===================================================================

#[cfg(test)]
mod tests {
    use crate::{ArcItem, ArcSlab, PageSize};

    #[test]
    fn instantiate() {
        ArcSlab::<ArcItem<u8>, (), PageSize<1024>>::new(());
    }
}
