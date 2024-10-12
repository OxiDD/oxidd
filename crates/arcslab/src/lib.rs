//! [`slab`][slab] but with atomically reference counted items
//!
//! [slab]: https://crates.io/crates/slab

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
// We use const assertions for checking configurations and need to make sure
// that they are evaluated
#![allow(clippy::let_unit_value)]

use std::alloc;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::marker::PhantomPinned;
use std::mem::size_of;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr;
use std::ptr::addr_of;
use std::ptr::addr_of_mut;
use std::ptr::NonNull;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;

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
pub struct ArcSlab<I, D, const PAGE_SIZE: usize> {
    /// Additional data
    data: D,

    /// Internal fields
    ///
    /// These fields are cache padded to avoid false sharing with `data`:
    /// In many use cases, adding new items and cloning or dropping external
    /// handles does not modify `data`.
    int: CachePadded<ArcSlabInt<I, D, PAGE_SIZE>>,
}

struct ArcSlabInt<I, D, const PAGE_SIZE: usize> {
    /// Count of external references to the `ArcSlab`
    rc: AtomicUsize,

    /// Number of items
    items: AtomicUsize,

    /// Linked list of pages where the items are
    pages: Mutex<PageList<I, D, PAGE_SIZE>>,

    _pin: PhantomPinned,
}

/// List of the allocated pages
struct PageList<I, D, const PAGE_SIZE: usize> {
    /// Pointer to the current page, which contains at least a free slot
    ///
    /// SAFETY invariant: `current_page` always points to a valid `Page`
    current_page: NonNull<Page<I, D, PAGE_SIZE>>,

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
struct Page<I, D, const PAGE_SIZE: usize> {
    /// Pointer back to the [`ArcSlab`]
    ///
    /// Use `NonNull<ArcSlab<I, D>>` and not `ArcSlabRef<T>` because we don't
    /// want reference counting here.
    ///
    /// SAFETY invariant: `ArcSlab` is always a valid `ArcSlab` pointer. Only
    /// during `ArcSlab::new()`, the pointee may be uninitialized. After
    /// `Page::new()`, the value is never changed.
    slab: NonNull<ArcSlab<I, D, PAGE_SIZE>>,

    /// Pointer to the previous page in the page list (or null)
    ///
    /// SAFETY invariant: Either `prev` is null, or a valid `Page` pointer
    prev: *mut Page<I, D, PAGE_SIZE>,

    /// Actually, the element count is
    ///
    ///   PAGE_SIZE - (addr_of!(items) - addr_of!(slab)) / size_of::<Slot<I>>()
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

impl<I, D, const PAGE_SIZE: usize> ArcSlab<I, D, PAGE_SIZE> {
    const ASSERT_PAGE_SIZE_POWER_OF_TWO: () = assert!(
        PAGE_SIZE.is_power_of_two(),
        "PAGE_SIZE must be a power of 2"
    );

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
    pub unsafe fn new_with(init_data: impl FnOnce(*mut D)) -> ArcSlabRef<I, D, PAGE_SIZE> {
        // Ensure that our assertion is evaluated
        let _ = Self::ASSERT_PAGE_SIZE_POWER_OF_TWO;

        // Use a `Box` here such that we do not have to implement the allocation
        // ourselves. Inspired by the `Arc` implementation in the Rustonomicon.
        let mut boxed = Box::new(MaybeUninit::<ArcSlab<I, D, PAGE_SIZE>>::uninit());
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
        init_data(unsafe { addr_of_mut!((*ptr).data) });

        // SAFETY: `ptr` and `(*ptr).int` are valid, we have exclusive access
        // (`PageList::new()` and `Page::new()` do not access the slab).
        MaybeUninit::write(
            unsafe { uninit(addr_of_mut!((*ptr).int)) },
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
    pub fn new(data: D) -> ArcSlabRef<I, D, PAGE_SIZE> {
        // SAFETY: writing to `slot` is safe, we initialize the slot
        unsafe { Self::new_with(|slot| std::ptr::write(slot, data)) }
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

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> ArcSlab<I, D, PAGE_SIZE> {
    /// Get a free slot
    #[must_use]
    #[inline]
    pub fn add_item(&self, item: I) -> IntHandle<'_, I, D, PAGE_SIZE> {
        let slot = self.int.pages.lock().get_slot();
        self.int.items.fetch_add(1, Relaxed);
        let item = Slot {
            item: ManuallyDrop::new(item),
        };
        // SAFETY: the `slot` is valid and we have exclusive ownership
        unsafe { ptr::write(slot.as_ptr(), item) };

        IntHandle(slot, PhantomData)
    }
}

// SAFETY: it is safe to send `&ArcSlab<I, D, PAGE_SIZE>` to another thread
unsafe impl<I: Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Sync
    for ArcSlab<I, D, PAGE_SIZE>
{
}

// --- Implementations for `PageList` ------------------------------------------

impl<I, D, const PAGE_SIZE: usize> PageList<I, D, PAGE_SIZE> {
    /// Creates a new page list with the given page
    fn new(arc_slab: NonNull<ArcSlab<I, D, PAGE_SIZE>>) -> Self {
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

impl<I, D, const PAGE_SIZE: usize> Drop for PageList<I, D, PAGE_SIZE> {
    fn drop(&mut self) {
        // SAFETY: `current_page` is always a valid pointer.
        unsafe { Page::dealloc(self.current_page) };
    }
}

// --- Implementations for `Page` ----------------------------------------------

impl<I, D, const PAGE_SIZE: usize> Page<I, D, PAGE_SIZE> {
    /// Compute the layout for (de-)allocation
    ///
    /// It is SAFE to use this layout for [`alloc::alloc()`] and
    /// [`alloc::dealloc()`].
    #[inline]
    fn layout() -> alloc::Layout {
        // Assert that a page has space for at least one element. Should
        // evaluate to true at compile time.
        assert!(
            size_of::<Page<I, D, PAGE_SIZE>>() + size_of::<Slot<I>>() <= PAGE_SIZE,
            "PAGE_SIZE is too small, need space for at least one item"
        );
        // Calling `alloc::alloc()` with zero-sized layout is unsafe. Should
        // also evaluate to true at compile time.
        assert!(size_of::<Page<I, D, PAGE_SIZE>>() > 0);

        alloc::Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap()
    }

    /// Create a new (initialized) page.
    ///
    /// `prev` refers to the previous page in the page list.
    ///
    /// The returned `Page` pointer is always valid.
    #[cold]
    fn new(
        slab: NonNull<ArcSlab<I, D, PAGE_SIZE>>,
        prev: *mut Page<I, D, PAGE_SIZE>,
    ) -> NonNull<Self> {
        let layout = Self::layout();
        // SAFETY: The layout is safe for use with `alloc::alloc()`.
        let raw_ptr = unsafe { alloc::alloc(layout) } as *mut Self;

        let ptr = match NonNull::new(raw_ptr) {
            None => alloc::handle_alloc_error(layout),
            Some(ptr) => ptr,
        };

        // Use transparent hugepages on Linux
        #[cfg(all(not(miri), target_os = "linux"))]
        if PAGE_SIZE >= 2 * 1024 * 1024 {
            unsafe {
                // spell-checker:ignore MADV
                let _ = libc::madvise(
                    raw_ptr as *mut std::ffi::c_void,
                    PAGE_SIZE,
                    libc::MADV_HUGEPAGE,
                );
            }
        }

        // SAFETY: `raw_ptr` is valid (the pointee is uninitialized, but this is
        // allowed in context of `addr_of_mut!()`).
        let first_slot = unsafe { addr_of_mut!((*raw_ptr).items) as *mut Slot<I> };

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
                .add(PAGE_SIZE)
                .offset_from(first_slot as *mut u8)
        } / size_of::<Slot<I>>() as isize;
        let mut slot = first_slot;
        let last_slot = unsafe { slot.offset(count - 1) };
        while slot != last_slot {
            let next = unsafe { slot.offset(1) };
            unsafe { ptr::write(addr_of_mut!((*slot).next_free), next) };
            slot = next;
        }
        unsafe { ptr::write(addr_of_mut!((*last_slot).next_free), ptr::null_mut()) };

        ptr
    }

    /// Obtain a pointer to the page's first slot
    ///
    /// SAFETY: `ptr` must be valid (but the pointee not necessarily
    /// initialized).
    #[inline]
    unsafe fn first_slot(ptr: NonNull<Self>) -> NonNull<Slot<I>> {
        unsafe { NonNull::new_unchecked(addr_of_mut!((*ptr.as_ptr()).items) as *mut Slot<I>) }
    }

    /// Obtain a pointer to the `Page` from a pointer that points somewhere at
    /// the page (e.g. a `Slot<I>`)
    #[inline]
    fn page_ptr<P>(ptr: *mut P) -> *mut Self {
        sptr::Strict::map_addr(ptr, |addr| addr & !(PAGE_SIZE - 1)) as _
    }

    /// Obtain a the `ArcSlab` corresponding to the page
    ///
    /// SAFETY: `page` must point to a valid `Page`.
    #[inline]
    unsafe fn slab(page: *const Self) -> NonNull<ArcSlab<I, D, PAGE_SIZE>> {
        // SAFETY: `page` points to a valid `Page`. The `ArcSlab` field is kept
        // constant for the entire lifecycle of a `Page`.
        unsafe { ptr::read(addr_of!((*page).slab)) }
    }

    /// Free the given slot, i.e. add it to the page's free slot list and
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
    unsafe fn dealloc(mut page: NonNull<Page<I, D, PAGE_SIZE>>) {
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
    unsafe fn release<D, const PAGE_SIZE: usize>(this: NonNull<Self>, f: impl FnOnce(I)) {
        // SAFETY:
        // - `this.as_ref()`: `this` is valid
        // - `.item`: the slot is non-empty
        // - `.release()`: we give up ownership of one reference
        if unsafe { this.as_ref().item.release() } == 1 {
            drop_item::<I, D, PAGE_SIZE>(this, f);
        }

        #[cold]
        fn drop_item<I, D, const PAGE_SIZE: usize>(mut this: NonNull<Slot<I>>, f: impl FnOnce(I)) {
            atomic::fence(Acquire);

            // SAFETY: We have exclusive access to the slot, we don't use the
            // item again
            f(unsafe { ManuallyDrop::take(&mut this.as_mut().item) });
            // SAFETY: The slot is empty
            unsafe { Page::<I, D, PAGE_SIZE>::free_slot(this) };
        }
    }

    /// Decrease the reference counter and return the item if the counter
    /// reaches 0.
    ///
    /// SAFETY: `this` must be a valid slot pointer, the slot must be non-empty
    /// and there must not be any further accesses to `this` (if it was the last
    /// reference). For every call to this function, there must have been a
    /// preceding call to `retain()`, or an initial count.
    unsafe fn release_move<D, const PAGE_SIZE: usize>(mut this: NonNull<Self>) -> Option<I> {
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
        unsafe { Page::<I, D, PAGE_SIZE>::free_slot(this) };

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
pub struct ArcSlabRef<I, D, const PAGE_SIZE: usize>(NonNull<ArcSlab<I, D, PAGE_SIZE>>);

impl<I, D, const PAGE_SIZE: usize> ArcSlabRef<I, D, PAGE_SIZE> {
    /// Convert this `ArcSlabRef` into its underlying pointer
    ///
    /// This does not change any reference counts. To avoid a memory leak, the
    /// pointer must be converted back to an `ArcSlabRef` using
    /// [`Self::from_raw()`].
    #[inline]
    pub fn into_raw(this: Self) -> NonNull<ArcSlab<I, D, PAGE_SIZE>> {
        let ptr = this.0;
        std::mem::forget(this);
        ptr
    }

    /// Construct an `ArcSlabRef` from a raw pointer
    ///
    /// # Safety
    ///
    /// The pointer must have been obtained from
    /// [`ArcSlabRef::<I, D, PAGE_SIZE>::into_raw()`] (`I`, `D` and `PAGE_SIZE`
    /// must match). This function does not change any reference counters, so
    /// calling this function multiple times for the same pointer may lead to
    /// use after free bugs depending on the usage of the returned `ArcSlabRef`.
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<ArcSlab<I, D, PAGE_SIZE>>) -> Self {
        Self(ptr)
    }
}

impl<I, D, const PAGE_SIZE: usize> std::ops::Deref for ArcSlabRef<I, D, PAGE_SIZE> {
    type Target = ArcSlab<I, D, PAGE_SIZE>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: `ArcSlab` is reference counted
        unsafe { self.0.as_ref() }
    }
}

impl<I, D, const PAGE_SIZE: usize> Clone for ArcSlabRef<I, D, PAGE_SIZE> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: `ArcSlab` is reference counted
        unsafe { self.0.as_ref() }.retain();
        Self(self.0)
    }
}

impl<I, D, const PAGE_SIZE: usize> Drop for ArcSlabRef<I, D, PAGE_SIZE> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `ArcSlab` is reference counted, `self.0` is valid and there
        // are no further uses of `self.0`.
        unsafe { ArcSlab::release(self.0) };
    }
}

impl<I, D, const PAGE_SIZE: usize> PartialEq for ArcSlabRef<I, D, PAGE_SIZE> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<I, D, const PAGE_SIZE: usize> Eq for ArcSlabRef<I, D, PAGE_SIZE> {}

impl<I, D, const PAGE_SIZE: usize> Hash for ArcSlabRef<I, D, PAGE_SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<I, D, const PAGE_SIZE: usize> PartialOrd for ArcSlabRef<I, D, PAGE_SIZE> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<I, D, const PAGE_SIZE: usize> Ord for ArcSlabRef<I, D, PAGE_SIZE> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

// SAFETY: It is safe to send `ArcSlabRef`s to another thread.
unsafe impl<I: Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Send
    for ArcSlabRef<I, D, PAGE_SIZE>
{
}
// SAFETY: It is safe to send `&ArcSlabRef`s to another thread.
unsafe impl<I: Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Sync
    for ArcSlabRef<I, D, PAGE_SIZE>
{
}

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
pub struct IntHandle<'a, I: AtomicRefCounted, D, const PAGE_SIZE: usize>(
    NonNull<Slot<I>>,
    PhantomData<(D, &'a ())>,
);

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> IntHandle<'_, I, D, PAGE_SIZE> {
    /// Move the referenced item out in case `this` is the last reference,
    /// otherwise return `None`
    #[inline]
    pub fn into_inner(this: Self) -> Option<I> {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety
        unsafe { Slot::release_move::<D, PAGE_SIZE>(slot) }
    }

    /// Drop this handle with a "custom drop implementation" `f` for the item
    #[inline]
    pub fn drop_with(this: Self, f: impl FnOnce(I)) {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety
        unsafe { Slot::release::<D, PAGE_SIZE>(slot, f) };
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
    /// [`IntHandle::<I, D, PAGE_SIZE>::into_raw()`] (`I`, `D` and `PAGE_SIZE`
    /// must match). Furthermore, the caller must ensure that the [`ArcSlab`]
    /// outlives the created `IntHandle`.
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<I>) -> Self {
        // The cast is fine: `Slot<I>` is a `union` with `repr(C)` and
        // `ManuallyDrop<I>` has `repr(transparent)`
        Self(ptr.cast(), PhantomData)
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Clone for IntHandle<'_, I, D, PAGE_SIZE> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { Slot::retain(self.0) };
        Self(self.0, PhantomData)
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Drop for IntHandle<'_, I, D, PAGE_SIZE> {
    #[inline]
    fn drop(&mut self) {
        unsafe { Slot::release::<D, PAGE_SIZE>(self.0, |_| {}) };
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Deref for IntHandle<'_, I, D, PAGE_SIZE> {
    type Target = I;

    #[inline]
    fn deref(&self) -> &I {
        // SAFETY: the slot is occupied
        unsafe { &self.0.as_ref().item }
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> PartialEq for IntHandle<'_, I, D, PAGE_SIZE> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Eq for IntHandle<'_, I, D, PAGE_SIZE> {}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> PartialOrd for IntHandle<'_, I, D, PAGE_SIZE> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Ord for IntHandle<'_, I, D, PAGE_SIZE> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Hash for IntHandle<'_, I, D, PAGE_SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Send
    for IntHandle<'_, I, D, PAGE_SIZE>
{
}
unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Sync
    for IntHandle<'_, I, D, PAGE_SIZE>
{
}

// --- `ExtHandle` -------------------------------------------------------------

/// External item handle
///
/// Unlike [`IntHandle`], this item handle also contributes to the external
/// reference counter of the [`ArcSlab`]. Therefore, we can always obtain an
/// `&ArcSlab`.
#[repr(transparent)]
pub struct ExtHandle<I: AtomicRefCounted, D, const PAGE_SIZE: usize>(
    NonNull<Slot<I>>,
    PhantomData<D>,
);

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> ExtHandle<I, D, PAGE_SIZE> {
    /// Move the referenced item out in case `this` is the last reference,
    /// otherwise return `None`
    #[inline]
    pub fn into_inner(this: Self) -> Option<I> {
        let slot = this.0;
        std::mem::forget(this); // "first things first" for exception safety

        // The item must be released before the `ArcSlab`: `this` might be the
        // last reference, so otherwise we might have a use after free.
        let res = unsafe { Slot::release_move::<D, PAGE_SIZE>(slot) };
        let page = Page::<I, D, PAGE_SIZE>::page_ptr(slot.as_ptr());
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
        unsafe { Slot::release::<D, PAGE_SIZE>(slot, f) };
        let page = Page::<I, D, PAGE_SIZE>::page_ptr(slot.as_ptr());
        unsafe { ArcSlab::release(Page::slab(page)) };
    }

    /// Obtain the [`ArcSlab`] storing the referenced item
    #[inline]
    pub fn slab(this: &Self) -> &ArcSlab<I, D, PAGE_SIZE> {
        let page = Page::<I, D, PAGE_SIZE>::page_ptr(this.0.as_ptr());
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
    /// [`ExtHandle::<I, D, PAGE_SIZE>::into_raw()`] (`I`, `D` and `PAGE_SIZE`
    /// must match).
    #[inline]
    pub unsafe fn from_raw(ptr: NonNull<I>) -> Self {
        // The cast is fine: `Slot<I>` is a `union` with `repr(C)` and
        // `ManuallyDrop<I>` has `repr(transparent)`
        Self(ptr.cast(), PhantomData)
    }
}

impl<'a, I: AtomicRefCounted, D, const PAGE_SIZE: usize> From<IntHandle<'a, I, D, PAGE_SIZE>>
    for ExtHandle<I, D, PAGE_SIZE>
{
    fn from(value: IntHandle<'a, I, D, PAGE_SIZE>) -> Self {
        let handle = Self(value.0, PhantomData);
        std::mem::forget(value);
        Self::slab(&handle).retain();
        handle
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Clone for ExtHandle<I, D, PAGE_SIZE> {
    #[inline]
    fn clone(&self) -> Self {
        // SAFETY: the slot is occupied
        unsafe { Slot::retain(self.0) };
        Self::slab(self).retain();
        Self(self.0, PhantomData)
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Drop for ExtHandle<I, D, PAGE_SIZE> {
    #[inline]
    fn drop(&mut self) {
        // The item must be released before the `ArcSlab`: `this` might be the
        // last reference, so otherwise we might have a use after free.
        unsafe { Slot::release::<D, PAGE_SIZE>(self.0, |_| {}) };
        let page = Page::<I, D, PAGE_SIZE>::page_ptr(self.0.as_ptr());
        unsafe { ArcSlab::release(Page::slab(page)) };
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Deref for ExtHandle<I, D, PAGE_SIZE> {
    type Target = I;

    #[inline]
    fn deref(&self) -> &I {
        // SAFETY: the slot is occupied
        unsafe { &self.0.as_ref().item }
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> PartialEq for ExtHandle<I, D, PAGE_SIZE> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Eq for ExtHandle<I, D, PAGE_SIZE> {}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> PartialOrd for ExtHandle<I, D, PAGE_SIZE> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Ord for ExtHandle<I, D, PAGE_SIZE> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<I: AtomicRefCounted, D, const PAGE_SIZE: usize> Hash for ExtHandle<I, D, PAGE_SIZE> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Send
    for ExtHandle<I, D, PAGE_SIZE>
{
}
unsafe impl<I: AtomicRefCounted + Send + Sync, D: Send + Sync, const PAGE_SIZE: usize> Sync
    for ExtHandle<I, D, PAGE_SIZE>
{
}

// === Tests ===================================================================

#[cfg(test)]
mod tests {
    use crate::{ArcItem, ArcSlab};

    #[test]
    fn instantiate() {
        ArcSlab::<ArcItem<u8>, (), 1024>::new(());
    }
}
