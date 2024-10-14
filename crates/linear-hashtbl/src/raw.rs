//! Partially unsafe `RawTable` API

use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::mem::MaybeUninit;

#[cfg(feature = "allocator-api2")]
use allocator_api2::{alloc::Allocator, alloc::Global, boxed::Box, vec::Vec};

#[cfg(not(feature = "allocator-api2"))]
use {
    crate::__alloc::{Allocator, Global},
    alloc::{boxed::Box, vec::Vec},
};

// === Structs =================================================================

/// Raw hash table with a (partially) unsafe API
pub struct RawTable<T, S: Status = usize, A: Allocator + Clone = Global> {
    #[cfg(feature = "allocator-api2")]
    data: Box<[Slot<T, S>], A>,
    #[cfg(not(feature = "allocator-api2"))]
    data: Box<[Slot<T, S>]>,

    /// The number of items in the table
    len: usize,
    /// The number of free slots
    free: usize,

    phantom: PhantomData<A>,
}

struct Slot<T, S: Status = usize> {
    status: S,
    data: MaybeUninit<T>,
}

/// Immutable iterator over the entries of a [`RawTable`] in arbitrary order
///
/// This `struct` is created by [`RawTable::iter()`].
pub struct Iter<'a, T, S: Status = usize> {
    iter: core::slice::Iter<'a, Slot<T, S>>,
    len: usize,
}

/// Mutable iterator over the entries of a [`RawTable`] in arbitrary order
///
/// This `struct` is created by [`RawTable::iter_mut()`].
pub struct IterMut<'a, T, S: Status = usize> {
    iter: core::slice::IterMut<'a, Slot<T, S>>,
    len: usize,
}

/// Owning iterator over the entries of a [`RawTable`] in arbitrary order
///
/// This `struct` is created by the [`IntoIterator::into_iter()`] implementation
/// for `RawTable`. The `RawTable` cannot be used anymore after calling
/// [`IntoIterator::into_iter()`].
pub struct IntoIter<T, S: Status = usize, A: Allocator = Global> {
    #[cfg(feature = "allocator-api2")]
    iter: allocator_api2::vec::IntoIter<Slot<T, S>, A>,
    #[cfg(not(feature = "allocator-api2"))]
    iter: alloc::vec::IntoIter<Slot<T, S>>,
    len: usize,
    phantom: PhantomData<A>,
}

/// A draining iterator over the entries of a [`RawTable`]
///
/// This `struct` is created by [`RawTable::drain()`]. While this iterator moves
/// elements out just like [`IntoIter`], it does neither consume the
/// [`RawTable`] nor free its underlying memory.
///
/// Note: Forgetting `Drain` (e.g. via [`core::mem::forget()`]) and using the
/// table afterwards is a very bad idea. It is not `unsafe` but it causes
/// correctness issues since there exist non-empty slots while the length is
/// already set to `0`.
pub struct Drain<'a, T, S: Status = usize> {
    iter: core::slice::IterMut<'a, Slot<T, S>>,
    len: usize,
}

// === Status Impls ============================================================

/// Status of a slot in the hash table
///
/// A status can either be free, a tombstone, or a hash value of the associated
/// item. This can be implemented as a `u32` or a `usize`, for instance. Since
/// there are possibly more `u64` hash values than `Status` hash values, it is
/// fine to apply some mapping (e.g. only take the lowest `n - 1` bits).
///
/// Part of the idea of storing hash values is that we can avoid recomputing the
/// hash values when rehashing (resizing) the table. Consequently, we cannot
/// meaningfully address a table that is larger than the number of `Status` hash
/// values. This is why we have [`Self::check_capacity()`].
///
/// # Safety
///
/// All valid status values are either [`Status::FREE`], a [`Status::TOMBSTONE`]
/// or a hash value. [`Status::from_hash()`] must always return a hash value,
/// and [`Status::is_hash()`] must return `true` if and only if the given
/// `Status` is indeed a hash value.
pub unsafe trait Status: Copy + Eq {
    /// Marker for the slot being free
    const FREE: Self;
    /// Marker that is placed for deleted entries in a collision list
    const TOMBSTONE: Self;

    /// Convert a `u64` hash value (e.g. as returned by
    /// [`Hash`][core::hash::Hash]) into a `Status` hash value
    fn from_hash(hash: u64) -> Self;
    /// Panics if `capacity` exceeds the number of possible `Status` hash values
    fn check_capacity(capacity: usize);
    /// Check if the `Status` is a hash value
    fn is_hash(self) -> bool;
    /// Convert a `Status` hash value into a `usize`
    ///
    /// The result is unspecified if the status is free, a tombstone, or
    /// invalid. In this case, the implementation is allowed to panic.
    fn hash_as_usize(self) -> usize;
}

// SAFETY: `FREE`, `TOMBSTONE`, and hash values are distinct.
unsafe impl Status for usize {
    const FREE: Self = usize::MAX;
    const TOMBSTONE: Self = usize::MAX - 1;

    #[inline]
    fn from_hash(hash: u64) -> Self {
        hash as usize & (usize::MAX >> 1)
    }

    #[inline]
    #[track_caller]
    fn check_capacity(capacity: usize) {
        assert!(
            capacity <= 1 << (usize::BITS - 1),
            "requested capacity {capacity} is too large"
        );
    }

    #[inline]
    fn is_hash(self) -> bool {
        self <= usize::MAX >> 1
    }

    #[inline]
    fn hash_as_usize(self) -> usize {
        debug_assert!(self.is_hash());
        self
    }
}

// SAFETY: `FREE`, `TOMBSTONE`, and hash values are distinct.
unsafe impl Status for u32 {
    const FREE: Self = u32::MAX;
    const TOMBSTONE: Self = u32::MAX - 1;

    #[inline]
    fn from_hash(hash: u64) -> Self {
        hash as u32 & (u32::MAX >> 1)
    }

    #[inline]
    #[track_caller]
    fn check_capacity(capacity: usize) {
        assert!(
            capacity <= 1 << (u32::BITS - 1),
            "requested capacity {capacity} is too large"
        );
    }

    #[inline]
    fn is_hash(self) -> bool {
        self <= u32::MAX >> 1
    }

    #[inline]
    fn hash_as_usize(self) -> usize {
        debug_assert!(self.is_hash());
        self as usize
    }
}

// === Slot Impls ==============================================================

impl<T, S: Status> Slot<T, S> {
    const FREE: Self = Slot {
        status: S::FREE,
        data: MaybeUninit::uninit(),
    };
}

impl<T: Clone, S: Status> Clone for Slot<T, S> {
    fn clone(&self) -> Self {
        Self {
            status: self.status,
            data: if self.status.is_hash() {
                // SAFETY: hash status means that the data is initialized
                MaybeUninit::new(unsafe { self.data.assume_init_ref() }.clone())
            } else {
                MaybeUninit::uninit()
            },
        }
    }
}

// === RawTable Impls =========================================================

impl<T, S: Status> RawTable<T, S> {
    /// Create a new `HashTable` with zero capacity
    #[inline]
    pub fn new() -> Self {
        RawTable {
            data: Vec::new().into_boxed_slice(),
            len: 0,
            free: 0,
            phantom: PhantomData,
        }
    }

    /// Create a new `HashTable` with capacity for at least `capacity` elements
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = Self::next_capacity(capacity);
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(Slot::FREE);
        }
        RawTable {
            data: data.into_boxed_slice(),
            len: 0,
            free: capacity,
            phantom: PhantomData,
        }
    }
}

#[cfg(feature = "allocator-api2")]
impl<T, S: Status, A: Clone + Allocator> RawTable<T, S, A> {
    /// Create a new `HashTable` with zero capacity
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        RawTable {
            data: Vec::new_in(alloc).into_boxed_slice(),
            len: 0,
            free: 0,
            phantom: PhantomData,
        }
    }

    /// Create a new `HashTable` with capacity for at least `capacity` elements
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let capacity = Self::next_capacity(capacity);
        let mut data = Vec::with_capacity_in(capacity, alloc);
        for _ in 0..capacity {
            data.push(Slot {
                status: S::FREE,
                data: MaybeUninit::uninit(),
            });
        }
        RawTable {
            data: data.into_boxed_slice(),
            len: 0,
            free: capacity,
            phantom: PhantomData,
        }
    }
}

/// Numerator for the fraction of usable slots
const RATIO_N: usize = 3;
/// Denominator for the fraction of usable slots
const RATIO_D: usize = 4;

impl<T, S: Status, A: Clone + Allocator> RawTable<T, S, A> {
    /// Get the next largest array capacity for `requested` elements
    ///
    /// `requested` does not include the 25 % spare slots, while the returned
    /// value does. The result is guaranteed to be a power of two
    /// (or 0 in case `requested == 0`).
    ///
    /// This uses [`Status::check_capacity()`] to check if we can address the
    /// new array properly.
    #[inline]
    fn next_capacity(requested: usize) -> usize {
        if requested == 0 {
            return 0;
        }
        let capacity = core::cmp::max((requested * RATIO_D / RATIO_N).next_power_of_two(), 16);
        S::check_capacity(capacity);
        capacity
    }

    /// Get the number of elements stored in the table
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` iff no elements are stored in the table
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity (excluding spare slots)
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.len() / RATIO_D * RATIO_N
    }

    /// Get the number of slots (i.e. [`Self::capacity()`] plus spare slots)
    #[inline]
    pub fn slots(&self) -> usize {
        self.data.len()
    }

    /// Reserve space for `additional` elements
    ///
    /// If there are no other modifying operations in between, the next
    /// `additional` insertions are guaranteed to not cause a rehash (or resize)
    /// of the table.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let spare = additional + self.data.len() / RATIO_D * (RATIO_D - RATIO_N);
        if self.free < spare {
            self.reserve_rehash(additional)
        }
    }
    /// Rehash the table with capacity for `self.len + additional` elements
    #[cold]
    fn reserve_rehash(&mut self, additional: usize) {
        let new_cap = Self::next_capacity(self.len + additional);

        #[cfg(feature = "allocator-api2")]
        let (empty, mut new_data) = {
            let alloc = Box::allocator(&self.data).clone();
            let empty = Vec::new_in(alloc.clone());
            (empty, Vec::with_capacity_in(new_cap, alloc))
        };
        #[cfg(not(feature = "allocator-api2"))]
        let (empty, mut new_data) = (Vec::new(), Vec::with_capacity(new_cap));

        let old_data = core::mem::replace(&mut self.data, empty.into_boxed_slice()).into_vec();

        new_data.resize_with(new_cap, || Slot::FREE);

        let new_mask = new_cap - 1;
        for slot in old_data {
            let status = slot.status;
            if !status.is_hash() {
                continue;
            }
            // SAFETY: hash status means that the data is initialized
            let data = unsafe { slot.data.assume_init() };

            let mut index = status.hash_as_usize() & new_mask;
            loop {
                // SAFETY: masked index is in bounds (`new_data.len() != 0`)
                let new_slot = unsafe { new_data.get_unchecked_mut(index) };
                if new_slot.status == S::FREE {
                    new_slot.data.write(data);
                    new_slot.status = status;
                    break;
                }
                index = (index + 1) & new_mask;
            }
        }

        self.data = new_data.into_boxed_slice();
        self.free = new_cap - self.len;
    }

    /// Clear the table
    ///
    /// This does not change the table's capacity.
    pub fn clear(&mut self) {
        if self.len == 0 {
            return;
        }
        for slot in self.data.iter_mut() {
            let status = slot.status;
            slot.status = S::FREE;
            if status.is_hash() {
                // SAFETY: hash status means that the data is initialized. We
                // marked the slot as `FREE` above, so there is no danger of
                // double frees.
                unsafe { slot.data.assume_init_drop() };
                self.len -= 1;
                if self.len == 0 {
                    return;
                }
            }
        }
        // SAFETY: `self.len <= self.data.len()` holds by invariant
        unsafe { core::hint::unreachable_unchecked() }
    }

    /// [`Self::clear()`] but without dropping any entry
    pub fn clear_no_drop(&mut self) {
        if self.len == 0 {
            return;
        }
        for slot in self.data.iter_mut() {
            let status = slot.status;
            slot.status = S::FREE;
            if status.is_hash() {
                self.len -= 1;
                if self.len == 0 {
                    return;
                }
            }
        }
        // SAFETY: `self.len <= self.data.len()` holds by invariant
        unsafe { core::hint::unreachable_unchecked() }
    }

    /// Like [`Self::clear_no_drop()`], but also sets the capacity to 0
    ///
    /// If the space is not needed anymore, this should generally be faster
    /// [`Self::clear_no_drop()`], since we do not need to mark every slot as
    /// free.
    #[inline]
    pub fn reset_no_drop(&mut self) {
        self.len = 0;

        #[cfg(feature = "allocator-api2")]
        let empty = Vec::new_in(Box::allocator(&self.data).clone());
        #[cfg(not(feature = "allocator-api2"))]
        let empty = Vec::new();
        self.data = empty.into_boxed_slice();
    }

    /// Returns `true` iff there is an entry at `slot`
    ///
    /// # Safety
    ///
    /// `slot` must be less than `self.slots()`
    #[inline(always)]
    pub unsafe fn is_slot_occupied_unchecked(&self, slot: usize) -> bool {
        debug_assert!(slot < self.data.len());
        // SAFETY: `slot <= self.slots.len()` is ensured by the caller
        unsafe { self.data.get_unchecked(slot) }.status.is_hash()
    }

    /// Find the index of an element or a slot
    #[inline]
    pub fn find(&self, hash: u64, eq: impl Fn(&T) -> bool) -> Option<usize> {
        debug_assert_ne!(self.free, 0, "find may diverge");
        if self.len == 0 {
            return None;
        }

        debug_assert!(self.data.len().is_power_of_two());
        let mask = self.data.len() - 1;
        let mut index = hash as usize & mask;
        let hash_status = S::from_hash(hash);

        loop {
            // SAFETY: masked index is in bounds (`self.data.len() != 0`)
            let slot = unsafe { self.data.get_unchecked(index) };
            if slot.status == hash_status {
                // SAFETY: hash status means that the data is initialized
                if eq(unsafe { slot.data.assume_init_ref() }) {
                    return Some(index);
                }
            } else if slot.status == S::FREE {
                return None;
            }
            index = (index + 1) & mask;
        }
    }

    /// Find the index of an element or a slot to insert it
    ///
    /// Returns `Ok(index)` if the element was found and `Err(index)` for an
    /// insertion slot.
    ///
    /// `eq` is guaranteed to only be called for entries that are present in the
    /// table.
    #[inline]
    pub fn find_or_find_insert_slot(
        &mut self,
        hash: u64,
        eq: impl Fn(&T) -> bool,
    ) -> Result<usize, usize> {
        self.reserve(1);

        debug_assert!(self.data.len().is_power_of_two());
        let mask = self.data.len() - 1;
        let mut index = hash as usize & mask;
        let mut first_tombstone = None;
        let hash_status = S::from_hash(hash);

        loop {
            // SAFETY: masked index is in bounds (`self.data.len() != 0`)
            let slot = unsafe { self.data.get_unchecked(index) };
            if slot.status == hash_status {
                // SAFETY: hash status means that the data is initialized
                if eq(unsafe { slot.data.assume_init_ref() }) {
                    return Ok(index);
                }
            } else if slot.status == S::FREE {
                return Err(first_tombstone.unwrap_or(index));
            } else if slot.status == S::TOMBSTONE {
                first_tombstone = Some(index);
            }
            index = (index + 1) & mask;
        }
    }

    /// Get a reference to an entry in the table
    ///
    /// `hash` is the entry's hash value and `eq` returns true if the given
    /// element is the searched one.
    #[inline]
    pub fn get(&self, hash: u64, eq: impl Fn(&T) -> bool) -> Option<&T> {
        let index = self.find(hash, eq)?;
        debug_assert!(self.data[index].status.is_hash());
        // SAFETY: `find()` guarantees the returned index to be in bounds and
        // the entry there to be occupied.
        Some(unsafe { self.data.get_unchecked(index).data.assume_init_ref() })
    }

    /// Get a mutable reference to an entry in the table
    ///
    /// `hash` is the entry's hash value and `eq` returns true if the given
    /// element is the searched one.
    #[inline]
    pub fn get_mut(&mut self, hash: u64, eq: impl Fn(&T) -> bool) -> Option<&mut T> {
        let index = self.find(hash, eq)?;
        debug_assert!(self.data[index].status.is_hash());
        // SAFETY: `find()` guarantees the returned index to be in bounds and
        // the entry there to be occupied.
        Some(unsafe { self.data.get_unchecked_mut(index).data.assume_init_mut() })
    }

    /// Get a reference to the entry at `slot`
    ///
    /// # Safety
    ///
    /// `slot` must be the index of an occupied slot. This is the case if `slot`
    /// has been returned by [`RawTable::find()`] or
    /// [`RawTable::find_or_find_insert_slot()`] in the `Ok` case, and no
    /// modifications happened in between.
    #[inline]
    pub unsafe fn get_at_slot_unchecked(&self, slot: usize) -> &T {
        debug_assert!(self.data[slot].status.is_hash(), "slot is empty");
        // SAFETY: The caller ensures that `slot` is the index of an occupied
        // slot.
        unsafe { self.data.get_unchecked(slot).data.assume_init_ref() }
    }

    /// Get a mutable reference to the entry entry at `slot`
    ///
    /// # Safety
    ///
    /// `slot` must be the index of an occupied slot. This is the case if `slot`
    /// has been returned by [`RawTable::find()`] or
    /// [`RawTable::find_or_find_insert_slot()`] in the `Ok` case, and no
    /// modifications happened in between.
    #[inline]
    pub unsafe fn get_at_slot_unchecked_mut(&mut self, slot: usize) -> &mut T {
        debug_assert!(self.data[slot].status.is_hash(), "slot is empty");
        // SAFETY: The caller ensures that `slot` is the index of an occupied
        // slot.
        unsafe { self.data.get_unchecked_mut(slot).data.assume_init_mut() }
    }

    /// Insert `val` in `slot`
    ///
    /// `hash` is the hash value of `val`. Returns a mutable reference to the
    /// inserted value.
    ///
    /// # Safety
    ///
    /// `slot` must be the index of an empty slot. This is the case if `slot`
    /// has been returned by [`RawTable::find_or_find_insert_slot()`] in the
    /// `Err` case, and no modifications happened in between.
    #[inline]
    pub unsafe fn insert_in_slot_unchecked(&mut self, hash: u64, slot: usize, val: T) -> &mut T {
        debug_assert!(!self.data[slot].status.is_hash(), "slot is occupied");

        // SAFETY: Validity of the slot is guaranteed by the caller.
        let slot = unsafe { self.data.get_unchecked_mut(slot) };
        if slot.status != S::TOMBSTONE {
            debug_assert!(slot.status == S::FREE);
            self.free -= 1;
        }
        self.len += 1;
        let res = slot.data.write(val);
        slot.status = S::from_hash(hash);
        res
    }

    /// Find and remove an entry from the table, returning it
    ///
    /// `hash` is the entry's hash value and `eq` returns true if the given
    /// element is the searched one.
    #[inline]
    pub fn remove_entry(&mut self, hash: u64, eq: impl Fn(&T) -> bool) -> Option<T> {
        let index = self.find(hash, eq)?;
        // SAFETY: `find()` guarantees the returned index to be in bounds and
        // the entry there to be occupied.
        Some(unsafe { self.remove_at_slot_unchecked(index) })
    }

    /// Remove the entry at `slot`
    ///
    /// Returns the entry value.
    ///
    /// # Safety
    ///
    /// `slot` must be the index of an occupied slot. This is the case if `slot`
    /// has been returned by [`RawTable::find()`] or
    /// [`RawTable::find_or_find_insert_slot()`] in the `Ok` case, and no
    /// modifications happened in between.
    #[inline]
    pub unsafe fn remove_at_slot_unchecked(&mut self, slot: usize) -> T {
        debug_assert_ne!(self.len, 0);
        debug_assert!(self.data[slot].status.is_hash());
        let next_slot_index = (slot + 1) & (self.data.len() - 1);
        // SAFETY (next 2): The caller ensures that `slot` is in bounds, hence
        // `self.data.len() != 0` and `next_slot_index` index is in bounds, too.
        let next_slot_status = unsafe { self.data.get_unchecked(next_slot_index) }.status;
        let slot = unsafe { self.data.get_unchecked_mut(slot) };
        slot.status = if next_slot_status == S::FREE {
            self.free += 1;
            S::FREE
        } else {
            S::TOMBSTONE
        };
        self.len -= 1;
        // SAFETY: The slot's status was a hash meaning that the entry is
        // occupied. We set the status to `FREE` or `TOMBSTONE`, so we don't
        // duplicate the value.
        unsafe { slot.data.assume_init_read() }
    }

    /// Get an immutable iterator over the entries of the table
    #[inline]
    pub fn iter(&self) -> Iter<T, S> {
        Iter {
            iter: self.data.iter(),
            len: self.len,
        }
    }

    /// Get a mutable iterator over the entries of the table
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T, S> {
        IterMut {
            iter: self.data.iter_mut(),
            len: self.len,
        }
    }

    /// Get a draining iterator over the entries of the table
    ///
    /// A draining iterator removes all elements from the table but does not
    /// change the table's capacity.
    ///
    /// Note: Forgetting the returned `Drain` (e.g. via [`core::mem::forget()`])
    /// and using the table afterwards is a very bad idea. It is not `unsafe`
    /// but it causes correctness issues since there exist non-empty slots while
    /// the length is already set to `0`.
    #[inline]
    pub fn drain(&mut self) -> Drain<T, S> {
        let len = self.len;
        self.len = 0;
        self.free = self.data.len();
        Drain {
            iter: self.data.iter_mut(),
            len,
        }
    }

    /// Retain only the elements for which `predicate` returns `true`
    ///
    /// `predicate` is guaranteed to only be called for entries that are present
    /// in the table.
    /// `drop` is called for every element that is removed from the table.
    pub fn retain(&mut self, mut predicate: impl FnMut(&mut T) -> bool, mut drop: impl FnMut(T)) {
        if self.len == 0 {
            return;
        }
        debug_assert!(self.data.len() >= self.len);
        let mut i = self.len;
        let mut last_is_free = self.data[0].status == S::FREE;
        for slot in self.data.iter_mut().rev() {
            if !slot.status.is_hash() {
                if slot.status == S::FREE {
                    last_is_free = true;
                } else {
                    debug_assert!(slot.status == S::TOMBSTONE);
                    if last_is_free {
                        slot.status = S::FREE;
                        self.free += 1;
                    } else {
                        last_is_free = false;
                    }
                }
                continue;
            }
            // SAFETY: The slot's status is a hash value meaning that the data
            // is initialized.
            if !predicate(unsafe { slot.data.assume_init_mut() }) {
                self.len -= 1;
                if last_is_free {
                    slot.status = S::FREE;
                    self.free += 1;
                } else {
                    slot.status = S::TOMBSTONE;
                }
                // SAFETY: `slot.data` is initialized (see above). We set the
                // status to `FREE` or `TOMBSTONE` above, so we don't duplicate
                // the value.
                drop(unsafe { slot.data.assume_init_read() });
            }
            i -= 1;
            if i == 0 {
                if self.len < self.data.len() / RATIO_D * (RATIO_D - RATIO_N) {
                    // shrink the table
                    self.reserve_rehash(0);
                }
                return;
            }
        }
        // SAFETY: `self.len <= self.data.len()` holds by invariant
        unsafe { core::hint::unreachable_unchecked() }
    }
}

impl<T: Clone, S: Status, A: Clone + Allocator> Clone for RawTable<T, S, A> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            len: self.len,
            free: self.free,
            phantom: PhantomData,
        }
    }
}

impl<T, S: Status, A: Clone + Default + Allocator> Default for RawTable<T, S, A> {
    fn default() -> Self {
        RawTable {
            #[cfg(feature = "allocator-api2")]
            data: Vec::new_in(A::default()).into_boxed_slice(),
            #[cfg(not(feature = "allocator-api2"))]
            data: Vec::new().into_boxed_slice(),
            len: 0,
            free: 0,
            phantom: PhantomData,
        }
    }
}

impl<T, S: Status, A: Clone + Allocator> Drop for RawTable<T, S, A> {
    fn drop(&mut self) {
        if core::mem::needs_drop::<T>() {
            self.clear();
        }
    }
}

impl<T, S: Status, A: Clone + Allocator> IntoIterator for RawTable<T, S, A> {
    type Item = T;
    type IntoIter = IntoIter<T, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        IntoIter {
            // SAFETY: We move out of `this` (`this` is `ManuallyDrop` and never
            // dropped)
            iter: unsafe { core::ptr::read(&this.data) }
                .into_vec()
                .into_iter(),
            len: this.len,
            phantom: PhantomData,
        }
    }
}

// === Iterators ===============================================================

// --- Iter --------------------------------------------------------------------

impl<'a, T, S: Status> Iterator for Iter<'a, T, S> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            return None;
        }
        loop {
            let next = self.iter.next();
            debug_assert!(next.is_some());
            // SAFETY: The remaining part of the slice has at least `self.len`
            // elements by invariant
            let slot = unsafe { next.unwrap_unchecked() };
            if slot.status.is_hash() {
                self.len -= 1;
                // SAFETY: `slot.status` is a hash value, so `slot.data` is
                // initialized.
                return Some(unsafe { slot.data.assume_init_ref() });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T, S: Status> ExactSizeIterator for Iter<'_, T, S> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T, S: Status> FusedIterator for Iter<'_, T, S> {}

// --- IterMut -----------------------------------------------------------------

impl<'a, T, S: Status> Iterator for IterMut<'a, T, S> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            return None;
        }
        loop {
            let next = self.iter.next();
            debug_assert!(next.is_some());
            // SAFETY: The remaining part of the slice has at least `self.len`
            // elements by invariant
            let slot = unsafe { next.unwrap_unchecked() };
            if slot.status.is_hash() {
                self.len -= 1;
                // SAFETY: `slot.status` is a hash value, so `slot.data` is
                // initialized.
                return Some(unsafe { slot.data.assume_init_mut() });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T, S: Status> ExactSizeIterator for IterMut<'_, T, S> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T, S: Status> FusedIterator for IterMut<'_, T, S> {}

// --- IntoIter ----------------------------------------------------------------

impl<T, S: Status, A: Allocator> Iterator for IntoIter<T, S, A> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        loop {
            let next = self.iter.next();
            debug_assert!(next.is_some());
            // SAFETY: The remaining part of the vector has at least `self.len`
            // elements by invariant
            let slot = unsafe { next.unwrap_unchecked() };
            if slot.status.is_hash() {
                self.len -= 1;
                // SAFETY: `slot.status` is a hash value, so `slot.data` is
                // initialized.
                return Some(unsafe { slot.data.assume_init() });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T, S: Status, A: Allocator> ExactSizeIterator for IntoIter<T, S, A> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T, S: Status, A: Allocator> FusedIterator for IntoIter<T, S, A> {}

impl<T, S: Status, A: Allocator> Drop for IntoIter<T, S, A> {
    fn drop(&mut self) {
        while self.len != 0 {
            let next = self.iter.next();
            debug_assert!(next.is_some());
            // SAFETY: The remaining part of the vector has at least `self.len`
            // elements by invariant
            let mut slot = unsafe { next.unwrap_unchecked() };
            if slot.status.is_hash() {
                self.len -= 1;
                // SAFETY: `slot.status` is a hash value, so `slot.data` is
                // initialized. We drop/forget `slot`, so we don't risk a double
                // free.
                unsafe { slot.data.assume_init_drop() };
            }
        }
    }
}

// --- Drain -------------------------------------------------------------------

impl<T, S: Status> Iterator for Drain<'_, T, S> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        loop {
            let next = self.iter.next();
            debug_assert!(next.is_some());
            // SAFETY: The remaining part of the slice has at least `self.len`
            // elements by invariant
            let slot = unsafe { next.unwrap_unchecked() };
            let status = slot.status;
            slot.status = S::FREE;
            if status.is_hash() {
                self.len -= 1;
                // SAFETY: `slot.status` is a hash value, so `slot.data` is
                // initialized. We set `slot.status = S::FREE` above, so we
                // don't duplicate the value.
                return Some(unsafe { slot.data.assume_init_read() });
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T, S: Status> ExactSizeIterator for Drain<'_, T, S> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T, S: Status> FusedIterator for Drain<'_, T, S> {}

impl<T, S: Status> Drop for Drain<'_, T, S> {
    fn drop(&mut self) {
        while self.len != 0 {
            let next = self.iter.next();
            debug_assert!(next.is_some());
            // SAFETY: The remaining part of the slice has at least `self.len`
            // elements by invariant
            let slot = unsafe { next.unwrap_unchecked() };
            let status = slot.status;
            slot.status = S::FREE;
            if status.is_hash() {
                self.len -= 1;
                // SAFETY: `slot.status` is a hash value, so `slot.data` is
                // initialized. We set `slot.status = S::FREE` above, so we
                // don't drop the value twice.
                unsafe { slot.data.assume_init_drop() };
            }
        }
    }
}

// === Tests ===================================================================

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new_cap_len_0() {
        let table = RawTable::<u32, u32>::new();
        assert_eq!(table.capacity(), 0);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn insert_get_iter() {
        let mut table = RawTable::<u32, u32>::new();
        let numbers = [1, 4, 0, 5, 14, 4];
        let sorted = [0, 1, 4, 5, 14];

        for number in numbers {
            match table.find_or_find_insert_slot(number as u64, |&x| x == number) {
                Ok(_) => assert_eq!(number, 4),
                Err(slot) => unsafe {
                    table.insert_in_slot_unchecked(number as u64, slot, number);
                },
            }
        }
        assert_eq!(table.len(), 5);
        assert!(table.capacity() >= 5);

        for mut number in numbers {
            assert_eq!(table.get(number as u64, |&x| x == number), Some(&number));
            assert_eq!(
                table.get_mut(number as u64, |&x| x == number),
                Some(&mut number)
            );
        }

        let iter = table.iter();
        assert_eq!(iter.len(), 5);
        let mut res: Vec<u32> = iter.copied().collect();
        res.sort();
        assert_eq!(&res[..], &sorted[..]);

        let iter = table.iter_mut();
        assert_eq!(iter.len(), 5);
        let mut res: Vec<u32> = iter.map(|&mut x| x).collect();
        res.sort();
        assert_eq!(&res[..], &sorted[..]);

        let iter = table.into_iter();
        assert_eq!(iter.len(), 5);
        let mut res: Vec<u32> = iter.collect();
        res.sort();
        assert_eq!(&res[..], &sorted[..]);
    }

    #[test]
    fn retain_drain() {
        let mut table = RawTable::<u32, u32>::new();
        let numbers = [1, 2, 3, 4, 5, 6, 7];

        for number in numbers {
            match table.find_or_find_insert_slot(number as u64, |&x| x == number) {
                Ok(_) => unreachable!(),
                Err(slot) => unsafe {
                    table.insert_in_slot_unchecked(number as u64, slot, number);
                },
            }
        }
        assert_eq!(table.len(), 7);

        table.retain(|&mut x| x % 2 == 0, |x| assert!(x % 2 == 1));
        assert_eq!(table.len(), 3);

        let iter = table.drain();
        assert_eq!(iter.len(), 3);
        let mut res: Vec<u32> = iter.collect();
        res.sort();
        assert_eq!(&res[..], &[2, 4, 6]);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn remove() {
        let mut table = RawTable::<u32, u32>::new();
        let numbers = [4, 0, 2];

        for number in numbers {
            match table.find_or_find_insert_slot(number as u64, |&x| x == number) {
                Ok(_) => unreachable!(),
                Err(slot) => unsafe {
                    table.insert_in_slot_unchecked(number as u64, slot, number);
                },
            }
        }
        assert_eq!(table.len(), 3);

        assert_eq!(table.remove_entry(0, |&x| x == 0), Some(0));
        assert_eq!(table.len(), 2);
        assert_eq!(table.remove_entry(0, |&x| x == 0), None);
        assert_eq!(table.len(), 2);
    }
}
