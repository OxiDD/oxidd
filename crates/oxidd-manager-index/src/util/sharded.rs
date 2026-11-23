//! A hash set split into independently locked shards
//!
//! Elements are looked up by keys `Q: Hash + Equivalent<E>`, in the style of
//! `hashbrown` and `indexmap`: a key is hashed exactly once, and that hash
//! selects both the shard and the bucket in the shard's table. The tables
//! store the hash of every element, so elements themselves are never hashed;
//! the keys used to insert and look up an element just need to agree on its
//! hash. The hash function is given by the `H: Hasher + Default` type
//! parameter.

use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::DerefMut;

use crossbeam_utils::CachePadded;
use equivalent::Equivalent;
use parking_lot::{Mutex, MutexGuard};

#[cfg(not(feature = "hugealloc"))]
type RawTable<E> = linear_hashtbl::raw::RawTable<E, u32>;
#[cfg(not(feature = "hugealloc"))]
type RawIntoIter<E> = linear_hashtbl::raw::IntoIter<E, u32>;
#[cfg(feature = "hugealloc")]
type RawTable<E> = linear_hashtbl::raw::RawTable<E, u32, hugealloc::HugeAlloc>;
#[cfg(feature = "hugealloc")]
type RawIntoIter<E> = linear_hashtbl::raw::IntoIter<E, u32, hugealloc::HugeAlloc>;

/// Number of shards (a power of two) to use for `threads` threads
///
/// Like DashMap, this uses four times as many shards as there are threads,
/// rounded up to the next power of two. This keeps the probability that two
/// threads contend for the same shard low.
pub(crate) fn shard_count(threads: usize) -> usize {
    let n = threads.max(1).saturating_mul(4).next_power_of_two();
    // `shard_index()` takes the shard index from the hash bits 40..64
    n.min(1 << 24)
}

/// The hash of `key`
#[inline]
fn hash_key<H: Hasher + Default, Q: Hash + ?Sized>(key: &Q) -> u64 {
    let mut hasher = H::default();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Index of the shard for `hash` among `n` (a power of two) shards
///
/// Uses bits that the shard's table uses neither for the bucket index (low
/// bits) nor for the tag (high bits) as long as a shard has fewer than 2^40
/// buckets.
#[inline(always)]
fn shard_index(hash: u64, n: usize) -> usize {
    debug_assert!(n.is_power_of_two());
    (hash >> 40) as usize & (n - 1)
}

// === Table ===================================================================

/// A single shard: a hash table of `E`s
///
/// Dropping a `Table` does **not** drop its elements, they are leaked. This
/// is for elements that cannot be dropped on their own (e.g., edges that need
/// their manager to be dropped). Such elements must be removed via
/// [`Shards::drain()`], [`Sharded::into_inner()`], etc. beforehand.
pub(crate) struct Table<E>(RawTable<E>);

impl<E> Default for Table<E> {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<E> Drop for Table<E> {
    #[inline]
    fn drop(&mut self) {
        self.0.reset_no_drop();
    }
}

impl<E> IntoIterator for Table<E> {
    type Item = E;
    type IntoIter = RawIntoIter<E>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        // SAFETY: We move out of `this` (and forget `this`)
        let table = unsafe { std::ptr::read(&this.0) };
        table.into_iter()
    }
}

impl<E> Table<E> {
    #[inline]
    fn entry<T, Q>(table: T, hash: u64, key: &Q) -> Entry<T>
    where
        T: DerefMut<Target = Self>,
        Q: Equivalent<E> + ?Sized,
    {
        let mut table = table;
        match table
            .0
            .find_or_find_insert_slot(hash, |e| key.equivalent(e))
        {
            Ok(slot) => Entry::Occupied(OccupiedEntry { table, slot }),
            Err(slot) => Entry::Vacant(VacantEntry { table, hash, slot }),
        }
    }
}

// === Entries =================================================================

/// An entry of a [`Table`] for some key, borrowing the table via `T` (a
/// reference or a lock guard)
pub(crate) enum Entry<T> {
    Occupied(OccupiedEntry<T>),
    Vacant(VacantEntry<T>),
}

/// An element present in a [`Table`]
pub(crate) struct OccupiedEntry<T> {
    table: T,
    /// SAFETY invariant: `slot` is occupied in `table`, which has not been
    /// modified since `slot` was obtained
    slot: usize,
}

/// A place to insert an element into a [`Table`]
pub(crate) struct VacantEntry<T> {
    table: T,
    hash: u64,
    /// SAFETY invariant: `slot` was returned for `hash` by
    /// `find_or_find_insert_slot` on `table`, which has not been modified
    /// since
    slot: usize,
}

impl<E, T: DerefMut<Target = Table<E>>> OccupiedEntry<T> {
    /// Get the element
    #[inline]
    pub(crate) fn get(&self) -> &E {
        // SAFETY: by invariant
        unsafe { self.table.0.get_at_slot_unchecked(self.slot) }
    }

    /// Remove the element from the table
    #[inline]
    pub(crate) fn remove(mut self) -> E {
        // SAFETY: by invariant
        unsafe { self.table.0.remove_at_slot_unchecked(self.slot) }
    }
}

impl<E, T: DerefMut<Target = Table<E>>> VacantEntry<T> {
    /// Insert `element`, which must be equivalent to the key this entry was
    /// created for
    #[inline]
    pub(crate) fn insert(mut self, element: E) {
        // SAFETY: by invariant
        unsafe {
            self.table
                .0
                .insert_in_slot_unchecked(self.hash, self.slot, element)
        };
    }
}

// === Sharded =================================================================

/// A hash set of `E`s split into a power of two many shards, each behind its own
/// (cache padded) lock, using the hash function `H`
///
/// An operation on a single element locks only the shard of its key (see
/// [`Self::lock_entry()`]), whereas operations on the whole set lock all
/// shards (see [`Self::lock()`]).
pub(crate) struct Sharded<E, H> {
    shards: Box<[CachePadded<Mutex<Table<E>>>]>,
    hasher: PhantomData<fn() -> H>,
}

impl<E, H: Hasher + Default> Sharded<E, H> {
    /// Create an empty set with `shards` shards
    ///
    /// `shards` must be a power of two, see [`shard_count()`].
    pub(crate) fn new(shards: usize) -> Self {
        assert!(shards.is_power_of_two());
        Self {
            shards: (0..shards)
                .map(|_| CachePadded::new(Mutex::new(Table::default())))
                .collect(),
            hasher: PhantomData,
        }
    }

    /// Lock the shard for `key` and look `key` up in it
    #[inline]
    pub(crate) fn lock_entry<Q>(&self, key: &Q) -> Entry<MutexGuard<'_, Table<E>>>
    where
        Q: Hash + Equivalent<E> + ?Sized,
    {
        let hash = hash_key::<H, Q>(key);
        let shard = self.shards[shard_index(hash, self.shards.len())].lock();
        Table::entry(shard, hash, key)
    }

    /// Lock all shards, in order
    #[inline]
    pub(crate) fn lock(&self) -> ShardedGuard<'_, E, H> {
        Shards::new(self.shards.iter().map(|m| m.lock()).collect())
    }

    /// Consume `self` and iterate over all elements
    pub(crate) fn into_inner(self) -> impl Iterator<Item = E> {
        self.shards
            .into_vec()
            .into_iter()
            .flat_map(|m| CachePadded::into_inner(m).into_inner())
    }
}

// === Shards ==================================================================

/// A shard that gives access to its [`Table`]: locked (`MutexGuard`) or
/// owned (`Table`)
pub(crate) trait Shard {
    /// Element type of the table
    type Elem;

    /// Get the table
    fn table(&self) -> &Table<Self::Elem>;

    /// Get the table
    fn table_mut(&mut self) -> &mut Table<Self::Elem>;
}

impl<E> Shard for MutexGuard<'_, Table<E>> {
    type Elem = E;

    #[inline(always)]
    fn table(&self) -> &Table<E> {
        self
    }

    #[inline(always)]
    fn table_mut(&mut self) -> &mut Table<E> {
        self
    }
}

impl<E> Shard for Table<E> {
    type Elem = E;

    #[inline(always)]
    fn table(&self) -> &Table<E> {
        self
    }

    #[inline(always)]
    fn table_mut(&mut self) -> &mut Table<E> {
        self
    }
}

/// The shards of a [`Sharded`], all accessible at once: either locked (see
/// [`ShardedGuard`]) or taken out of the [`Sharded`] (see [`TakenShards`])
pub(crate) struct Shards<S, H>(Vec<S>, PhantomData<fn() -> H>);

/// All shards of a [`Sharded`], locked
pub(crate) type ShardedGuard<'a, E, H> = Shards<MutexGuard<'a, Table<E>>, H>;

/// The contents of all shards of a [`Sharded`], moved out of it (see
/// [`ShardedGuard::take()`])
pub(crate) type TakenShards<E, H> = Shards<Table<E>, H>;

impl<S, H> Shards<S, H> {
    #[inline]
    fn new(shards: Vec<S>) -> Self {
        Self(shards, PhantomData)
    }
}

impl<S: Shard, H: Hasher + Default> Shards<S, H> {
    /// Get the number of elements
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.iter().map(|s| s.table().0.len()).sum()
    }

    /// Reserve space for `additional` elements (assuming they are evenly
    /// distributed over the shards)
    pub(crate) fn reserve(&mut self, additional: usize) {
        let per_shard = additional.div_ceil(self.0.len());
        for s in &mut self.0 {
            s.table_mut().0.reserve(per_shard);
        }
    }

    /// Get the element equivalent to `key`
    #[inline]
    pub(crate) fn find<Q>(&self, key: &Q) -> Option<&S::Elem>
    where
        Q: Hash + Equivalent<S::Elem> + ?Sized,
    {
        let hash = hash_key::<H, Q>(key);
        self.0[shard_index(hash, self.0.len())]
            .table()
            .0
            .get(hash, |e| key.equivalent(e))
    }

    /// Look `key` up
    #[inline]
    pub(crate) fn entry<Q>(&mut self, key: &Q) -> Entry<&mut Table<S::Elem>>
    where
        Q: Hash + Equivalent<S::Elem> + ?Sized,
    {
        let hash = hash_key::<H, Q>(key);
        let n = self.0.len();
        Table::entry(self.0[shard_index(hash, n)].table_mut(), hash, key)
    }

    /// Remove all elements for which `predicate` returns `false`, passing
    /// them to `drop`
    pub(crate) fn retain(
        &mut self,
        mut predicate: impl FnMut(&mut S::Elem) -> bool,
        mut drop: impl FnMut(S::Elem),
    ) {
        for s in &mut self.0 {
            s.table_mut().0.retain(&mut predicate, &mut drop);
        }
    }

    /// Iterate over all elements
    #[inline]
    pub(crate) fn iter(&self) -> ShardsIter<linear_hashtbl::raw::Iter<'_, S::Elem, u32>> {
        ShardsIter(self.0.iter().map(|s| s.table().0.iter()).collect())
    }

    /// Remove and iterate over all elements
    pub(crate) fn drain<'s>(&'s mut self) -> impl Iterator<Item = S::Elem> + 's
    where
        S::Elem: 's,
    {
        self.0.iter_mut().flat_map(|s| s.table_mut().0.drain())
    }

    /// Swap the contents of all shards of `self` and `other`
    pub(crate) fn swap(&mut self, other: &mut Self) {
        debug_assert_eq!(self.0.len(), other.0.len());
        for (a, b) in self.0.iter_mut().zip(other.0.iter_mut()) {
            std::mem::swap(a.table_mut(), b.table_mut());
        }
    }
}

impl<S: Shard, H> Shards<S, H> {
    /// Move the contents of all shards out, leaving empty shards behind
    ///
    /// This is O(number of shards): the tables are moved, not rebuilt.
    pub(crate) fn take(&mut self) -> TakenShards<S::Elem, H> {
        Shards::new(
            self.0
                .iter_mut()
                .map(|s| std::mem::take(s.table_mut()))
                .collect(),
        )
    }
}

// === ShardsIter ==============================================================

/// Concatenation of the iterators over the shards (in reverse order)
pub(crate) struct ShardsIter<I>(Vec<I>);

impl<I: ExactSizeIterator> Iterator for ShardsIter<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let it = self.0.last_mut()?;
            if let Some(item) = it.next() {
                return Some(item);
            }
            // Exhausted iterators are dropped, so they are never called
            // again. Hence, `ShardsIter` is fused.
            self.0.pop();
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for ShardsIter<I> {
    #[inline]
    fn len(&self) -> usize {
        self.0.iter().map(|it| it.len()).sum()
    }
}

impl<I: ExactSizeIterator> FusedIterator for ShardsIter<I> {}
