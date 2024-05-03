//! Fixed-size apply cache with least-frequently used eviction strategy

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;

use crate::util::Mutex;

use oxidd_core::util::Borrowed;
use oxidd_core::util::DropWith;
use oxidd_core::util::GCContainer;
use oxidd_core::ApplyCache;
use oxidd_core::Manager;

// spell-checker:ignore freqs

#[cfg(feature = "hugealloc")]
type Box<T> = allocator_api2::boxed::Box<T, hugealloc::HugeAlloc>;
#[cfg(feature = "hugealloc")]
type Vec<T> = allocator_api2::vec::Vec<T, hugealloc::HugeAlloc>;

/// Fixed-size apply cache with least-frequently used eviction strategy
pub struct LFUApplyCache<M: Manager, O, H, const ARITY: usize, const BUCKET_SIZE: usize>(
    Box<[Mutex<Bucket<M, O, ARITY, BUCKET_SIZE>>]>,
    PhantomData<H>,
);

impl<M: Manager, O: Copy + Eq, H, const ARITY: usize, const BUCKET_SIZE: usize> DropWith<M::Edge>
    for LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
{
    fn drop_with(self, drop_edge: impl Fn(M::Edge)) {
        let mut this = ManuallyDrop::new(self);
        for bucket in &mut *this.0 {
            bucket.get_mut().clear_with(&drop_edge);
        }
        // SAFETY: The passed pointer is a `&mut` ref, `this` is `ManuallyDrop`
        unsafe { std::ptr::drop_in_place(&mut this.0) };
    }
}

impl<M: Manager, O, H, const ARITY: usize, const BUCKET_SIZE: usize> Drop
    for LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
{
    fn drop(&mut self) {
        eprintln!(
            "`LFUApplyCache` must not be dropped. Use `DropWith::drop_with()`. Backtrace:\n{}",
            std::backtrace::Backtrace::capture()
        );

        #[cfg(feature = "static_leak_check")]
        {
            extern "C" {
                #[link_name = "\n\n`LFUApplyCache`s must not be dropped. Use `DropWith::drop_with()`.\n"]
                fn trigger() -> !;
            }
            // SAFETY: This won't call a function as it will trigger a linker error.
            unsafe { trigger() }
        }
    }
}

/// A fixed-size bucket which may contain up to `SIZE` entries with the same
/// hash
struct Bucket<M: Manager, O, const ARITY: usize, const SIZE: usize> {
    num_entries: u8,
    // Invariant: all entries up to num_entries are initialized.
    entries: [MaybeUninit<Entry<M, O, ARITY>>; SIZE],
}

/// An entry containing the key and value as well as the access frequency
#[must_use]
struct Entry<M: Manager, O, const ARITY: usize> {
    // Edges usually have an alignment of 8 bytes, and the operator should not
    // need more than 4 bytes. Therefore, we can store the frequency as u32
    // without needing extra space.
    freq: u32,
    operator: O,
    arity: u8,
    /// Invariant: The first `arity` operands are initialized
    operands: [MaybeUninit<M::Edge>; ARITY],
    value: M::Edge,
}

impl<M: Manager, O, const ARITY: usize> DropWith<M::Edge> for Entry<M, O, ARITY> {
    fn drop_with(self, drop_edge: impl Fn(M::Edge)) {
        for operand in &self.operands[..self.arity as usize] {
            // SAFETY: The first arity elements are initialized, we drop `self`
            drop_edge(unsafe { operand.assume_init_read() });
        }
        drop_edge(self.value);
    }
}

impl<M, O, H, const ARITY: usize, const BUCKET_SIZE: usize>
    LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
where
    M: Manager,
    O: Copy + Hash + Eq,
    H: Hasher + Default,
{
    const CHECKS: () = {
        assert!(
            0 < BUCKET_SIZE && BUCKET_SIZE <= u8::MAX as usize,
            "BUCKET_SIZE must be in range [1, 255]",
        );
        assert!(
            0 < ARITY && ARITY <= u8::MAX as usize,
            "ARITY must be in range [1, 255]",
        );
    };

    /// Create a new `ApplyCache` with the given capacity (entries).
    pub fn with_capacity(capacity: usize) -> Self {
        let _ = Self::CHECKS;
        let buckets = (capacity / BUCKET_SIZE)
            .checked_next_power_of_two()
            .expect("capacity is too large");
        #[cfg(not(feature = "hugealloc"))]
        let mut vec = Vec::with_capacity(buckets);
        #[cfg(feature = "hugealloc")]
        let mut vec = Vec::with_capacity_in(buckets, hugealloc::HugeAlloc);

        for _ in 0..buckets {
            vec.push(Mutex::new(Bucket::new()))
        }

        LFUApplyCache(vec.into_boxed_slice(), PhantomData)
    }

    /// Get the bucket for the given operator and operands
    #[inline]
    fn bucket(
        &self,
        operator: O,
        operands: &[Borrowed<M::Edge>],
    ) -> &Mutex<Bucket<M, O, ARITY, BUCKET_SIZE>> {
        let mut hasher = H::default();
        operator.hash(&mut hasher);
        for o in operands {
            o.hash(&mut hasher);
        }
        let mask = (self.0.len() - 1) as u64; // bucket count is a power of two
        let index = (hasher.finish() & mask) as usize;
        debug_assert!(index < self.0.len());
        // SAFETY: access is guaranteed to be in bounds
        unsafe { self.0.get_unchecked(index) }
    }
}

#[cfg(feature = "statistics")]
static STAT_ACCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "statistics")]
static STAT_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "statistics")]
static STAT_INSERTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl<M, O, H, const ARITY: usize, const BUCKET_SIZE: usize> crate::StatisticsGenerator
    for LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
where
    M: Manager,
    H: Hasher + Default,
{
    #[cfg(not(feature = "statistics"))]
    fn print_stats(&self) {}

    #[cfg(feature = "statistics")]
    fn print_stats(&self) {
        let count = self.0.len();
        debug_assert!(count > 0);
        let mut num_entries = Vec::with_capacity(count);
        let mut freq_min = Vec::with_capacity(count);
        let mut freq_max = Vec::with_capacity(count);

        for bucket in &*self.0 {
            let bucket = bucket.lock();
            let mut min = u32::MAX;
            let mut max = u32::MIN;
            for entry in &bucket.entries[0..bucket.num_entries as usize] {
                // SAFETY: all entries up to `bucket.num_entries` are initialized
                let entry = unsafe { entry.assume_init_ref() };
                min = std::cmp::min(min, entry.freq);
                max = std::cmp::max(max, entry.freq);
            }
            num_entries.push(bucket.num_entries);
            freq_min.push(min);
            freq_max.push(max);
        }

        let num_entries_sum = num_entries.iter().map(|&x| x as usize).sum::<usize>() as f32;
        let full_buckets = num_entries
            .iter()
            .filter(|&&n| n == BUCKET_SIZE as u8)
            .count();
        let empty_buckets = num_entries.iter().filter(|&&n| n == 0).count();
        let freq_max = freq_max.iter().copied().max().unwrap();

        let accesses = STAT_ACCESSES.swap(0, std::sync::atomic::Ordering::Relaxed);
        let hits = STAT_HITS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let insertions = STAT_INSERTIONS.swap(0, std::sync::atomic::Ordering::Relaxed);
        println!(
            "[LFUApplyCache] fill level: {:.2} %, empty/full buckets: {empty_buckets}/{full_buckets}, accesses: {accesses}, hits: {hits}, insertions: {insertions}, hit ratio ~{:.2} %, max freq: {freq_max}",
            100.0 * num_entries_sum / (count * BUCKET_SIZE) as f32,
            100.0 * hits as f32 / accesses as f32,
        );
    }
}

impl<M: Manager, O: Copy + Eq, const ARITY: usize, const SIZE: usize> Bucket<M, O, ARITY, SIZE> {
    /// Create a new `Bucket`
    #[inline]
    fn new() -> Self {
        Self {
            num_entries: 0,
            entries: [Entry::UNINIT; SIZE],
        }
    }

    /// Get a mutable reference to the entry for the given operator and operands
    /// without modifying the frequency counter.
    #[inline]
    fn entry_mut(
        &mut self,
        operator: O,
        operands: &[Borrowed<M::Edge>],
    ) -> Option<&mut Entry<M, O, ARITY>> {
        debug_assert!(operands.len() <= ARITY);
        let len = self.num_entries as usize;
        'outer: for entry in &mut self.entries[0..len] {
            // SAFETY: the first `self.num_entries` elements are initialized
            let entry = unsafe { entry.assume_init_mut() };
            if entry.operator == operator && entry.arity as usize == operands.len() {
                for (o1, o2) in operands.iter().zip(&entry.operands) {
                    // SAFETY: the first `entry.arity` == `operands.len()` operands are initialized
                    if &**o1 != unsafe { o2.assume_init_ref() } {
                        continue 'outer;
                    }
                }
                return Some(entry);
            }
        }
        None
    }

    /// Find the edge for the given operator and operands.
    ///
    /// We return an `Edge` and not a `&E::Weak` because of lifetime
    /// restrictions.
    #[inline]
    fn find_edge(
        &mut self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
    ) -> Option<M::Edge> {
        #[cfg(feature = "statistics")]
        STAT_ACCESSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let entry = self.entry_mut(operator, operands)?;

        #[cfg(feature = "statistics")]
        STAT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let res = Some(manager.clone_edge(&entry.value));
        entry.freq += 1;
        if entry.freq == u32::MAX {
            // TODO: is `u32::MAX` too large?
            self.div_freqs();
        }
        res
    }

    /// Divide all frequencies in case they got too large (to be checked by the
    /// caller)
    #[cold]
    fn div_freqs(&mut self) {
        #[cfg(feature = "statistics")]
        println!("[LFUApplyCache] div_freqs({self:p})");

        let len = self.num_entries as usize;
        for entry in &mut self.entries[0..len] {
            // SAFETY: the first `self.num_entries` elements are initialized
            let entry = unsafe { entry.assume_init_mut() };
            entry.freq >>= 1; // TODO: which amount is best?
        }
    }

    /// Insert a key-value pair into this bucket
    ///
    /// Does not check for duplicate entries.
    #[inline]
    fn insert(
        &mut self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        value: Borrowed<M::Edge>,
    ) {
        debug_assert!(operands.len() <= ARITY);
        #[cfg(feature = "statistics")]
        STAT_INSERTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let len = self.num_entries as usize;
        let mut new = Entry {
            freq: 0,
            operator,
            arity: operands.len() as u8,
            operands: [Entry::<M, O, ARITY>::UNINIT_OPERAND; ARITY],
            value: manager.clone_edge(&*value),
        };
        for (src, dst) in operands.iter().zip(&mut new.operands) {
            dst.write(manager.clone_edge(src));
        }
        if len < SIZE {
            self.entries[len] = MaybeUninit::new(new);
            self.num_entries += 1;
            return;
        }
        let entry = self.entries.iter_mut().min_by_key(|e| {
            // SAFETY: the first `self.num_entries` elements are initialized and
            // `num_entries >= SIZE`.
            let entry = unsafe { e.assume_init_ref() };
            entry.freq
        });
        // SAFETY: the minimal element is initialized
        std::mem::replace(unsafe { entry.unwrap().assume_init_mut() }, new)
            .drop_with_manager(manager);
    }

    /// Perform garbage collection on this bucket
    ///
    /// We allow different `should_gc` implementations for testing purposes.
    fn gc(&mut self, manager: &M, should_gc: impl Fn(&Entry<M, O, ARITY>, &M) -> bool) {
        let mut len = self.num_entries as usize;
        let mut i = 0;
        while i < len {
            let entry = &mut self.entries[i];
            // SAFETY: the first `len` elements are initialized
            if should_gc(unsafe { entry.assume_init_ref() }, manager) {
                // SAFETY: the first `len` elements are initialized
                unsafe { entry.assume_init_read() }.drop_with_manager(manager);
                len -= 1;
                if i >= len {
                    break; // dropped the last entry
                }
                // Move the last entry to the current index (i.e. `entry`). We
                // cannot simply write `entry` below because of the borrow
                // checker.
                // SAFETY: the first `len + 1` elements except the `i`th element
                // are initialized, and `len != i`.
                self.entries[i].write(unsafe { self.entries[len].assume_init_read() });
            } else {
                i += 1;
            }
        }
        self.num_entries = len as _;
    }

    /// Remove all entries from the bucket
    #[inline]
    fn clear_with(&mut self, drop_edge: impl Fn(M::Edge)) {
        let len = self.num_entries as usize;
        for entry in &mut self.entries[0..len] {
            // SAFETY: all entries up to `bucket.num_entries` are initialized
            unsafe { entry.assume_init_read() }.drop_with(&drop_edge);
        }
        self.num_entries = 0;
    }

    /// Remove all entries from the bucket
    #[inline]
    fn clear(&mut self, manager: &M) {
        self.clear_with(|e| manager.drop_edge(e));
    }
}

impl<M: Manager, O, const ARITY: usize, const SIZE: usize> Drop for Bucket<M, O, ARITY, SIZE> {
    fn drop(&mut self) {
        debug_assert_eq!(self.num_entries, 0, "Must clear bucket before dropping it");
    }
}

impl<M: Manager, O, const ARITY: usize> Entry<M, O, ARITY> {
    const UNINIT: MaybeUninit<Entry<M, O, ARITY>> = MaybeUninit::uninit();
    const UNINIT_OPERAND: MaybeUninit<M::Edge> = MaybeUninit::uninit();

    /// Returns whether garbage collection should remove this entry
    ///
    /// The current implementation checks whether one of the operand's (main)
    /// reference counts is 0.
    fn should_gc(&self, manager: &M) -> bool {
        //if E::from_weak(&self.value).ref_counts().0 == 0 {
        //    return true;
        //}
        // SAFETY: The first `arity` operands are initialized
        self.operands[..self.arity as usize]
            .iter()
            .any(|o| manager.get_node(unsafe { o.assume_init_ref() }).ref_count() == 0)
    }
}

impl<M, O, H, const ARITY: usize, const BUCKET_SIZE: usize> GCContainer<M>
    for LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
where
    M: Manager,
    O: Copy + Hash + Ord,
    H: Hasher + Default,
{
    fn pre_gc(&self, manager: &M) {
        for bucket in self.0.iter() {
            if let Some(mut bucket) = bucket.try_lock() {
                bucket.gc(manager, Entry::should_gc);
            }
        }
    }

    unsafe fn post_gc(&self, _manager: &M) {
        // Nothing to do
    }
}

impl<M, O, H, const ARITY: usize, const BUCKET_SIZE: usize> ApplyCache<M, O>
    for LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
where
    M: Manager,
    O: Copy + Ord + Hash,
    H: Hasher + Default,
{
    #[inline]
    fn get(&self, manager: &M, operator: O, operands: &[Borrowed<M::Edge>]) -> Option<M::Edge> {
        if operands.len() > ARITY {
            return None;
        }
        self.bucket(operator, operands)
            .try_lock()?
            .find_edge(manager, operator, operands)
    }

    #[inline]
    fn add(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        value: Borrowed<M::Edge>,
    ) {
        if operands.len() > ARITY {
            return;
        }
        if let Some(mut bucket) = self.bucket(operator, operands).try_lock() {
            if bucket.entry_mut(operator, operands).is_none() {
                bucket.insert(manager, operator, operands, value);
            }
        }
    }

    fn clear(&self, manager: &M) {
        self.0
            .iter()
            .for_each(|bucket| bucket.lock().clear(manager));
    }
}

impl<M, O, H, const ARITY: usize, const BUCKET_SIZE: usize> fmt::Debug
    for LFUApplyCache<M, O, H, ARITY, BUCKET_SIZE>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.0
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, b.lock()))
                    .filter(|(_, b)| b.num_entries > 0),
            )
            .finish()
    }
}

impl<M, O, const ARITY: usize, const SIZE: usize> fmt::Debug for Bucket<M, O, ARITY, SIZE>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = &self.entries[..self.num_entries as usize];
        // SAFETY: All elements of `entries` (the first `self.num_entries` of
        // `self.entries`) are initialized.
        f.debug_list()
            .entries(entries.iter().map(|e| unsafe { e.assume_init_ref() }))
            .finish()
    }
}

impl<M, O, const ARITY: usize> fmt::Debug for Entry<M, O, ARITY>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_assert_ne!(self.arity, 0);
        // SAFETY: The first `arity` operands are initialized
        write!(f, "{{{{ {:?}({:?}", self.operator, unsafe {
            self.operands[0].assume_init_ref()
        })?;
        for operand in &self.operands[1..self.arity as usize] {
            // SAFETY: The first `arity` operands are initialized
            write!(f, ", {:?}", unsafe { operand.assume_init_ref() })?;
        }
        write!(f, ") = {:?}; freq: {} }}}}", &self.value, self.freq)
    }
}

#[cfg(test)]
mod test {
    use oxidd_core::Edge;
    use oxidd_core::Manager;
    use oxidd_test_utils::assert_ref_counts;
    use oxidd_test_utils::edge::{DummyEdge, DummyManager};

    use super::Bucket;

    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct Op;

    fn drop_edges(it: impl IntoIterator<Item = DummyEdge>) {
        for edge in it {
            DummyManager.drop_edge(edge);
        }
    }

    #[test]
    fn bucket_repeated_ins() {
        let mut bucket: Bucket<DummyManager, Op, 2, 4> = Bucket::new();
        let e1 = DummyEdge::new();
        let e2 = DummyEdge::new();
        let e3 = DummyEdge::new();

        // `insert()` does not check for duplicates, but the first entry in the
        // bucket should be returned. (Note that first entry does not
        // necessarily mean the oldest, but here this is the case.)
        assert_eq!(bucket.num_entries, 0);
        bucket.insert(&DummyManager, Op, &[e1.borrowed()], e2.borrowed());
        assert_eq!(bucket.num_entries, 1);
        bucket.insert(&DummyManager, Op, &[e1.borrowed()], e3.borrowed());
        assert_eq!(bucket.num_entries, 2);
        let v = &bucket.entry_mut(Op, &[e1.borrowed()]).unwrap().value;
        assert_ref_counts!(e1 = 3; e2 = 2; e3 = 2);
        assert_eq!(v, &e2);

        bucket.clear(&DummyManager);
        assert_ref_counts!(e1 = 1; e2 = 1; e3 = 1);
        assert_eq!(bucket.num_entries, 0);
        bucket.insert(&DummyManager, Op, &[e1.borrowed()], e2.borrowed());
        assert_eq!(bucket.num_entries, 1);

        bucket.clear(&DummyManager);
        drop_edges([e1, e2, e3]);
    }

    #[test]
    fn bucket_ins_full() {
        let mut bucket: Bucket<DummyManager, Op, 1, 2> = Bucket::new();
        let (l1, v1) = (DummyEdge::new(), DummyEdge::new());
        let (l2, v2) = (DummyEdge::new(), DummyEdge::new());
        let (l3, v3) = (DummyEdge::new(), DummyEdge::new());

        bucket.insert(&DummyManager, Op, &[l1.borrowed()], v1.borrowed());
        bucket.insert(&DummyManager, Op, &[l2.borrowed()], v2.borrowed());
        // `entry_mut()` does not modify the freq counter
        let f1 = bucket.entry_mut(Op, &[l1.borrowed()]).unwrap().freq;
        let f2 = bucket.entry_mut(Op, &[l2.borrowed()]).unwrap().freq;
        assert_eq!(f1, 0);
        assert_eq!(f2, 0);
        DummyManager.drop_edge(
            bucket
                .find_edge(&DummyManager, Op, &[l2.borrowed()])
                .unwrap(),
        );
        let f2 = bucket.entry_mut(Op, &[l2.borrowed()]).unwrap().freq;
        assert_eq!(f2, 1);

        // least-frequently used replacement strategy
        bucket.insert(&DummyManager, Op, &[l3.borrowed()], v3.borrowed());
        assert_ref_counts!(l1, v1 = 1; l2, v2 = 2; l3, v3 = 2);

        let f3 = bucket.entry_mut(Op, &[l3.borrowed()]).unwrap().freq;
        assert_eq!(f3, 0);

        bucket.insert(&DummyManager, Op, &[l1.borrowed()], v1.borrowed());
        assert_ref_counts!(l1, v1 = 2; l2, v2 = 2; l3, v3 = 1);

        bucket.clear(&DummyManager);
        drop_edges([l1, l2, l3, v1, v2, v3]);
    }

    #[test]
    fn bucket_gc() {
        let mut bucket: Bucket<DummyManager, Op, 1, 2> = Bucket::new();
        let (l1, v1) = (DummyEdge::new(), DummyEdge::new());
        let (l2, v2) = (DummyEdge::new(), DummyEdge::new());

        bucket.insert(&DummyManager, Op, &[l1.borrowed()], v1.borrowed());
        bucket.insert(&DummyManager, Op, &[l2.borrowed()], v2.borrowed());
        assert_ref_counts!(l1, v1 = 2; l2, v2 = 2);

        bucket.gc(&DummyManager, |entry, _| entry.value == v2);
        assert_ref_counts!(l1, v1 = 2; l2, v2 = 1);

        bucket.insert(&DummyManager, Op, &[l2.borrowed()], v2.borrowed());
        assert_ref_counts!(l1, v1 = 2; l2, v2 = 2);

        bucket.gc(&DummyManager, |entry, _| entry.value == v1);
        assert_ref_counts!(l1, v1 = 1; l2, v2 = 2);

        bucket.insert(&DummyManager, Op, &[l1.borrowed()], v1.borrowed());
        assert_ref_counts!(l1, v1 = 2; l2, v2 = 2);

        bucket.gc(&DummyManager, |_, _| true);
        assert_ref_counts!(l1, v1 = 1; l2, v2 = 1);

        drop_edges([l1, l2, v1, v2]);
    }
}
