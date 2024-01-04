//! Fixed-size direct mapped apply cache

use std::cell::UnsafeCell;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;

use parking_lot::lock_api::RawMutex;

use oxidd_core::util::Borrowed;
use oxidd_core::util::DropWith;
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::Manager;

#[cfg(feature = "hugealloc")]
type Box<T> = allocator_api2::boxed::Box<T, hugealloc::HugeAlloc>;
#[cfg(feature = "hugealloc")]
type Vec<T> = allocator_api2::vec::Vec<T, hugealloc::HugeAlloc>;

/// Fixed-size direct mapped apply cache
pub struct DMApplyCache<M: Manager, O, H, const ARITY: usize = 2>(
    Box<[Entry<M, O, ARITY>]>,
    PhantomData<H>,
);

impl<M: Manager, O, H, const ARITY: usize> DropWith<M::Edge> for DMApplyCache<M, O, H, ARITY>
where
    O: Copy + Eq + Hash,
{
    fn drop_with(self, _drop_edge: impl Fn(M::Edge)) {
        // The plain drop impl suffices
    }
}

/// Entry containing key and value
struct Entry<M: Manager, O, const ARITY: usize> {
    /// Mutex for all the `UnsafeCell`s in here
    mutex: crate::util::RawMutex,
    /// Arity of the operator. If 0, this entry is not occupied.
    arity: UnsafeCell<u8>,
    /// Operator of the key. Initialized if `arity != 0`.
    operator: UnsafeCell<MaybeUninit<O>>,
    /// Oprands of the key. The first `arity` elements are initialized.
    operands: UnsafeCell<[MaybeUninit<M::Edge>; ARITY]>,
    /// Initialized if `arity != 0`
    value: UnsafeCell<MaybeUninit<M::Edge>>,
}

struct EntryGuard<'a, M: Manager, O, const ARITY: usize>(&'a Entry<M, O, ARITY>);

// SAFETY: `Entry` is like a `Mutex`
unsafe impl<M: Manager, O: Send, const ARITY: usize> Send for Entry<M, O, ARITY> where M::Edge: Send {}
unsafe impl<M: Manager, O: Send, const ARITY: usize> Sync for Entry<M, O, ARITY> where M::Edge: Send {}

impl<M: Manager, O: Copy + Eq, const ARITY: usize> Entry<M, O, ARITY> {
    const UNINIT_OPERAND: MaybeUninit<M::Edge> = MaybeUninit::uninit();

    #[inline]
    fn new() -> Self {
        Self {
            mutex: crate::util::RawMutex::INIT,
            operator: UnsafeCell::new(MaybeUninit::uninit()),
            arity: UnsafeCell::new(0),
            operands: UnsafeCell::new([Self::UNINIT_OPERAND; ARITY]),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    #[inline]
    fn lock(&self) -> EntryGuard<M, O, ARITY> {
        self.mutex.lock();
        EntryGuard(self)
    }

    #[inline]
    fn try_lock(&self) -> Option<EntryGuard<M, O, ARITY>> {
        if self.mutex.try_lock() {
            Some(EntryGuard(self))
        } else {
            None
        }
    }
}

impl<'a, M: Manager, O, const ARITY: usize> Drop for EntryGuard<'a, M, O, ARITY> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: The entry is locked.
        unsafe { self.0.mutex.unlock() }
    }
}

impl<'a, M: Manager, O, const ARITY: usize> EntryGuard<'a, M, O, ARITY>
where
    O: Copy + Eq,
{
    /// Is this entry occupied?
    #[inline]
    fn is_occupied(&self) -> bool {
        // SAFETY: The entry is locked.
        unsafe { *self.0.arity.get() != 0 }
    }

    /// Get the value of this entry if it is occupied and the key (`operator`
    /// and `operands`) matches
    #[inline]
    fn get(&self, manager: &M, operator: O, operands: &[Borrowed<M::Edge>]) -> Option<M::Edge> {
        debug_assert_ne!(operands.len(), 0);

        #[cfg(feature = "statistics")]
        STAT_ACCESSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // SAFETY: The entry is locked.
        if operands.len() != unsafe { *self.0.arity.get() } as usize {
            return None;
        }
        // SAFETY: The entry is locked and occupied.
        if operator != unsafe { (*self.0.operator.get()).assume_init() } {
            return None;
        }
        // SAFETY: The entry is locked.
        for (o1, o2) in operands.iter().zip(unsafe { &*self.0.operands.get() }) {
            // SAFETY: The first arity = `operands.len()` operands are initialized
            let o2 = unsafe { o2.assume_init_ref() };
            if &**o1 != o2 {
                return None;
            }
        }

        #[cfg(feature = "statistics")]
        STAT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // SAFETY: The entry is locked and occupied.
        Some(manager.clone_edge(unsafe { (*self.0.value.get()).assume_init_ref() }))
    }

    /// Set the key/value of this entry
    ///
    /// If `self` is already occupied and the key matches, the entry is not
    /// updated (`operands` and `value` are not cloned).
    #[inline]
    fn set(
        &mut self,
        _manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        value: Borrowed<M::Edge>,
    ) {
        debug_assert_ne!(operands.len(), 0);
        self.clear();

        #[cfg(feature = "statistics")]
        STAT_INSERTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let arity = operands.len();
        // SAFETY (next 4 `.get()` calls): The entry is locked.
        unsafe { &mut *self.0.operator.get() }.write(operator);
        for (src, dst) in operands.iter().zip(unsafe { &mut *self.0.operands.get() }) {
            // SAFETY: The referenced node lives at least until the next garbage
            // collection / reordering. Before this operation garbage
            // collection, we clear the entire cache.
            dst.write(ManuallyDrop::into_inner(unsafe {
                Borrowed::into_inner(src.borrowed())
            }));
        }
        // SAFETY: as above
        let value = unsafe { Borrowed::into_inner(value) };
        unsafe { &mut *self.0.value.get() }.write(ManuallyDrop::into_inner(value));
        // Important: Set the arity last for exception safety (the functions above might panic).
        unsafe { *self.0.arity.get() = arity as u8 };
    }

    /// Clear this entry
    #[inline(always)]
    fn clear(&mut self) {
        // SAFETY: The entry is locked.
        unsafe { *self.0.arity.get() = 0 };
        // `Edge`s are just borrowed, so nothing to do.
    }
}

impl<M, O, H, const ARITY: usize> DMApplyCache<M, O, H, ARITY>
where
    M: Manager,
    O: Copy + Eq + Hash,
    H: Hasher + Default,
{
    const CHECK_ARITY: () = {
        assert!(
            0 < ARITY && ARITY <= u8::MAX as usize,
            "ARITY must be in range [1, 255]"
        );
    };

    /// Create a new `ApplyCache` with the given capacity (entries).
    ///
    /// SAFETY: The apply cache must only be used inside a manager that
    /// guarantees all node deletions to be wrapped inside an
    /// [`ApplyCache::pre_gc()`] / [`ApplyCache::post_gc()`] pair.
    pub unsafe fn with_capacity(capacity: usize) -> Self {
        let _ = Self::CHECK_ARITY;
        let buckets = capacity
            .checked_next_power_of_two()
            .expect("capacity is too large");
        #[cfg(not(feature = "hugealloc"))]
        let mut vec = Vec::with_capacity(buckets);
        #[cfg(feature = "hugealloc")]
        let mut vec = Vec::with_capacity_in(buckets, hugealloc::HugeAlloc);

        for _ in 0..buckets {
            vec.push(Entry::new())
        }

        DMApplyCache(vec.into_boxed_slice(), PhantomData)
    }

    /// Get the bucket for the given operator and operands
    #[inline]
    fn bucket(&self, operator: O, operands: &[Borrowed<M::Edge>]) -> &Entry<M, O, ARITY> {
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

impl<M, O, H, const ARITY: usize> crate::StatisticsGenerator for DMApplyCache<M, O, H, ARITY>
where
    M: Manager,
    O: Copy + Eq,
{
    #[cfg(not(feature = "statistics"))]
    fn print_stats(&self) {}

    #[cfg(feature = "statistics")]
    fn print_stats(&self) {
        let count = self.0.len();
        debug_assert!(count > 0);
        let occupied = self.0.iter().filter(|e| e.lock().is_occupied()).count();

        let accesses = STAT_ACCESSES.swap(0, std::sync::atomic::Ordering::Relaxed);
        let hits = STAT_HITS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let insertions = STAT_INSERTIONS.swap(0, std::sync::atomic::Ordering::Relaxed);
        println!(
            "[DMApplyCache] fill level: {:.2} %, accesses: {accesses}, hits: {hits}, insertions: {insertions}, hit ratio ~{:.2} %",
            100.0 * occupied as f32 / count as f32,
            100.0 * hits as f32 / accesses as f32,
        );
    }
}

impl<M, O, H, const ARITY: usize> ApplyCache<M, O> for DMApplyCache<M, O, H, ARITY>
where
    M: Manager,
    O: Copy + Hash + Ord,
    H: Hasher + Default,
{
    #[inline(always)]
    fn get(&self, manager: &M, operator: O, operands: &[Borrowed<M::Edge>]) -> Option<M::Edge> {
        if operands.len() == 0 || operands.len() > ARITY {
            return None;
        }
        self.bucket(operator, operands)
            .try_lock()?
            .get(manager, operator, operands)
    }

    #[inline(always)]
    fn add(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        value: Borrowed<M::Edge>,
    ) {
        if operands.len() == 0 || operands.len() > ARITY {
            return;
        }
        if let Some(mut entry) = self.bucket(operator, operands).try_lock() {
            entry.set(manager, operator, operands, value);
        }
    }

    fn clear(&self, _manager: &M) {
        for entry in &*self.0 {
            entry.lock().clear();
        }
    }

    fn pre_gc(&self, _manager: &M) {
        // FIXME: We should probably do something smarter than clearing the
        // entire cache.
        for entry in &*self.0 {
            let mut entry = entry.lock();
            entry.clear();
            // Don't unlock!
            std::mem::forget(entry);
        }
    }

    unsafe fn post_gc(&self, _manager: &M) {
        for entry in &*self.0 {
            // SAFETY: `post_gc()` is called at most once after `pre_gc()` and
            // reordering. Hence, the mutex is locked. The cache is empty, so
            // we don't risk that a call to `get()` returns an invalid edge.
            unsafe { entry.mutex.unlock() };
        }
    }
}

impl<M, O, H, const ARITY: usize> fmt::Debug for DMApplyCache<M, O, H, ARITY>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.0
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (i, e.lock()))
                    .filter(|(_, e)| e.is_occupied()),
            )
            .finish()
    }
}

impl<'a, M, O, const ARITY: usize> fmt::Debug for EntryGuard<'a, M, O, ARITY>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The entry is locked.
        let arity = unsafe { *self.0.arity.get() } as usize;
        if arity == 0 {
            write!(f, "None")
        } else {
            // SAFETY: The entry is locked and occupied.
            let operator = unsafe { (*self.0.operator.get()).assume_init() };
            // SAFETY: The entry is locked.
            let operands = unsafe { &(*self.0.operands.get())[..arity] };
            // SAFETY: The first `arity` (> 0) operands are initialized.
            write!(f, "{{{{ {operator:?}({:?}", unsafe {
                operands[0].assume_init_ref()
            })?;
            for operand in &operands[1..] {
                // SAFETY: The first `arity` operands are initialized.
                write!(f, ", {:?}", unsafe { operand.assume_init_ref() })?;
            }
            // SAFETY: The entry is locked and occupied.
            let value = unsafe { (*self.0.value.get()).assume_init_ref() };
            write!(f, ") = {value:?} }}}}")
        }
    }
}
