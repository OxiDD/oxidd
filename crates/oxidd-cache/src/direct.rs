//! Fixed-size direct mapped apply cache

use std::cell::UnsafeCell;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;

use oxidd_core::util::GCContainer;
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

union Operand<E> {
    edge: ManuallyDrop<E>,
    numeric: u32,
    uninit: (),
}

impl<E> Operand<E> {
    const UNINIT: Self = Self { uninit: () };

    /// SAFETY: `self` must be initialized as `edge`
    unsafe fn assume_edge_ref(&self) -> &E {
        // SAFETY: see above
        unsafe { &self.edge }
    }

    /// SAFETY: `self` must be initialized as `numeric`
    unsafe fn assume_numeric(&self) -> u32 {
        // SAFETY: see above
        unsafe { self.numeric }
    }

    fn write_edge(&mut self, edge: Borrowed<E>) {
        // SAFETY: The referenced node lives at least until the next garbage
        // collection / reordering. Before this operation garbage
        // collection, we clear the entire cache.
        self.edge = unsafe { Borrowed::into_inner(edge) };
    }

    fn write_numeric(&mut self, num: u32) {
        self.numeric = num;
    }
}

/// Entry containing key and value
struct Entry<M: Manager, O, const ARITY: usize> {
    /// Mutex for all the `UnsafeCell`s in here
    mutex: crate::util::RawMutex,
    /// Count of edge operands. If 0, this entry is not occupied.
    edge_operands: UnsafeCell<u8>,
    /// Count of numeric operands
    numeric_operands: UnsafeCell<u8>,
    /// Operator of the key. Initialized if `edge_operands != 0`.
    operator: UnsafeCell<MaybeUninit<O>>,
    /// Operands of the key. The first `edge_operands` elements are edges, the
    /// following `numeric_operands` are numeric.
    operands: UnsafeCell<[Operand<M::Edge>; ARITY]>,
    /// Initialized if `arity != 0`
    value: UnsafeCell<MaybeUninit<M::Edge>>,
}

struct EntryGuard<'a, M: Manager, O, const ARITY: usize>(&'a Entry<M, O, ARITY>);

// SAFETY: `Entry` is like a `Mutex`
unsafe impl<M: Manager, O: Send, const ARITY: usize> Send for Entry<M, O, ARITY> where M::Edge: Send {}
unsafe impl<M: Manager, O: Send, const ARITY: usize> Sync for Entry<M, O, ARITY> where M::Edge: Send {}

impl<M: Manager, O: Copy + Eq, const ARITY: usize> Entry<M, O, ARITY> {
    #[inline]
    fn new() -> Self {
        Self {
            mutex: crate::util::RawMutex::INIT,
            operator: UnsafeCell::new(MaybeUninit::uninit()),
            edge_operands: UnsafeCell::new(0),
            numeric_operands: UnsafeCell::new(0),
            operands: UnsafeCell::new([Operand::UNINIT; ARITY]),
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

impl<M: Manager, O, const ARITY: usize> Drop for EntryGuard<'_, M, O, ARITY> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: The entry is locked.
        unsafe { self.0.mutex.unlock() }
    }
}

impl<M: Manager, O, const ARITY: usize> EntryGuard<'_, M, O, ARITY>
where
    O: Copy + Eq,
{
    /// Is this entry occupied?
    #[inline]
    fn is_occupied(&self) -> bool {
        // SAFETY: The entry is locked.
        unsafe { *self.0.edge_operands.get() != 0 }
    }

    /// Get the value of this entry if it is occupied and the key (`operator`
    /// and `operands`) matches
    #[inline]
    fn get(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        numeric_operands: &[u32],
    ) -> Option<M::Edge> {
        debug_assert_ne!(operands.len(), 0);

        #[cfg(feature = "statistics")]
        STAT_ACCESSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let num_edge_operands = operands.len();
        let num_numeric_operands = numeric_operands.len();

        // SAFETY: The entry is locked.
        if num_edge_operands != unsafe { *self.0.edge_operands.get() } as usize
            || numeric_operands.len() != unsafe { *self.0.numeric_operands.get() } as usize
        {
            return None;
        }
        // SAFETY: The entry is locked and occupied.
        if operator != unsafe { (*self.0.operator.get()).assume_init() } {
            return None;
        }
        // SAFETY: The entry is locked.
        let (entry_operands, remaining) =
            unsafe { &*self.0.operands.get() }.split_at(num_edge_operands);
        let entry_numeric_operands = &remaining[..num_numeric_operands];

        for (o1, o2) in operands.iter().zip(entry_operands) {
            // SAFETY: The first `num_edge_operands` operands are edges
            if &**o1 != unsafe { o2.assume_edge_ref() } {
                return None;
            }
        }
        for (o1, o2) in numeric_operands.iter().zip(entry_numeric_operands) {
            // SAFETY: The operands in range
            // `num_edge_operands..num_edge_operands + num_numeric_operands`
            // operands are numeric
            if *o1 != unsafe { o2.assume_numeric() } {
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
    ///
    /// Assumes that `operands.len() + numeric_operands.len() <= ARITY`
    #[inline]
    fn set(
        &mut self,
        _manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        numeric_operands: &[u32],
        value: Borrowed<M::Edge>,
    ) {
        debug_assert_ne!(operands.len(), 0);
        self.clear();

        #[cfg(feature = "statistics")]
        STAT_INSERTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let num_edge_operands = operands.len();
        let num_numeric_operands = numeric_operands.len();

        // SAFETY (next 2 `.get()` calls): The entry is locked.
        unsafe { &mut *self.0.operator.get() }.write(operator);
        let (entry_operands, remaining) =
            unsafe { &mut *self.0.operands.get() }.split_at_mut(num_edge_operands);
        let entry_numeric_operands = &mut remaining[..num_numeric_operands];

        for (src, dst) in operands.iter().zip(entry_operands) {
            dst.write_edge(src.borrowed());
        }
        for (src, dst) in numeric_operands.iter().zip(entry_numeric_operands) {
            dst.write_numeric(*src);
        }

        // SAFETY: The referenced node lives at least until the next garbage
        // collection / reordering. Before this operation garbage
        // collection, we clear the entire cache.
        let value = unsafe { Borrowed::into_inner(value) };
        // SAFETY (next 3 `.get()` calls): The entry is locked.
        unsafe { &mut *self.0.value.get() }.write(ManuallyDrop::into_inner(value));
        // Important: Set the arity last for exception safety (the functions above might
        // panic).
        unsafe { *self.0.edge_operands.get() = num_edge_operands as u8 };
        unsafe { *self.0.numeric_operands.get() = num_numeric_operands as u8 };
    }

    /// Clear this entry
    #[inline(always)]
    fn clear(&mut self) {
        // SAFETY: The entry is locked.
        unsafe { *self.0.edge_operands.get() = 0 };
        // `Edge`s are just borrowed, so nothing else to do.
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
    /// # Safety
    ///
    /// The apply cache must only be used inside a manager that guarantees all
    /// node deletions to be wrapped inside an [`GCContainer::pre_gc()`] /
    /// [`GCContainer::post_gc()`] pair.
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
    fn get_with_numeric(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        numeric_operands: &[u32],
    ) -> Option<M::Edge> {
        if operands.is_empty() || operands.len() + numeric_operands.len() > ARITY {
            return None;
        }
        self.bucket(operator, operands).try_lock()?.get(
            manager,
            operator,
            operands,
            numeric_operands,
        )
    }

    #[inline(always)]
    fn add_with_numeric(
        &self,
        manager: &M,
        operator: O,
        operands: &[Borrowed<M::Edge>],
        numeric_operands: &[u32],
        value: Borrowed<M::Edge>,
    ) {
        if operands.is_empty() || operands.len() + numeric_operands.len() > ARITY {
            return;
        }
        if let Some(mut entry) = self.bucket(operator, operands).try_lock() {
            entry.set(manager, operator, operands, numeric_operands, value);
        }
    }

    fn clear(&self, _manager: &M) {
        for entry in &*self.0 {
            entry.lock().clear();
        }
    }
}

impl<M, O, H, const ARITY: usize> GCContainer<M> for DMApplyCache<M, O, H, ARITY>
where
    M: Manager,
    O: Copy + Hash + Ord,
    H: Hasher + Default,
{
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

impl<M, O, const ARITY: usize> fmt::Debug for EntryGuard<'_, M, O, ARITY>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The entry is locked.
        let edge_operands = unsafe { *self.0.edge_operands.get() } as usize;
        if edge_operands == 0 {
            write!(f, "None")
        } else {
            // SAFETY: The entry is locked and occupied.
            let operator = unsafe { (*self.0.operator.get()).assume_init() };
            // SAFETY: The entry is locked.
            let operands = unsafe { &(*self.0.operands.get())[..edge_operands] };
            // SAFETY: The first `arity` (> 0) operands are initialized.
            write!(f, "{{{{ {operator:?}({:?}", unsafe {
                operands[0].assume_edge_ref()
            })?;
            for operand in &operands[1..] {
                // SAFETY: The first `arity` operands are initialized.
                write!(f, ", {:?}", unsafe { operand.assume_edge_ref() })?;
            }
            // SAFETY: The entry is locked and occupied.
            let value = unsafe { (*self.0.value.get()).assume_init_ref() };
            write!(f, ") = {value:?} }}}}")
        }
    }
}
