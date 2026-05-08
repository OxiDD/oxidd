//! Fixed-size direct mapped apply cache

use std::cell::UnsafeCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};

use parking_lot::lock_api::RawMutex;

use oxidd_core::util::{Borrowed, DropWith};
use oxidd_core::{ApplyCache, Edge, Manager, ManagerEventSubscriber};

#[cfg(feature = "hugealloc")]
type Box<T> = allocator_api2::boxed::Box<T, hugealloc::HugeAlloc>;
#[cfg(feature = "hugealloc")]
type Vec<T> = allocator_api2::vec::Vec<T, hugealloc::HugeAlloc>;

/// Fixed-size direct mapped apply cache
pub struct DMApplyCache<M: Manager, O, H, const ENTRY_CAP: usize = 3>(
    Box<[Entry<M, O, ENTRY_CAP>]>,
    PhantomData<H>,
);

impl<M: Manager, O, H, const ENTRY_CAP: usize> DropWith<M::Edge>
    for DMApplyCache<M, O, H, ENTRY_CAP>
where
    O: Copy + Eq + Hash,
{
    fn drop_with(self, _drop_edge: impl Fn(M::Edge)) {
        // The plain drop impl suffices
    }
}

union Datum<E> {
    edge: ManuallyDrop<E>,
    numeric: u32,
    uninit: (),
}

impl<E> Datum<E> {
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
        // collection / reordering. Before that operation, we clear the entire
        // cache.
        self.edge = unsafe { Borrowed::into_inner(edge) };
    }

    fn write_numeric(&mut self, num: u32) {
        self.numeric = num;
    }
}

const KIND_BITS: u32 = 4;
const KIND_COUNT: usize = 1 << KIND_BITS;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CountPair(u8);

impl CountPair {
    const NULL: Self = CountPair(0);

    const fn new(edge: usize, numeric: usize) -> Self {
        debug_assert!(edge < KIND_COUNT);
        debug_assert!(numeric < KIND_COUNT);
        Self((numeric << 4 | edge) as u8)
    }

    const fn edge(self) -> usize {
        self.0 as usize & (KIND_COUNT - 1)
    }

    const fn numeric(self) -> usize {
        (self.0 >> KIND_BITS) as usize
    }
}

/// Entry containing key and value
struct Entry<M: Manager, O, const ENTRY_CAP: usize> {
    /// Mutex for all the `UnsafeCell`s in here
    mutex: crate::util::RawMutex,
    /// Count of operands. If 0, this entry is not occupied.
    operands: UnsafeCell<CountPair>,
    /// Count of values
    values: UnsafeCell<CountPair>,
    /// Operator of the key. Initialized if `operands != 0`.
    operator: UnsafeCell<MaybeUninit<O>>,
    /// Operands and values. The first `operands.edge()` elements are edges, the
    /// following `operands.numeric()` elements are numeric. Then follow
    /// `values.edge()` edges and `values.numeric()` numeric elements.
    data: UnsafeCell<[Datum<M::Edge>; ENTRY_CAP]>,
}

struct EntryGuard<'a, M: Manager, O, const ENTRY_CAP: usize>(&'a Entry<M, O, ENTRY_CAP>);

// SAFETY: `Entry` is like a `Mutex`
unsafe impl<M: Manager, O: Send, const ENTRY_CAP: usize> Send for Entry<M, O, ENTRY_CAP> where
    M::Edge: Send
{
}
unsafe impl<M: Manager, O: Send, const ENTRY_CAP: usize> Sync for Entry<M, O, ENTRY_CAP> where
    M::Edge: Send
{
}

impl<M: Manager, O: Copy + Eq, const ENTRY_CAP: usize> Entry<M, O, ENTRY_CAP> {
    // Regarding the lint: the intent here is not to modify the `AtomicBool` in
    // a const context but to create the `RawMutex` in a const context.
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self {
        mutex: crate::util::RawMutex::INIT,
        operands: UnsafeCell::new(CountPair::NULL),
        values: UnsafeCell::new(CountPair::NULL),
        operator: UnsafeCell::new(MaybeUninit::uninit()),
        data: UnsafeCell::new([Datum::UNINIT; ENTRY_CAP]),
    };

    #[inline]
    fn lock(&self) -> EntryGuard<'_, M, O, ENTRY_CAP> {
        self.mutex.lock();
        EntryGuard(self)
    }

    #[inline]
    fn try_lock(&self) -> Option<EntryGuard<'_, M, O, ENTRY_CAP>> {
        if self.mutex.try_lock() {
            Some(EntryGuard(self))
        } else {
            None
        }
    }
}

impl<M: Manager, O, const ENTRY_CAP: usize> Drop for EntryGuard<'_, M, O, ENTRY_CAP> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: The entry is locked.
        unsafe { self.0.mutex.unlock() }
    }
}

impl<M: Manager, O, const ENTRY_CAP: usize> EntryGuard<'_, M, O, ENTRY_CAP>
where
    O: Copy + Eq,
{
    /// Is this entry occupied?
    #[inline]
    fn is_occupied(&self) -> bool {
        // SAFETY: The entry is locked.
        unsafe { *self.0.operands.get() != CountPair::NULL }
    }

    /// Get the value of this entry if it is occupied and the key (`operator`
    /// and `operands`) matches
    ///
    /// Assumes that
    /// - there is at least one operand,
    /// - there are at most `KIND_COUNT` operands and values of each kind, and
    /// - the count of operands and values is at most `ENTRY_CAP`.
    #[inline]
    fn get<const E: usize, const N: usize>(
        &self,
        manager: &M,
        operator: O,
        operands: (&[Borrowed<M::Edge>], &[u32]),
    ) -> Option<([M::Edge; E], [u32; N])> {
        // These conditions are ensured when called by
        // `DMApplyCache::get_with_numeric()`
        debug_assert_ne!(operands.0.len() + operands.1.len(), 0);
        debug_assert!(operands.0.len() <= KIND_COUNT);
        debug_assert!(operands.1.len() <= KIND_COUNT);
        debug_assert!(E <= KIND_COUNT);
        debug_assert!(N <= KIND_COUNT);
        debug_assert!(operands.0.len() + operands.1.len() + E + N <= ENTRY_CAP);

        #[cfg(feature = "statistics")]
        STAT_ACCESSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let num_operands = CountPair::new(operands.0.len(), operands.1.len());
        // SAFETY: The entry is locked.
        if unsafe { *self.0.operands.get() } != num_operands {
            return None;
        }

        // SAFETY: The entry is locked.
        let mut data = unsafe { &*self.0.data.get() }.iter();
        for (o1, o2) in operands.0.iter().zip(data.by_ref()) {
            // SAFETY: The first `operands.len()` operands are edges
            if &**o1 != unsafe { o2.assume_edge_ref() } {
                return None;
            }
        }
        for (&o1, o2) in operands.1.iter().zip(data.by_ref()) {
            // SAFETY: The next `num_numeric_operands` operands are numeric
            if o1 != unsafe { o2.assume_numeric() } {
                return None;
            }
        }

        // SAFETY: The entry is (1,2) locked and (2) occupied.
        if unsafe { (*self.0.operator.get()).assume_init() } != operator
            || unsafe { *self.0.values.get() } != const { CountPair::new(E, N) }
        {
            return None;
        }

        #[cfg(feature = "statistics")]
        STAT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let (edge_values, remaining) = data.as_slice().split_at(E);
        let numeric_values = &remaining[..N];
        Some((
            // SAFETY: The next `E` values in `data` are edges
            std::array::from_fn(|i| unsafe {
                manager.clone_edge(edge_values[i].assume_edge_ref())
            }),
            // SAFETY: The final `N` values in `data` are numeric
            std::array::from_fn(|i| unsafe { numeric_values[i].assume_numeric() }),
        ))
    }

    /// Set the key/value of this entry
    ///
    /// If `self` is already occupied and the key matches, the entry is not
    /// updated (`operands` and `value` are not cloned).
    ///
    /// Assumes that
    /// - there is at least one operand,
    /// - there are at most `KIND_COUNT` operands and values of each kind, and
    /// - the count of operands and values is at most `ENTRY_CAP`.
    #[inline]
    fn set(
        &mut self,
        operator: O,
        operands: (&[Borrowed<M::Edge>], &[u32]),
        values: (&[Borrowed<M::Edge>], &[u32]),
    ) {
        debug_assert_ne!(operands.0.len() + operands.1.len(), 0);
        debug_assert!(operands.0.len() <= KIND_COUNT);
        debug_assert!(operands.1.len() <= KIND_COUNT);
        debug_assert!(values.0.len() <= KIND_COUNT);
        debug_assert!(values.1.len() <= KIND_COUNT);
        debug_assert!(
            operands.0.len() + operands.1.len() + values.0.len() + values.1.len() <= ENTRY_CAP
        );

        self.clear();

        #[cfg(feature = "statistics")]
        STAT_INSERTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // SAFETY (next 2 dereference ops): The entry is locked.
        unsafe { &mut *self.0.operator.get() }.write(operator);
        let mut data = unsafe { &mut *self.0.data.get() }.iter_mut();
        for (src, dst) in operands.0.iter().zip(data.by_ref()) {
            dst.write_edge(src.borrowed());
        }
        for (&src, dst) in operands.1.iter().zip(data.by_ref()) {
            dst.write_numeric(src);
        }
        for (src, dst) in values.0.iter().zip(data.by_ref()) {
            dst.write_edge(src.borrowed());
        }
        for (&src, dst) in values.1.iter().zip(data) {
            dst.write_numeric(src);
        }

        // Important: Set the counts last for exception safety (the functions
        // above might panic).
        unsafe { *self.0.values.get() = CountPair::new(values.0.len(), values.1.len()) };
        unsafe { *self.0.operands.get() = CountPair::new(operands.0.len(), operands.1.len()) };
    }

    /// Clear this entry
    #[inline(always)]
    fn clear(&mut self) {
        // SAFETY: The entry is locked.
        unsafe { *self.0.operands.get() = CountPair::NULL };
        // `Edge`s are just borrowed, so nothing else to do.
    }
}

impl<M, O, H, const ENTRY_CAP: usize> DMApplyCache<M, O, H, ENTRY_CAP>
where
    M: Manager,
    O: Copy + Eq + Hash,
    H: Hasher + Default,
{
    const CHECK_ENTRY_CAP: () = {
        assert!(
            0 < ENTRY_CAP && ENTRY_CAP <= KIND_COUNT * 4,
            "ENTRY_CAP must be in range [1, 64]"
        );
    };

    /// Create a new `ApplyCache` with the given capacity (entries).
    ///
    /// # Safety
    ///
    /// The apply cache must only be used inside a manager that guarantees all
    /// node deletions to be wrapped inside an
    /// [`ManagerEventSubscriber::pre_gc()`] /
    /// [`ManagerEventSubscriber::post_gc()`] pair.
    pub unsafe fn with_capacity(capacity: usize) -> Self {
        let () = Self::CHECK_ENTRY_CAP;
        let buckets = capacity
            .checked_next_power_of_two()
            .expect("capacity is too large");
        #[cfg(not(feature = "hugealloc"))]
        let mut vec = Vec::with_capacity(buckets);
        #[cfg(feature = "hugealloc")]
        let mut vec = Vec::with_capacity_in(buckets, hugealloc::HugeAlloc);

        vec.resize_with(buckets, || Entry::INIT);
        DMApplyCache(vec.into_boxed_slice(), PhantomData)
    }

    /// Get the bucket for the given operator and operands
    #[inline]
    fn bucket(
        &self,
        operator: O,
        operands: (&[Borrowed<M::Edge>], &[u32]),
    ) -> &Entry<M, O, ENTRY_CAP> {
        let mut hasher = H::default();
        operator.hash(&mut hasher);
        for o in operands.0 {
            o.hash(&mut hasher);
        }
        for o in operands.1 {
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

impl<M, O, H, const ENTRY_CAP: usize> crate::StatisticsGenerator
    for DMApplyCache<M, O, H, ENTRY_CAP>
where
    M: Manager,
    O: Copy + Eq,
{
    #[cfg(not(feature = "statistics"))]
    fn print_stats(&self) {}

    #[cfg(feature = "statistics")]
    fn print_stats(&self) {
        let count = self.0.len();
        debug_assert_ne!(count, 0);
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

impl<M, O, H, const ENTRY_CAP: usize> ApplyCache<M, O> for DMApplyCache<M, O, H, ENTRY_CAP>
where
    M: Manager,
    O: Copy + Hash + Ord,
    H: Hasher + Default,
{
    #[inline(always)]
    fn get_extended<const E: usize, const N: usize>(
        &self,
        manager: &M,
        operator: O,
        operands: (&[Borrowed<M::Edge>], &[u32]),
    ) -> Option<([M::Edge; E], [u32; N])> {
        let total_operands = operands.0.len() + operands.1.len();
        if total_operands == 0
            || total_operands + (N + E) > ENTRY_CAP
            || operands.0.len() > KIND_COUNT
            || operands.1.len() > KIND_COUNT
            || N > KIND_COUNT
            || E > KIND_COUNT
        {
            return None;
        }
        self.bucket(operator, operands)
            .try_lock()?
            .get(manager, operator, operands)
    }

    #[inline(always)]
    fn add_extended(
        &self,
        _manager: &M,
        operator: O,
        operands: (&[Borrowed<M::Edge>], &[u32]),
        values: (&[Borrowed<M::Edge>], &[u32]),
    ) {
        let total_operands = operands.0.len() + operands.1.len();
        if total_operands == 0
            || total_operands + (values.0.len() + values.1.len()) > ENTRY_CAP
            || operands.0.len() > KIND_COUNT
            || operands.1.len() > KIND_COUNT
            || values.0.len() > KIND_COUNT
            || values.1.len() > KIND_COUNT
        {
            return;
        }
        if let Some(mut entry) = self.bucket(operator, operands).try_lock() {
            entry.set(operator, operands, values);
        }
    }

    fn clear(&self, _manager: &M) {
        for entry in &*self.0 {
            entry.lock().clear();
        }
    }
}

impl<M, O, H, const ENTRY_CAP: usize> ManagerEventSubscriber<M> for DMApplyCache<M, O, H, ENTRY_CAP>
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

impl<M, O, H, const ENTRY_CAP: usize> fmt::Debug for DMApplyCache<M, O, H, ENTRY_CAP>
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

impl<M, O, const ENTRY_CAP: usize> fmt::Debug for EntryGuard<'_, M, O, ENTRY_CAP>
where
    M: Manager,
    M::Edge: fmt::Debug,
    O: Copy + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The entry is locked.
        let operands = unsafe { *self.0.operands.get() };
        if operands == CountPair::NULL {
            return f.write_str("None");
        }

        // SAFETY: The entry is (1-3) locked and (1) occupied.
        let operator = unsafe { (*self.0.operator.get()).assume_init() };
        let values = unsafe { *self.0.operands.get() };
        let mut data = unsafe { &*self.0.data.get() }.iter();

        f.write_str("{{ ")?;
        operator.fmt(f)?;

        let mut tuple = f.debug_tuple("");
        for operand in data.by_ref().take(operands.edge()) {
            tuple.field(unsafe { operand.assume_edge_ref() });
        }
        for operand in data.by_ref().take(operands.numeric()) {
            tuple.field(&unsafe { operand.assume_numeric() });
        }
        tuple.finish()?;

        f.write_str(" = ")?;

        let mut tuple = f.debug_tuple("");
        for value in data.by_ref().take(values.edge()) {
            tuple.field(unsafe { value.assume_edge_ref() });
        }
        for value in data.by_ref().take(values.numeric()) {
            tuple.field(&unsafe { value.assume_numeric() });
        }
        tuple.finish()?;

        f.write_str(" }}")
    }
}
