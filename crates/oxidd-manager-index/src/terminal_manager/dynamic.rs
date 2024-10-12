use std::cell::UnsafeCell;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use crossbeam_utils::CachePadded;
use linear_hashtbl::raw::RawTable;
use oxidd_core::util::OutOfMemory;
use parking_lot::Mutex;
use parking_lot::MutexGuard;
use rustc_hash::FxHasher;

use oxidd_core::util::AllocResult;
use oxidd_core::Tag;

use crate::manager::Edge;
use crate::manager::InnerNodeCons;
use crate::manager::TerminalManagerCons;
use crate::node::NodeBase;

use super::TerminalManager;

pub struct DynamicTerminalManager<'id, T, N, ET, const TERMINALS: usize> {
    store: Box<[UnsafeCell<Slot<T>>]>,
    state: CachePadded<Mutex<State<'id, N, ET>>>,
}

struct State<'id, N, ET> {
    /// Next free index in the `store` of `DynamicTerminalManager`
    ///
    /// SAFETY invariant: Unless `next_free` is out of bounds for the store
    /// array of [`DynamicTerminalManager`], it references an empty [`Slot`]. If
    /// the state lock is held, there is exclusive access to that slot.
    next_free: u32,

    /// Set of indices into the `store` field of `DynamicTerminalManager`
    ///
    /// The hash values are those of the referenced terminal values.
    unique_table: RawTable<u32, u32>,

    phantom: PhantomData<Edge<'id, N, ET>>,
}

union Slot<T> {
    node: ManuallyDrop<ArcItem<T>>,

    /// SAFETY invariant: Unless `next_free` is out of bounds for the store
    /// array of [`DynamicTerminalManager`], it references an empty [`Slot`]. If
    /// the state lock is held, there is exclusive access to that slot.
    next_free: u32,
}

struct ArcItem<T> {
    rc: AtomicU32,
    value: T,
}

/// SAFETY: `id` must be a valid terminal ID for `store`
unsafe fn retain<T>(store: &[UnsafeCell<Slot<T>>], id: usize) {
    // SAFETY: Since `id` is a valid terminal ID, it is `<= store.len()`.
    // Furthermore, we have shared access to the referenced slot and the `Slot`
    // is a `node`.
    let item = unsafe { &(*store.get_unchecked(id).get()).node };
    let old_rc = item.rc.fetch_add(1, Relaxed);
    if old_rc > (u32::MAX >> 1) {
        std::process::abort(); // prevent overflow
    }
}

fn hash<T: Hash>(terminal: &T) -> u64 {
    let mut hasher = FxHasher::default();
    terminal.hash(&mut hasher);
    hasher.finish()
}

impl<T, N, ET, const TERMINALS: usize> DynamicTerminalManager<'_, T, N, ET, TERMINALS> {
    const CHECK_TERMINALS: () = assert!(
        TERMINALS < (1 << (u32::BITS - 1)),
        "`TERMINALS` is too large"
    );
}

impl<'id, T, N, ET, const TERMINALS: usize> TerminalManager<'id, N, ET, TERMINALS>
    for DynamicTerminalManager<'id, T, N, ET, TERMINALS>
where
    T: Eq + Hash,
    N: NodeBase,
    ET: Tag,
{
    type TerminalNode = T;
    type TerminalNodeRef<'a>
        = &'a T
    where
        Self: 'a;

    type Iterator<'a>
        = DynamicTerminalIterator<'a, 'id, T, N, ET>
    where
        Self: 'a,
        'id: 'a;

    fn with_capacity(capacity: u32) -> Self {
        let _ = Self::CHECK_TERMINALS;
        let capacity = std::cmp::min(TERMINALS, capacity as usize);

        let mut store_vec = Vec::new();
        let mut i = 0u32;
        store_vec.resize_with(capacity, || {
            i += 1;
            UnsafeCell::new(Slot { next_free: i })
        });

        Self {
            store: store_vec.into_boxed_slice(),
            state: CachePadded::new(Mutex::new(State {
                next_free: 0,
                unique_table: RawTable::new(),
                phantom: PhantomData,
            })),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.state.lock().unique_table.len()
    }

    #[inline]
    unsafe fn get_terminal(&self, id: usize) -> &T {
        // SAFETY: Since `id` is a valid terminal ID, it is
        // `<= self.store.len()`. Furthermore, we have shared access to the
        // referenced `Slot` and the slot is a `node`.
        &unsafe { &(*self.store.get_unchecked(id).get()).node }.value
    }

    #[inline]
    unsafe fn retain(&self, id: usize) {
        // SAFETY: `id` is a valid terminal ID for `self.store`
        unsafe { retain(&self.store, id) }
    }

    #[inline]
    unsafe fn release(&self, id: usize) {
        // SAFETY: Since `id` is a valid terminal ID, it is
        // `<= self.store.len()`. Furthermore, we have shared access to the
        // referenced `Slot` and the slot is a `node`.
        let item = unsafe { &(*self.store.get_unchecked(id).get()).node };
        // Synchronizes-with the load in `Self::gc()`
        let _old_rc = item.rc.fetch_sub(1, Release);
        debug_assert!(
            _old_rc > 1,
            "dropping the last reference should only happen during garbage collection"
        );
    }

    #[inline]
    fn get_edge(&self, terminal: T) -> AllocResult<Edge<'id, N, ET>> {
        let mut state = self.state.lock();
        let hash = hash(&terminal);
        let id = match state.unique_table.find_or_find_insert_slot(
            hash,
            // SAFETY: The IDs stored in the table are valid, hence
            // `<= self.store.len()`. We have shared access to the referenced
            // `Slot` and the slot is a `node`.
            |&id| unsafe { &(*self.store.get_unchecked(id as usize).get()).node }.value == terminal,
        ) {
            Ok(slot) => {
                // SAFETY: `slot` was returned by
                // `state.unique_table.find_or_find_insert_slot()` and there
                // were no modifications of the table in between.
                let id = *unsafe { state.unique_table.get_at_slot_unchecked(slot) };
                unsafe { self.retain(id as usize) };
                id
            }
            Err(table_slot) => {
                let id = state.next_free;
                if id == self.store.len() as u32 {
                    return Err(OutOfMemory);
                }
                // SAFETY: holds by invariant of `state.next_free`
                let store_slot = unsafe { &mut *self.store.get_unchecked(id as usize).get() };
                state.next_free = unsafe { store_slot.next_free };
                store_slot.node = ManuallyDrop::new(ArcItem {
                    rc: AtomicU32::new(2),
                    value: terminal,
                });
                // SAFETY: `table_slot` was returned by
                // `state.unique_table.find_or_find_insert_slot()` and there
                // were no modifications of the table in between.
                unsafe {
                    state
                        .unique_table
                        .insert_in_slot_unchecked(hash, table_slot, id)
                };
                id
            }
        };

        Ok(unsafe { Edge::from_terminal_id(id) })
    }

    #[inline]
    fn iter<'a>(&'a self) -> Self::Iterator<'a>
    where
        Self: 'a,
    {
        let state = self.state.lock();
        let len = state.unique_table.len();
        DynamicTerminalIterator {
            store: &self.store,
            state,
            next_slot: 0,
            len,
        }
    }

    #[inline(always)]
    fn gc(&self) -> u32 {
        let mut collected = 0;
        let mut state = self.state.lock();
        // Use a local variable here because otherwise, the borrow checker
        // complains about `state` being borrowed mutably twice. In case of a
        // panic (which should not happen) we would potentially loose a few
        // slots, but this is not a SAFETY issue.
        let mut next_free = state.next_free;
        state.unique_table.retain(
            |&mut id| {
                // SAFETY: The IDs stored in the table are valid, hence
                // `<= self.store.len()`. We have shared access to the
                // referenced `Slot` and the slot is a `node`.
                let node = unsafe { &(*self.store.get_unchecked(id as usize).get()).node };
                // The following load synchronizes-with the `fetch_sub()` in
                // `release()`. Releasing terminal nodes and garbage collection
                // may run in parallel.
                node.rc.load(Acquire) != 1
            },
            |id| {
                // SAFETY: as above
                let slot = unsafe { &mut *self.store.get_unchecked(id as usize).get() };
                unsafe { ManuallyDrop::drop(&mut slot.node) };
                slot.next_free = next_free;
                next_free = id;
                collected += 1;
            },
        );
        state.next_free = next_free;
        collected
    }
}

unsafe impl<T: Send + Sync, N: Send + Sync, ET: Send + Sync, const TERMINALS: usize> Send
    for DynamicTerminalManager<'_, T, N, ET, TERMINALS>
{
}
unsafe impl<T: Send + Sync, N: Send + Sync, ET: Send + Sync, const TERMINALS: usize> Sync
    for DynamicTerminalManager<'_, T, N, ET, TERMINALS>
{
}

pub struct DynamicTerminalManagerCons<T>(PhantomData<T>);

impl<
        T: Hash + Eq + Send + Sync,
        NC: InnerNodeCons<ET>,
        ET: Tag + Send + Sync,
        const TERMINALS: usize,
    > TerminalManagerCons<NC, ET, TERMINALS> for DynamicTerminalManagerCons<T>
{
    type TerminalNode = T;
    type T<'id> = DynamicTerminalManager<'id, T, NC::T<'id>, ET, TERMINALS>;
}

pub struct DynamicTerminalIterator<'a, 'id, T, N, ET> {
    store: &'a [UnsafeCell<Slot<T>>],
    state: MutexGuard<'a, State<'id, N, ET>>,
    next_slot: usize,
    len: usize,
}

impl<'id, T, N: NodeBase, ET: Tag> Iterator for DynamicTerminalIterator<'_, 'id, T, N, ET> {
    type Item = Edge<'id, N, ET>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }
        self.len -= 1;
        let unique_table = &self.state.unique_table;
        // SAFETY: `self.len` is the number of occupied slots
        // `>= self.next_slot`. Hence, there is a slot `< unique_table.slots()`
        // that is occupied.
        while !unsafe { unique_table.is_slot_occupied_unchecked(self.next_slot) } {
            self.next_slot += 1;
        }
        // SAFETY: `self.next_slot` is occupied.
        let id = *unsafe { unique_table.get_at_slot_unchecked(self.next_slot) };
        self.next_slot += 1;
        // SAFETY: `id` was obtained from the unique_table, hence it is a valid
        // terminal ID for `self.store`.
        unsafe { retain(self.store, id as usize) };
        Some(unsafe { Edge::from_terminal_id(id) })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T, N: NodeBase, ET: Tag> FusedIterator for DynamicTerminalIterator<'_, '_, T, N, ET> {}

impl<T, N: NodeBase, ET: Tag> ExactSizeIterator for DynamicTerminalIterator<'_, '_, T, N, ET> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
