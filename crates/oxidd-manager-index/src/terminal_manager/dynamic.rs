use std::cell::UnsafeCell;
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use crossbeam_utils::CachePadded;
use linear_hashtbl::raw::RawTable;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHasher;

use oxidd_core::error::OutOfMemory;
use oxidd_core::util::{AllocResult, Own};

use crate::{DiagramInfo, manager::Edge, terminal_manager::TerminalManager};

pub struct DynamicTerminalManager<'id, DI: DiagramInfo> {
    store: Box<[UnsafeCell<Slot<DI::TerminalNode>>]>,
    state: CachePadded<Mutex<State<'id, DI>>>,
}

struct State<'id, DI: DiagramInfo> {
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

    phantom: PhantomData<Edge<'id, DI>>,
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
unsafe fn retain<T>(store: &[UnsafeCell<Slot<T>>], id: u32) {
    // SAFETY: Since `id` is a valid terminal ID, it is `<= store.len()`.
    // Furthermore, we have shared access to the referenced slot and the `Slot`
    // is a `node`.
    let item = unsafe { &(*store.get_unchecked(id as usize).get()).node };
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

impl<'id, DI: DiagramInfo> TerminalManager<'id, DI> for DynamicTerminalManager<'id, DI> {
    type TerminalNodeRef<'a>
        = &'a DI::TerminalNode
    where
        Self: 'a;

    type Iterator<'a>
        = DynamicTerminalIterator<'a, 'id, DI>
    where
        Self: 'a,
        'id: 'a;

    fn with_capacity(capacity: u32) -> Self {
        const {
            assert!(
                DI::TERMINALS < (1 << (u32::BITS - 1)),
                "`TERMINALS` is too large"
            )
        };
        let capacity = std::cmp::min(DI::TERMINALS, capacity);

        let mut store_vec = Vec::new();
        let mut i = 0u32;
        store_vec.resize_with(capacity as usize, || {
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
    unsafe fn get_terminal(&self, id: u32) -> &DI::TerminalNode {
        // SAFETY: Since `id` is a valid terminal ID, it is
        // `<= self.store.len()`. Furthermore, we have shared access to the
        // referenced `Slot` and the slot is a `node`.
        &unsafe { &(*self.store.get_unchecked(id as usize).get()).node }.value
    }

    #[inline]
    unsafe fn retain(&self, id: u32) {
        // SAFETY: `id` is a valid terminal ID for `self.store`
        unsafe { retain(&self.store, id) }
    }

    #[inline]
    unsafe fn release(&self, id: u32) {
        // SAFETY: Since `id` is a valid terminal ID, it is
        // `<= self.store.len()`. Furthermore, we have shared access to the
        // referenced `Slot` and the slot is a `node`.
        let item = unsafe { &(*self.store.get_unchecked(id as usize).get()).node };
        // Synchronizes-with the load in `Self::gc()`
        let _old_rc = item.rc.fetch_sub(1, Release);
        debug_assert!(
            _old_rc > 1,
            "dropping the last reference should only happen during garbage collection"
        );
    }

    #[inline]
    fn get_edge(&self, terminal: DI::TerminalNode) -> AllocResult<Own<Edge<'id, DI>>> {
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
                unsafe { self.retain(id) };
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

        Ok(unsafe { Own::from_raw(Edge::from_terminal_id(id)) })
    }

    #[inline]
    fn iter<'a>(&'a self) -> Self::Iterator<'a> {
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

unsafe impl<DI: DiagramInfo> Send for DynamicTerminalManager<'_, DI> {}
unsafe impl<DI: DiagramInfo> Sync for DynamicTerminalManager<'_, DI> {}

pub struct DynamicTerminalIterator<'a, 'id, DI: DiagramInfo> {
    store: &'a [UnsafeCell<Slot<DI::TerminalNode>>],
    state: MutexGuard<'a, State<'id, DI>>,
    next_slot: usize,
    len: usize,
}

impl<'id, DI: DiagramInfo> Iterator for DynamicTerminalIterator<'_, 'id, DI> {
    type Item = Own<Edge<'id, DI>>;

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
        unsafe { retain(self.store, id) };
        Some(unsafe { Own::from_raw(Edge::from_terminal_id(id)) })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<DI: DiagramInfo> FusedIterator for DynamicTerminalIterator<'_, '_, DI> {}

impl<DI: DiagramInfo> ExactSizeIterator for DynamicTerminalIterator<'_, '_, DI> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}
