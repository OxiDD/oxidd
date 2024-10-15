//! Index-Based Manager Implementation
//!
//! Abbreviations of generics:
//!
//! | Abbreviation | Meaning                           |
//! | ------------ | --------------------------------- |
//! | `N`          | Inner Node                        |
//! | `ET`         | Edge Tag                          |
//! | `TM`         | Terminal Manager                  |
//! | `R`          | Diagram Rules                     |
//! | `MD`         | Manager Data                      |
//! | `NC`         | Inner Node Type Constructor       |
//! | `TMC`        | Terminal Manager Type Constructor |
//! | `RC`         | Diagram Rules Type Constructor    |
//! | `OP`         | Operation                         |

use std::cell::{Cell, UnsafeCell};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::Ordering::{Acquire, Relaxed};
use std::sync::Arc;

use bitvec::vec::BitVec;
use crossbeam_utils::CachePadded;
use linear_hashtbl::raw::RawTable;
use parking_lot::{Condvar, Mutex, MutexGuard};
use rustc_hash::FxHasher;

use oxidd_core::function::EdgeOfFunc;
use oxidd_core::util::{AbortOnDrop, AllocResult, Borrowed, DropWith, GCContainer, OutOfMemory};
use oxidd_core::{DiagramRules, InnerNode, LevelNo, Tag};

use crate::node::NodeBase;
use crate::terminal_manager::TerminalManager;
use crate::util::{rwlock::RwLock, Invariant, TryLock};

// === Type Constructors =======================================================

/// Inner node type constructor
pub trait InnerNodeCons<ET: Tag> {
    type T<'id>: NodeBase + InnerNode<Edge<'id, Self::T<'id>, ET>> + Send + Sync;
}

/// Terminal manager type constructor
pub trait TerminalManagerCons<NC: InnerNodeCons<ET>, ET: Tag, const TERMINALS: usize>:
    Sized
{
    type TerminalNode: Send + Sync;
    type T<'id>: Send
        + Sync
        + TerminalManager<'id, NC::T<'id>, ET, TERMINALS, TerminalNode = Self::TerminalNode>;
}

/// Diagram rules type constructor
pub trait DiagramRulesCons<
    NC: InnerNodeCons<ET>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, TERMINALS>,
    MDC: ManagerDataCons<NC, ET, TMC, Self, TERMINALS>,
    const TERMINALS: usize,
>: Sized
{
    type T<'id>: Send
        + Sync
        + DiagramRules<
            Edge<'id, NC::T<'id>, ET>,
            NC::T<'id>,
            <TMC::T<'id> as TerminalManager<'id, NC::T<'id>, ET, TERMINALS>>::TerminalNode,
        >;
}

/// Manager data type constructor
pub trait ManagerDataCons<
    NC: InnerNodeCons<ET>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, TERMINALS>,
    RC: DiagramRulesCons<NC, ET, TMC, Self, TERMINALS>,
    const TERMINALS: usize,
>: Sized
{
    type T<'id>: Send
        + Sync
        + DropWith<Edge<'id, NC::T<'id>, ET>>
        + GCContainer<Manager<'id, NC::T<'id>, ET, TMC::T<'id>, RC::T<'id>, Self::T<'id>, TERMINALS>>;
}

// === Manager & Edges =========================================================

/// "Signals" used to communicate with the garbage collection thread
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GCSignal {
    RunGc,
    Quit,
}

pub struct Store<'id, N, ET, TM, R, MD, const TERMINALS: usize>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    inner_nodes: Box<SlotSlice<'id, N, TERMINALS>>,
    manager: RwLock<Manager<'id, N, ET, TM, R, MD, TERMINALS>>,
    terminal_manager: TM,
    state: CachePadded<Mutex<SharedStoreState>>,
    gc_signal: (Mutex<GCSignal>, Condvar),
    workers: crate::workers::Workers,
}

unsafe impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> Sync
    for Store<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>> + Send + Sync,
    ET: Tag + Send + Sync,
    TM: TerminalManager<'id, N, ET, TERMINALS> + Send + Sync,
    MD: DropWith<Edge<'id, N, ET>> + Send + Sync,
{
}

/// Size of a pre-allocation (number of nodes)
const CHUNK_SIZE: u32 = 64 * 1024;

/// State of automatic background garbage collection
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GCState {
    /// Automatic garbage collection is disabled entirely
    Disabled,
    /// Initial state
    Init,
    /// Garbage collection has been triggered because the state was `Init` and
    /// the node count exceeded the high water mark
    ///
    /// The collecting thread sets this state back to `Init` if the node count
    /// reaches a value less than the low water mark.
    Triggered,
}

struct SharedStoreState {
    /// IDs of the next free slots
    ///
    /// Each element of the vector corresponds to a linked list of free slots.
    /// If a worker runs out of locally allocated free slots, it will first try
    /// to take a list of free slots from here. If this is not possible, a new
    /// chunk of uninitialized slots will be allocated. New entries to this
    /// vector are added during garbage collection. Each list should not be
    /// longer than `CHUNK_SIZE` (this is not a strict requirement, though).
    next_free: Vec<u32>,

    /// All slots in range `..allocated` are either initialized (a `node` or a
    /// `next_free` ID), or `uninit` and pre-allocated by a worker.
    ///
    /// Note that this is an index into the slice, so no `- TERMINALS` is needed
    /// to access the slot.
    allocated: u32,

    /// Eventually consistent count of inner nodes
    ///
    /// This is an `i64` and not an `u32` because of the following scenario:
    /// Worker A creates n nodes such that its `node_count_delta` becomes n - 1
    /// and the shared `node_count` is 1. At least two of the newly created
    /// nodes are solely referenced by a [`Function`]. An application thread
    /// drops these two functions. This will directly decrement the shared
    /// `node_count` by 1 twice. If we used a `u32`, this would lead to an
    /// overflow.
    node_count: i64,

    /// Background garbage collection state (see [`GCState`])
    gc_state: GCState,
    /// Low water mark for background garbage collection (see [`GCState`])
    gc_lwm: u32,
    /// High water mark for background garbage collection (see [`GCState`])
    gc_hwm: u32,
}

#[repr(align(64))] // all fields on a single cache line
struct LocalStoreState {
    /// ID of the next free slot
    ///
    /// `0` means that there is no such slot so far. Either `initialized` can be
    /// incremented or we are out of memory. Since `0` is always a terminal (we
    /// require `TERMINALS >= 1`), there is no ambiguity.
    next_free: Cell<u32>,

    /// All slots in range `..initialized` are not `uninit`, i.e. either a
    /// `node` or a `next_free` ID. All slots until the next multiple of
    /// `CHUNK_SIZE` can be used by the local thread. So if
    /// `initialized % CHUNK_SIZE == 0` the worker needs to allocate a new chunk
    /// from the shared store state.
    ///
    /// Note that this is an index into the slice, so no `- TERMINALS` is needed
    /// to access the slot.
    initialized: Cell<u32>,

    /// Address of the associated `Store`
    ///
    /// Before using slots from this `LocalStoreState`, the implementation needs
    /// to ensure that the slots come from the respective `Store`.
    current_store: Cell<usize>,

    /// Local changes to the shared node counter
    ///
    /// In general, we synchronize with the shared counter if request slots from
    /// the shared manager or we return the `next_free` list.  The latter
    /// happens if this counter reaches `-CHUNK_SIZE`.
    node_count_delta: Cell<i32>,
}

thread_local! {
    static LOCAL_STORE_STATE: LocalStoreState = const {
        LocalStoreState {
            next_free: Cell::new(0),
            initialized: Cell::new(0),
            current_store: Cell::new(0),
            node_count_delta: Cell::new(0),
        }
    };
}

struct LocalStoreStateGuard<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize>(
    &'a Store<'id, N, ET, TM, R, MD, TERMINALS>,
)
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>;

union Slot<N> {
    node: ManuallyDrop<N>,
    next_free: u32,
    #[allow(unused)]
    uninit: (),
}

#[repr(transparent)]
struct SlotSlice<'id, N, const TERMINALS: usize> {
    phantom: PhantomData<Invariant<'id>>,
    slots: [UnsafeCell<Slot<N>>],
}

/// Edge type, see [`oxidd_core::Edge`]
///
/// Internally, this is represented as a `u32` index.
#[repr(transparent)]
#[must_use]
pub struct Edge<'id, N, ET>(
    /// `node_index | edge_tag << (u32::BITS - Self::TAG_BITS)`
    u32,
    PhantomData<(Invariant<'id>, N, ET)>,
);

#[repr(C)]
pub struct Manager<'id, N, ET, TM, R, MD, const TERMINALS: usize>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    unique_table: Vec<Mutex<LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>>>,
    data: ManuallyDrop<MD>,
    /// Pointer back to the store, obtained via [`Arc::as_ptr()`].
    ///
    /// Theoretically, we should be able to get the pointer from a `&Manager`
    /// reference, but this leads to provenance issues.
    store: *const Store<'id, N, ET, TM, R, MD, TERMINALS>,
    reorder_count: u64,
    gc_ongoing: TryLock,
}

/// Type "constructor" for the manager from `InnerNodeCons` etc.
type M<'id, NC, ET, TMC, RC, MDC, const TERMINALS: usize> = Manager<
    'id,
    <NC as InnerNodeCons<ET>>::T<'id>,
    ET,
    <TMC as TerminalManagerCons<NC, ET, TERMINALS>>::T<'id>,
    <RC as DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>>::T<'id>,
    <MDC as ManagerDataCons<NC, ET, TMC, RC, TERMINALS>>::T<'id>,
    TERMINALS,
>;

unsafe impl<
        'id,
        N: NodeBase + InnerNode<Edge<'id, N, ET>> + Send + Sync,
        ET: Tag + Send + Sync,
        TM: TerminalManager<'id, N, ET, TERMINALS> + Send + Sync,
        R,
        MD: DropWith<Edge<'id, N, ET>> + Send + Sync,
        const TERMINALS: usize,
    > Send for Manager<'id, N, ET, TM, R, MD, TERMINALS>
{
}
unsafe impl<
        'id,
        N: NodeBase + InnerNode<Edge<'id, N, ET>> + Send + Sync,
        ET: Tag + Send + Sync,
        TM: TerminalManager<'id, N, ET, TERMINALS> + Send + Sync,
        R,
        MD: DropWith<Edge<'id, N, ET>> + Send + Sync,
        const TERMINALS: usize,
    > Sync for Manager<'id, N, ET, TM, R, MD, TERMINALS>
{
}

// --- Edge Impls --------------------------------------------------------------

impl<N: NodeBase, ET: Tag> Edge<'_, N, ET> {
    const TAG_BITS: u32 = {
        let bits = usize::BITS - ET::MAX_VALUE.leading_zeros();
        assert!(bits <= 16, "Maximum value of edge tag is too large");
        bits
    };
    /// Shift amount to store/retrieve the tag from the most significant bits
    const TAG_SHIFT: u32 = (u32::BITS - Self::TAG_BITS) % 32;

    /// Mask corresponding to [`Self::TAG_BITS`]
    const TAG_MASK: u32 = ((1 << Self::TAG_BITS) - 1) << Self::TAG_SHIFT;

    /// SAFETY: The edge must be untagged.
    #[inline(always)]
    unsafe fn node_id_unchecked(&self) -> u32 {
        debug_assert_eq!(self.0 & Self::TAG_MASK, 0);
        self.0
    }

    /// Get the node ID, i.e. the edge value without any tags
    #[inline(always)]
    fn node_id(&self) -> usize {
        (self.0 & !Self::TAG_MASK) as usize
    }

    /// Returns `true` iff the edge's tag != 0
    #[inline(always)]
    fn is_tagged(&self) -> bool {
        self.0 & Self::TAG_MASK != 0
    }

    /// Get the raw representation of the edge (also called "edge value")
    #[inline(always)]
    pub fn raw(&self) -> u32 {
        self.0
    }

    /// Get an edge from a terminal ID
    ///
    /// # Safety
    ///
    /// `id` must be a terminal ID, i.e. `id < TERMINALS`, and the caller must
    /// update the reference count for the terminal accordingly.
    #[inline(always)]
    pub unsafe fn from_terminal_id(id: u32) -> Self {
        Self(id, PhantomData)
    }
}

impl<N, ET> PartialEq for Edge<'_, N, ET> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<N, ET> Eq for Edge<'_, N, ET> {}

impl<N, ET> PartialOrd for Edge<'_, N, ET> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<N, ET> Ord for Edge<'_, N, ET> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<N, ET> Hash for Edge<'_, N, ET> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<N: NodeBase, ET: Tag> oxidd_core::Edge for Edge<'_, N, ET> {
    type Tag = ET;

    #[inline]
    fn borrowed(&self) -> Borrowed<Self> {
        Borrowed::new(Self(self.0, PhantomData))
    }

    #[inline]
    fn with_tag(&self, tag: Self::Tag) -> Borrowed<Self> {
        Borrowed::new(Self(
            (self.0 & !Self::TAG_MASK) | ((tag.as_usize() as u32) << Self::TAG_SHIFT),
            PhantomData,
        ))
    }

    #[inline]
    fn with_tag_owned(mut self, tag: Self::Tag) -> Self {
        self.0 = (self.0 & !Self::TAG_MASK) | ((tag.as_usize() as u32) << Self::TAG_SHIFT);
        self
    }

    #[inline]
    fn tag(&self) -> Self::Tag {
        ET::from_usize(((self.0 & Self::TAG_MASK) >> Self::TAG_SHIFT) as usize)
    }

    #[inline]
    fn node_id(&self) -> oxidd_core::NodeID {
        (self.0 & !Self::TAG_MASK) as oxidd_core::NodeID
    }
}

impl<N, ET> Drop for Edge<'_, N, ET> {
    #[inline(never)]
    #[cold]
    fn drop(&mut self) {
        eprintln!(
            "`Edge`s must not be dropped. Use `Manager::drop_edge()`. Backtrace:\n{}",
            std::backtrace::Backtrace::capture()
        );

        #[cfg(feature = "static_leak_check")]
        {
            extern "C" {
                #[link_name = "\n\n`Edge`s must not be dropped. Use `Manager::drop_edge()`.\n"]
                fn trigger() -> !;
            }
            unsafe { trigger() }
        }
    }
}

// --- SlotSlice Impl ----------------------------------------------------------

impl<'id, N: NodeBase, const TERMINALS: usize> SlotSlice<'id, N, TERMINALS> {
    // Create a new slot slice for up to `capacity` nodes
    fn new_boxed(capacity: u32) -> Box<Self> {
        let mut vec: Vec<UnsafeCell<Slot<N>>> = Vec::with_capacity(capacity as usize);

        // SAFETY: The new length is equal to the capacity. All elements are
        // "initialized" as `Slot::uninit`.
        //
        // Clippy's `uninit_vec` lint is a bit too strict here. `Slot`s are
        // somewhat like `MaybeUninit`, but Clippy wants `MaybeUninit`.
        #[allow(clippy::uninit_vec)]
        unsafe {
            vec.set_len(capacity as usize)
        };

        let boxed = vec.into_boxed_slice();
        // SAFETY: `SlotSlice` has `repr(transparent)` and thus the same
        // representation as `[UnsafeCell<Slot<N>>]`.
        unsafe { std::mem::transmute(boxed) }
    }

    /// SAFETY: `edge` must be untagged and reference an inner node
    #[inline]
    unsafe fn slot_pointer_unchecked<ET: Tag>(&self, edge: &Edge<'id, N, ET>) -> *mut Slot<N> {
        // SAFETY: `edge` is untagged
        let id = unsafe { edge.node_id_unchecked() } as usize;
        debug_assert!(id >= TERMINALS, "`edge` must reference an inner node");
        debug_assert!(id - TERMINALS < self.slots.len());
        // SAFETY: Indices derived from edges pointing to inner nodes are always
        // in bounds.
        unsafe { self.slots.get_unchecked(id - TERMINALS).get() }
    }

    /// Panics if `edge` references a terminal node
    #[inline]
    fn inner_node<ET: Tag>(&self, edge: &Edge<'id, N, ET>) -> &N {
        let id = edge.node_id();
        assert!(id >= TERMINALS, "`edge` must reference an inner node");
        debug_assert!(id - TERMINALS < self.slots.len());
        // SAFETY:
        // - Indices derived from edges pointing to inner nodes are in bounds
        // - If an edge points to an inner node, the node's `Slot` is immutable and a
        //   properly initialized node
        unsafe { &(*self.slots.get_unchecked(id - TERMINALS).get()).node }
    }

    /// SAFETY: `edge` must be untagged and reference an inner node
    #[inline(always)]
    unsafe fn inner_node_unchecked<ET: Tag>(&self, edge: &Edge<'id, N, ET>) -> &N {
        // SAFETY: If an edge points to an inner node, the node's `Slot` is
        // immutable and a properly initialized node.
        unsafe { &(*self.slot_pointer_unchecked(edge)).node }
    }

    /// SAFETY: `edge` must be untagged and reference an inner node
    #[inline(always)]
    unsafe fn clone_edge_unchecked<ET: Tag>(&self, edge: &Edge<'id, N, ET>) -> Edge<'id, N, ET> {
        unsafe { self.inner_node_unchecked(edge) }.retain();
        Edge(edge.0, PhantomData)
    }
}

// --- Store Impls -------------------------------------------------------------

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> Store<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    const CHECK_TERMINALS: () = {
        assert!(
            usize::BITS >= u32::BITS,
            "This manager implementation assumes usize::BITS >= u32::BITS"
        );
        assert!(TERMINALS >= 1, "TERMINALS must be >= 1");
        assert!(
            TERMINALS <= u32::MAX as usize,
            "TERMINALS must fit into an u32"
        );
    };

    #[inline(always)]
    fn addr(&self) -> usize {
        // TODO: Use the respective strict provenance method once stable
        self as *const Self as usize
    }

    #[inline]
    fn prepare_local_state(
        &self,
    ) -> Option<LocalStoreStateGuard<'_, 'id, N, ET, TM, R, MD, TERMINALS>> {
        LOCAL_STORE_STATE.with(|state| {
            if state.current_store.get() == 0 {
                state.next_free.set(0);
                state.initialized.set(0);
                state.current_store.set(self.addr());
                debug_assert_eq!(state.node_count_delta.get(), 0);
                Some(LocalStoreStateGuard(self))
            } else {
                None
            }
        })
    }

    /// Add a node to the store. Does not perform any duplicate checks.
    ///
    /// Panics if the store is full.
    #[inline]
    fn add_node(&self, node: N) -> AllocResult<[Edge<'id, N, ET>; 2]> {
        debug_assert_eq!(node.load_rc(Relaxed), 2);
        let res = LOCAL_STORE_STATE.with(|state| {
            let node_count_delta = if state.current_store.get() == self.addr() {
                let delta = state.node_count_delta.get() + 1;
                let id = state.next_free.get();
                if id != 0 {
                    // SAFETY: `id` is the ID of a free slot, we have exclusive access
                    let (next_free, slot) = unsafe { self.use_free_slot(id) };
                    state.next_free.set(next_free);
                    state.node_count_delta.set(delta);
                    return Ok((id, slot));
                }

                let index = state.initialized.get();
                if index % CHUNK_SIZE != 0 {
                    let slots = &self.inner_nodes.slots;
                    debug_assert!((index as usize) < slots.len());
                    // SAFETY: All slots in range
                    // `index..index.next_multiple_of(CHUNK_SIZE)` are
                    // uninitialized and pre-allocated, so we have exclusive
                    // access to them.
                    let slot = unsafe { &mut *slots.get_unchecked(index as usize).get() };
                    state.initialized.set(index + 1);
                    state.node_count_delta.set(delta);
                    return Ok((index + TERMINALS as u32, slot));
                }

                state.node_count_delta.set(0);
                delta
            } else {
                1
            };

            self.get_slot_from_shared(node_count_delta)
        });
        match res {
            Ok((id, slot)) => {
                slot.node = ManuallyDrop::new(node);
                Ok([Edge(id, PhantomData), Edge(id, PhantomData)])
            }
            Err(OutOfMemory) => {
                node.drop_with(|e| self.drop_edge(e));
                Err(OutOfMemory)
            }
        }
    }

    /// Get the free slot with ID `id` and load the `next_free` ID
    ///
    /// SAFETY: `id` must be the ID of a free slot and the current thread must
    /// have exclusive access to it.
    #[inline(always)]
    unsafe fn use_free_slot(&self, id: u32) -> (u32, &mut Slot<N>) {
        debug_assert!(id as usize >= TERMINALS);
        let index = id as usize - TERMINALS;
        debug_assert!(index < self.inner_nodes.slots.len());
        // SAFETY: All next-free ids (>= TERMINALS) are valid, we have
        // exclusive access to the node.
        let slot = unsafe { &mut *self.inner_nodes.slots.get_unchecked(index).get() };
        // SAFETY: The slot is free.
        let next_free = unsafe { slot.next_free };
        (next_free, slot)
    }

    /// Get a slot from the shared state
    ///
    /// `delta` is added to the shared node count.
    ///
    /// Returns the ID of a free slot.
    #[cold]
    fn get_slot_from_shared(&self, delta: i32) -> AllocResult<(u32, &mut Slot<N>)> {
        LOCAL_STORE_STATE.with(|local| {
            let mut shared = self.state.lock();

            shared.node_count += delta as i64;
            if shared.gc_state == GCState::Init && shared.node_count >= shared.gc_hwm as i64 {
                shared.gc_state = GCState::Triggered;
                self.gc_signal.1.notify_one();
            }

            if local.current_store.get() == self.addr() {
                debug_assert_eq!(local.next_free.get(), 0);
                debug_assert_eq!(local.initialized.get() % CHUNK_SIZE, 0);

                if let Some(id) = shared.next_free.pop() {
                    // SAFETY: `id` is the ID of a free slot, we have exclusive access
                    let (next_free, slot) = unsafe { self.use_free_slot(id) };
                    local.next_free.set(next_free);
                    return Ok((id, slot));
                }

                let index = shared.allocated;
                let slots = &self.inner_nodes.slots;
                if (index as usize + CHUNK_SIZE as usize) < slots.len() {
                    shared.allocated = (index / CHUNK_SIZE + 1) * CHUNK_SIZE;
                    local.initialized.set(index + 1);
                } else if (index as usize) < slots.len() {
                    shared.allocated += 1;
                } else {
                    return Err(OutOfMemory);
                }
                // SAFETY: `index` is in bounds, the slot is uninitialized
                let slot = unsafe { &mut *slots.get_unchecked(index as usize).get() };
                Ok((index + TERMINALS as u32, slot))
            } else {
                if let Some(id) = shared.next_free.pop() {
                    // SAFETY: `id` is the ID of a free slot, we have exclusive access
                    let (next_free, slot) = unsafe { self.use_free_slot(id) };
                    shared.next_free.push(next_free);
                    return Ok((id, slot));
                }

                let index = shared.allocated;
                let slots = &self.inner_nodes.slots;
                if (index as usize) >= slots.len() {
                    return Err(OutOfMemory);
                }
                shared.allocated += 1;
                // SAFETY: `index` is in bounds, the slot is uninitialized
                let slot = unsafe { &mut *slots.get_unchecked(index as usize).get() };
                Ok((index + TERMINALS as u32, slot))
            }
        })
    }

    /// Drop an edge that originates from the unique table.
    ///
    /// Edges from the unique table are untagged and point to inner nodes, so
    /// we need less case distinctions. Furthermore, we assume that the node's
    /// children are still present in the unique table (waiting for their
    /// revival or to be garbage collected), so we just decrement the children's
    /// reference counters. There is a debug assertion that checks this
    /// assumption.
    ///
    /// SAFETY: `edge` must be untagged and point to an inner node
    unsafe fn drop_unique_table_edge(&self, edge: Edge<'id, N, ET>) {
        // SAFETY (next 2): `edge` is untagged and points to an inner node
        let slot_ptr = unsafe { self.inner_nodes.slot_pointer_unchecked(&edge) };
        let id = unsafe { edge.node_id_unchecked() };
        std::mem::forget(edge);

        // SAFETY:
        // - `edge` points to an inner node
        // - We have shared access to nodes
        // - `edge` is forgotten.
        if unsafe { (*slot_ptr).node.release() } != 1 {
            return;
        }

        // We only have exclusive access to the other fields of node after the
        // fence. It synchronizes with the `NodeBase::release()` above (which is
        // guaranteed to have `Release` order).
        std::sync::atomic::fence(Acquire);

        // SAFETY: Now, we have exclusive access to the slot. It contains a
        // node. `id` is the ID of the slot.
        unsafe { self.free_slot(&mut *slot_ptr, id) };
    }

    /// Free `slot`, i.e. drop the node and add its ID (`id`) to the free list
    ///
    /// SAFETY: `slot` must contain a node. `id` is the ID of `slot`.
    #[inline]
    unsafe fn free_slot(&self, slot: &mut Slot<N>, id: u32) {
        // SAFETY: We don't use the node in `slot` again.
        unsafe { ManuallyDrop::take(&mut slot.node) }.drop_with(|edge| self.drop_edge(edge));

        LOCAL_STORE_STATE.with(|state| {
            if state.current_store.get() == self.addr() {
                slot.next_free = state.next_free.get();
                state.next_free.set(id);

                let delta = state.node_count_delta.get() - 1;
                if delta > -(CHUNK_SIZE as i32) {
                    state.node_count_delta.set(delta);
                } else {
                    let mut shared = self.state.lock();
                    shared.next_free.push(state.next_free.replace(0));
                    shared.node_count += state.node_count_delta.replace(0) as i64;
                }
            } else {
                #[cold]
                fn return_slot<N>(shared: &Mutex<SharedStoreState>, slot: &mut Slot<N>, id: u32) {
                    let mut shared = shared.lock();
                    slot.next_free = shared.next_free.pop().unwrap_or(0);
                    shared.next_free.push(id);
                    shared.node_count -= 1;
                }
                return_slot(&self.state, slot, id);
            }
        });
    }

    /// Drop an edge, assuming that it isn't the last one pointing to the
    /// referenced node
    ///
    /// This assumption is fulfilled in case the node is still stored in the
    /// unique table.
    ///
    /// There is a debug assertion that checks the aforementioned assumption. In
    /// release builds, this function will simply leak the node.
    #[inline]
    fn drop_edge(&self, edge: Edge<'id, N, ET>) {
        let id = edge.node_id();
        if id >= TERMINALS {
            // inner node
            let node = self.inner_nodes.inner_node(&edge);
            std::mem::forget(edge);
            // SAFETY: `edge` is forgotten
            let _old_rc = unsafe { node.release() };
            debug_assert!(_old_rc > 1);
        } else {
            std::mem::forget(edge);
            // SAFETY: `id` is a valid terminal ID
            unsafe { self.terminal_manager.release(id) };
        }
    }

    /// Clone `edge`
    ///
    /// `edge` may be tagged and point to an inner or a terminal node.
    #[inline]
    fn clone_edge(&self, edge: &Edge<'id, N, ET>) -> Edge<'id, N, ET> {
        let id = edge.node_id();
        if id >= TERMINALS {
            // inner node
            self.inner_nodes.inner_node(edge).retain();
        } else {
            // SAFETY: `id` is a valid terminal ID
            unsafe { self.terminal_manager.retain(id) };
        }
        Edge(edge.0, PhantomData)
    }
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> Drop for Store<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    fn drop(&mut self) {
        let manager = self.manager.get_mut();
        // We don't care about reference counters from here on.
        // SAFETY: We don't use `manager.data` again.
        unsafe { ManuallyDrop::take(&mut manager.data) }.drop_with(std::mem::forget);

        if N::needs_drop() {
            let unique_table = std::mem::take(&mut manager.unique_table);
            for level in unique_table {
                for edge in level.into_inner() {
                    // SAFETY (next 2): `edge` is untagged and points to an
                    // inner node
                    let slot_ptr = unsafe { self.inner_nodes.slot_pointer_unchecked(&edge) };
                    std::mem::forget(edge);
                    // SAFETY: We have exclusive access to all nodes in the
                    // store. `edge` points to an inner node.
                    let node = unsafe { ManuallyDrop::take(&mut (*slot_ptr).node) };

                    // We don't care about reference counts anymore.
                    node.drop_with(std::mem::forget);
                }
            }
        }
    }
}

// --- LocalStoreStateGuard impl -----------------------------------------------

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> Drop
    for LocalStoreStateGuard<'_, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    #[inline]
    fn drop(&mut self) {
        #[cold]
        fn drop_slow<N>(
            slots: &[UnsafeCell<Slot<N>>],
            shared_state: &Mutex<SharedStoreState>,
            terminals: u32,
        ) {
            LOCAL_STORE_STATE.with(|local| {
                local.current_store.set(0);
                let start = local.initialized.get();
                let next_free = if start % CHUNK_SIZE != 0 {
                    // We cannot simply give an uninitialized chunk back. Hence,
                    // we prepend the slots to the free list.
                    debug_assert!(start <= u32::MAX - CHUNK_SIZE);
                    let end = (start / CHUNK_SIZE + 1) * CHUNK_SIZE;
                    let last_slot = &slots[(end - 1) as usize];
                    unsafe { &mut *last_slot.get() }.next_free = local.next_free.get();
                    for (slot, next_id) in slots[start as usize..(end - 1) as usize]
                        .iter()
                        .zip(start + terminals + 1..)
                    {
                        unsafe { &mut *slot.get() }.next_free = next_id;
                    }
                    start + terminals
                } else {
                    local.next_free.get()
                };

                let mut shared = shared_state.lock();
                if next_free != 0 {
                    shared.next_free.push(next_free);
                }
                shared.node_count += local.node_count_delta.replace(0) as i64;
            });
        }

        LOCAL_STORE_STATE.with(|local| {
            if self.0.addr() == local.current_store.get()
                && (local.next_free.get() != 0
                    || local.initialized.get() % CHUNK_SIZE != 0
                    || local.node_count_delta.get() != 0)
            {
                drop_slow(&self.0.inner_nodes.slots, &self.0.state, TERMINALS as u32);
            }
        });
    }
}

// --- Manager Impls -----------------------------------------------------------

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> Manager<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    // Get a reference to the store
    fn store(&self) -> &Store<'id, N, ET, TM, R, MD, TERMINALS> {
        // We can simply get the store pointer by subtracting the offset of
        // `Manager` in `Store`. The only issue is that this violates Rust's
        // (proposed) aliasing rules. Hence, we only provide a hint that the
        // store's address can be computed without loading the value.
        let offset = const {
            std::mem::offset_of!(Store<'static, N, ET, TM, R, MD, TERMINALS>, manager)
                + RwLock::<Self>::DATA_OFFSET
        };
        // SAFETY: The resulting pointer is in bounds of the `Store` allocation.
        if unsafe { (self as *const Self as *const u8).sub(offset) } != self.store as *const u8 {
            // SAFETY: The pointers above are equal after initialization of `self.store`.
            unsafe { std::hint::unreachable_unchecked() };
        }

        // SAFETY: After initialization, `self.store` always points to the
        // containing `Store`.
        unsafe { &*self.store }
    }
}

unsafe impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> oxidd_core::Manager
    for Manager<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    R: oxidd_core::DiagramRules<Edge<'id, N, ET>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET>> + GCContainer<Self>,
{
    type Edge = Edge<'id, N, ET>;
    type EdgeTag = ET;
    type InnerNode = N;
    type Terminal = TM::TerminalNode;
    type TerminalRef<'a>
        = TM::TerminalNodeRef<'a>
    where
        Self: 'a;
    type Rules = R;

    type TerminalIterator<'a>
        = TM::Iterator<'a>
    where
        Self: 'a;
    type NodeSet = NodeSet;
    type LevelView<'a>
        = LevelView<'a, 'id, N, ET, TM, R, MD, TERMINALS>
    where
        Self: 'a;
    type LevelIterator<'a>
        = LevelIter<'a, 'id, N, ET, TM, R, MD, TERMINALS>
    where
        Self: 'a;

    #[inline]
    fn get_node(&self, edge: &Self::Edge) -> oxidd_core::Node<Self> {
        let store = self.store();
        let id = edge.node_id();
        if id >= TERMINALS {
            oxidd_core::Node::Inner(store.inner_nodes.inner_node(edge))
        } else {
            // SAFETY: `id` is a valid terminal ID
            oxidd_core::Node::Terminal(unsafe { store.terminal_manager.get_terminal(id) })
        }
    }

    #[inline]
    fn clone_edge(&self, edge: &Self::Edge) -> Self::Edge {
        self.store().clone_edge(edge)
    }

    #[inline]
    fn drop_edge(&self, edge: Self::Edge) {
        self.store().drop_edge(edge)
    }

    #[inline]
    fn num_inner_nodes(&self) -> usize {
        self.unique_table
            .iter()
            .map(|level| level.lock().len())
            .sum()
    }

    #[inline]
    fn approx_num_inner_nodes(&self) -> usize {
        let count = self.store().state.lock().node_count;
        if count < 0 {
            0
        } else {
            count as u64 as usize
        }
    }

    #[inline]
    fn num_levels(&self) -> LevelNo {
        self.unique_table.len() as LevelNo
    }

    #[inline]
    fn add_level(&mut self, f: impl FnOnce(LevelNo) -> Self::InnerNode) -> AllocResult<Self::Edge> {
        let store = self.store();
        let no = self.unique_table.len() as LevelNo;
        assert!(no != LevelNo::MAX, "Too many levels");
        let [e1, e2] = store.add_node(f(no))?;
        let mut set: LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS> = Default::default();
        set.insert(&store.inner_nodes, e1);
        self.unique_table.push(Mutex::new(set));
        Ok(e2)
    }

    #[inline(always)]
    fn level(&self, no: LevelNo) -> Self::LevelView<'_> {
        LevelView {
            store: self.store(),
            level: no,
            set: self.unique_table[no as usize].lock(),
        }
    }

    #[inline]
    fn levels(&self) -> Self::LevelIterator<'_> {
        LevelIter {
            store: self.store(),
            level_front: 0,
            level_back: self.unique_table.len() as LevelNo,
            it: self.unique_table.iter(),
        }
    }

    #[inline]
    fn get_terminal(&self, terminal: Self::Terminal) -> AllocResult<Self::Edge> {
        self.store().terminal_manager.get_edge(terminal)
    }

    #[inline]
    fn num_terminals(&self) -> usize {
        self.store().terminal_manager.len()
    }

    #[inline]
    fn terminals(&self) -> Self::TerminalIterator<'_> {
        self.store().terminal_manager.iter()
    }

    fn gc(&self) -> usize {
        if !self.gc_ongoing.try_lock() {
            // We don't want two concurrent garbage collections
            return 0;
        }
        let guard = AbortOnDrop("Garbage collection panicked.");

        #[cfg(feature = "statistics")]
        eprintln!(
            "[oxidd-manager-index] garbage collection started at {}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        self.data.pre_gc(self);

        let store = self.store();
        let mut collected = 0;
        for level in &self.unique_table {
            let mut level = level.lock();
            collected += level.len() as u32;
            // SAFETY: We prepared the garbage collection, hence there are no
            // "weak" edges.
            unsafe { level.gc(store) };
            collected -= level.len() as u32;
        }
        collected += store.terminal_manager.gc();

        // SAFETY: We called `pre_gc`, the garbage collection is done.
        unsafe { self.data.post_gc(self) };
        self.gc_ongoing.unlock();
        guard.defuse();

        #[cfg(feature = "statistics")]
        eprintln!(
            "[oxidd-manager-index] garbage collection finished at {}: removed {} nodes",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            collected,
        );
        collected as usize
    }

    fn reorder<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let guard = AbortOnDrop("Reordering panicked.");
        self.data.pre_gc(self);
        let res = f(self);
        // SAFETY: We called `pre_gc`, the reordering is done.
        unsafe { self.data.post_gc(self) };
        guard.defuse();
        self.reorder_count += 1;
        res
    }

    #[inline]
    fn reorder_count(&self) -> u64 {
        self.reorder_count
    }
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> oxidd_core::HasWorkers
    for Manager<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>> + Send + Sync,
    ET: Tag + Send + Sync,
    TM: TerminalManager<'id, N, ET, TERMINALS> + Send + Sync,
    R: oxidd_core::DiagramRules<Edge<'id, N, ET>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET>> + GCContainer<Self> + Send + Sync,
{
    type WorkerPool = crate::workers::Workers;

    #[inline]
    fn workers(&self) -> &Self::WorkerPool {
        &self.store().workers
    }
}

// === Unique Table ============================================================

/// The underlying data structure for level views
///
/// Conceptually, every [`LevelViewSet`] is tied to a [`Manager`]. It is a hash
/// table ensuring the uniqueness of nodes. However, it does not store nodes
/// internally, but edges referencing nodes. These edges are always untagged and
/// reference inner nodes.
///
/// Because a [`LevelViewSet`] on its own is not sufficient to drop nodes
/// accordingly, this will simply leak all contained edges, not calling the
/// `Edge`'s `Drop` implementation.
struct LevelViewSet<'id, N, ET, TM, R, MD, const TERMINALS: usize>(
    RawTable<Edge<'id, N, ET>, u32>,
    PhantomData<(TM, R, MD)>,
);

#[inline]
fn hash_node<N: NodeBase>(node: &N) -> u64 {
    let mut hasher = FxHasher::default();
    node.hash(&mut hasher);
    hasher.finish()
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    /// Get the number of nodes on this level
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Get an equality function for entries
    ///
    /// SAFETY: The returned function must be called on untagged edges
    /// referencing inner nodes only.
    #[inline]
    unsafe fn eq<'a>(
        nodes: &'a SlotSlice<'id, N, TERMINALS>,
        node: &'a N,
    ) -> impl Fn(&Edge<'id, N, ET>) -> bool + 'a {
        move |edge| unsafe { nodes.inner_node_unchecked(edge) == node }
    }

    /// Reserve space for `additional` nodes on this level
    #[inline]
    fn reserve(&mut self, additional: usize) {
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.reserve(additional)
    }

    #[inline]
    fn get(&self, nodes: &SlotSlice<'id, N, TERMINALS>, node: &N) -> Option<&Edge<'id, N, ET>> {
        let hash = hash_node(node);
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.get(hash, unsafe { Self::eq(nodes, node) })
    }

    /// Insert the given edge, assuming that the referenced node is already
    /// stored in `nodes`.
    ///
    /// Returns `true` if the edge was inserted, `false` if it was already
    /// present.
    ///
    /// Panics if `edge` points to a terminal node. May furthermore panic if
    /// `edge` is tagged, depending on the configuration.
    #[inline]
    fn insert(&mut self, nodes: &SlotSlice<'id, N, TERMINALS>, edge: Edge<'id, N, ET>) -> bool {
        debug_assert!(!edge.is_tagged(), "`edge` must not be tagged");
        let node = nodes.inner_node(&edge);
        let hash = hash_node(node);
        // SAFETY (next 2): The hash table only contains untagged edges
        // referencing inner nodes.
        match self
            .0
            .find_or_find_insert_slot(hash, unsafe { Self::eq(nodes, node) })
        {
            Ok(_) => {
                // We need to drop `edge`. This simply amounts to decrementing
                // the reference counter, since there is still one edge
                // referencing `node` stored in here.
                std::mem::forget(edge);
                // SAFETY: We forgot `edge`.
                let _old_rc = unsafe { node.release() };
                debug_assert!(_old_rc > 1);

                false
            }
            Err(slot) => {
                // SAFETY: `slot` was returned by `find_or_find_insert_slot`.
                // We have exclusive access to the hash table and did not modify
                // it in between.
                unsafe { self.0.insert_in_slot_unchecked(hash, slot, edge) };
                true
            }
        }
    }

    /// Get an edge for `node`
    ///
    /// If `node` is not yet present in the hash table, `insert` is called to
    /// insert the node into `nodes`.
    #[inline]
    fn get_or_insert(
        &mut self,
        nodes: &SlotSlice<'id, N, TERMINALS>,
        node: N,
        insert: impl FnOnce(N) -> AllocResult<[Edge<'id, N, ET>; 2]>,
        drop: impl FnOnce(N),
    ) -> AllocResult<Edge<'id, N, ET>> {
        let hash = hash_node(&node);
        // SAFETY (next 2): The hash table only contains untagged edges
        // referencing inner nodes.
        match self
            .0
            .find_or_find_insert_slot(hash, unsafe { Self::eq(nodes, &node) })
        {
            Ok(slot) => {
                drop(node);
                // SAFETY:
                // - `slot` was returned by `find_or_find_insert_slot`. We have exclusive access
                //   to the hash table and did not modify it in between.
                // - All edges in the table are untagged and refer to inner nodes.
                Ok(unsafe { nodes.clone_edge_unchecked(self.0.get_at_slot_unchecked(slot)) })
            }
            Err(slot) => {
                let [e1, e2] = insert(node)?;
                // SAFETY: `slot` was returned by `find_or_find_insert_slot`.
                // We have exclusive access to the hash table and did not modify
                // it in between.
                unsafe { self.0.insert_in_slot_unchecked(hash, slot, e1) };
                Ok(e2)
            }
        }
    }

    /// Perform garbage collection, i.e. remove all nodes without references
    /// besides the internal edge
    ///
    /// SAFETY: There must not be any "weak" edges, i.e. edges where the
    /// reference count is not materialized (apply cache implementations exploit
    /// this).
    unsafe fn gc(&mut self, store: &Store<'id, N, ET, TM, R, MD, TERMINALS>) {
        let inner_nodes = &*store.inner_nodes;
        self.0.retain(
            |edge| {
                // SAFETY: All edges in unique tables are untagged and point to
                // inner nodes.
                unsafe { inner_nodes.inner_node_unchecked(edge) }.load_rc(Acquire) != 1
            },
            |edge| {
                // SAFETY (next 2): `edge` is untagged and points to an inner node
                let slot_ptr = unsafe { inner_nodes.slot_pointer_unchecked(&edge) };
                let id = unsafe { edge.node_id_unchecked() };
                std::mem::forget(edge);

                // SAFETY: Since `rc` is 1, this is the last reference. We use
                // `Acquire` order above and `Release` order when decrementing
                // reference counters, so we have exclusive node access now. It
                // contains a node. `id` is the ID of the slot.
                unsafe { store.free_slot(&mut *slot_ptr, id) };
            },
        );
    }

    /// Remove `node`
    ///
    /// Returns `Some(edge)` if `node` was present, `None` otherwise
    ///
    /// SAFETY: There must not be any "weak" edges, i.e. edges where the
    /// reference count is not materialized (apply cache implementations exploit
    /// this).
    #[inline]
    unsafe fn remove(
        &mut self,
        nodes: &SlotSlice<'id, N, TERMINALS>,
        node: &N,
    ) -> Option<Edge<'id, N, ET>> {
        let hash = hash_node(node);
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.remove_entry(hash, unsafe { Self::eq(nodes, node) })
    }

    /// Iterate over all edges pointing to nodes in the set
    #[inline]
    fn iter(&self) -> LevelViewIter<'_, 'id, N, ET> {
        LevelViewIter(self.0.iter())
    }

    /// Iterator that consumes all [`Edge`]s in the set
    #[inline]
    fn drain(&mut self) -> linear_hashtbl::raw::Drain<Edge<'id, N, ET>, u32> {
        self.0.drain()
    }
}

impl<N, ET, TM, R, MD, const TERMINALS: usize> Drop
    for LevelViewSet<'_, N, ET, TM, R, MD, TERMINALS>
{
    #[inline]
    fn drop(&mut self) {
        // If the nodes need drop, this is handled by the `Manager` `Drop` impl
        // or the `TakenLevelView` `Drop` impl, respectively.
        self.0.reset_no_drop();
    }
}

impl<N, ET, TM, R, MD, const TERMINALS: usize> Default
    for LevelViewSet<'_, N, ET, TM, R, MD, TERMINALS>
{
    #[inline]
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> IntoIterator
    for LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>
{
    type Item = Edge<'id, N, ET>;

    type IntoIter = linear_hashtbl::raw::IntoIter<Edge<'id, N, ET>, u32>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        // SAFETY: We move out of `this` (and forget `this`)
        let set = unsafe { std::ptr::read(&this.0) };
        set.into_iter()
    }
}

pub struct LevelViewIter<'a, 'id, N, ET>(linear_hashtbl::raw::Iter<'a, Edge<'id, N, ET>, u32>);

impl<'a, 'id, N, ET> Iterator for LevelViewIter<'a, 'id, N, ET> {
    type Item = &'a Edge<'id, N, ET>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<N, ET> ExactSizeIterator for LevelViewIter<'_, '_, N, ET> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, 'id, N, ET> FusedIterator for LevelViewIter<'a, 'id, N, ET> where
    linear_hashtbl::raw::Iter<'a, Edge<'id, N, ET>, u32>: FusedIterator
{
}

pub struct LevelView<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    store: &'a Store<'id, N, ET, TM, R, MD, TERMINALS>,
    level: LevelNo,
    set: MutexGuard<'a, LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>>,
}

unsafe impl<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize>
    oxidd_core::LevelView<Edge<'id, N, ET>, N> for LevelView<'a, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    type Iterator<'b>
        = LevelViewIter<'b, 'id, N, ET>
    where
        Self: 'b,
        Edge<'id, N, ET>: 'b;

    type Taken = TakenLevelView<'a, 'id, N, ET, TM, R, MD, TERMINALS>;

    #[inline(always)]
    fn len(&self) -> usize {
        self.set.len()
    }

    #[inline(always)]
    fn level_no(&self) -> LevelNo {
        self.level
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.set.reserve(additional);
    }

    #[inline]
    fn get(&self, node: &N) -> Option<&Edge<'id, N, ET>> {
        self.set.get(&self.store.inner_nodes, node)
    }

    #[inline]
    fn insert(&mut self, edge: Edge<'id, N, ET>) -> bool {
        debug_assert!(!edge.is_tagged(), "`edge` should not be tagged");
        // No need to check if the node referenced by `edge` is stored in
        // `self.store` due to lifetime restrictions.
        let nodes = &self.store.inner_nodes;
        nodes.inner_node(&edge).assert_level_matches(self.level);
        self.set.insert(nodes, edge)
    }

    #[inline(always)]
    fn get_or_insert(&mut self, node: N) -> AllocResult<Edge<'id, N, ET>> {
        node.assert_level_matches(self.level);
        // No need to check if the children of `node` are stored in `self.store`
        // due to lifetime restrictions.
        self.set.get_or_insert(
            &self.store.inner_nodes,
            node,
            |node| self.store.add_node(node),
            |node| node.drop_with(|edge| self.store.drop_edge(edge)),
        )
    }

    #[inline]
    unsafe fn gc(&mut self) {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        unsafe { self.set.gc(self.store) };
    }

    #[inline]
    unsafe fn remove(&mut self, node: &N) -> bool {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        match unsafe { self.set.remove(&self.store.inner_nodes, node) } {
            Some(edge) => {
                // SAFETY: `edge` is untagged and points to an inner node
                unsafe { self.store.drop_unique_table_edge(edge) };
                true
            }
            None => false,
        }
    }

    #[inline(always)]
    unsafe fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.set, &mut other.set);
    }

    #[inline]
    fn iter(&self) -> Self::Iterator<'_> {
        self.set.iter()
    }

    #[inline(always)]
    fn take(&mut self) -> Self::Taken {
        TakenLevelView {
            store: self.store,
            level: self.level,
            set: std::mem::take(&mut self.set),
        }
    }
}

pub struct TakenLevelView<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    store: &'a Store<'id, N, ET, TM, R, MD, TERMINALS>,
    level: LevelNo,
    set: LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>,
}

unsafe impl<'id, N, ET, TM, R, MD, const TERMINALS: usize>
    oxidd_core::LevelView<Edge<'id, N, ET>, N>
    for TakenLevelView<'_, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    type Iterator<'b>
        = LevelViewIter<'b, 'id, N, ET>
    where
        Self: 'b,
        Edge<'id, N, ET>: 'b;

    type Taken = Self;

    #[inline(always)]
    fn len(&self) -> usize {
        self.set.len()
    }

    #[inline(always)]
    fn level_no(&self) -> LevelNo {
        self.level
    }

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.set.reserve(additional);
    }

    #[inline]
    fn get(&self, node: &N) -> Option<&Edge<'id, N, ET>> {
        self.set.get(&self.store.inner_nodes, node)
    }

    #[inline]
    fn insert(&mut self, edge: Edge<'id, N, ET>) -> bool {
        debug_assert!(!edge.is_tagged(), "`edge` should not be tagged");
        // No need to check if the node referenced by `edge` is stored in
        // `self.store` due to lifetime restrictions.
        let nodes = &self.store.inner_nodes;
        nodes.inner_node(&edge).assert_level_matches(self.level);
        self.set.insert(nodes, edge)
    }

    #[inline(always)]
    fn get_or_insert(&mut self, node: N) -> AllocResult<Edge<'id, N, ET>> {
        node.assert_level_matches(self.level);
        // No need to check if the children of `node` are stored in `self.store`
        // due to lifetime restrictions.
        self.set.get_or_insert(
            &self.store.inner_nodes,
            node,
            |node| self.store.add_node(node),
            |node| node.drop_with(|edge| self.store.drop_edge(edge)),
        )
    }

    #[inline]
    unsafe fn gc(&mut self) {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        unsafe { self.set.gc(self.store) };
    }

    #[inline]
    unsafe fn remove(&mut self, node: &N) -> bool {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        match unsafe { self.set.remove(&self.store.inner_nodes, node) } {
            Some(edge) => {
                // SAFETY: `edge` is untagged and points to an inner node
                unsafe { self.store.drop_unique_table_edge(edge) };
                true
            }
            None => false,
        }
    }

    #[inline(always)]
    unsafe fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.set, &mut other.set);
    }

    #[inline]
    fn iter(&self) -> Self::Iterator<'_> {
        self.set.iter()
    }

    #[inline(always)]
    fn take(&mut self) -> Self::Taken {
        Self {
            store: self.store,
            level: self.level,
            set: std::mem::take(&mut self.set),
        }
    }
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> Drop
    for TakenLevelView<'_, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    fn drop(&mut self) {
        for edge in self.set.drain() {
            // SAFETY: Edges in unique tables are untagged and point to inner
            // nodes.
            unsafe { self.store.drop_unique_table_edge(edge) }
        }
    }
}

// --- LevelIterator -----------------------------------------------------------

pub struct LevelIter<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    store: &'a Store<'id, N, ET, TM, R, MD, TERMINALS>,
    level_front: LevelNo,
    level_back: LevelNo,
    it: std::slice::Iter<'a, Mutex<LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>>>,
}

impl<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize> Iterator
    for LevelIter<'a, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    type Item = LevelView<'a, 'id, N, ET, TM, R, MD, TERMINALS>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.it.next() {
            Some(mutex) => {
                let level = self.level_front;
                self.level_front += 1;
                Some(LevelView {
                    store: self.store,
                    level,
                    set: mutex.lock(),
                })
            }
            None => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> ExactSizeIterator
    for LevelIter<'_, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.it.len()
    }
}
impl<'a, 'id, N, ET, TM, R, MD, const TERMINALS: usize> FusedIterator
    for LevelIter<'a, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
    std::slice::Iter<'a, Mutex<LevelViewSet<'id, N, ET, TM, R, MD, TERMINALS>>>: FusedIterator,
{
}

impl<'id, N, ET, TM, R, MD, const TERMINALS: usize> DoubleEndedIterator
    for LevelIter<'_, 'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            Some(mutex) => {
                self.level_back -= 1;
                Some(LevelView {
                    store: self.store,
                    level: self.level_back,
                    set: mutex.lock(),
                })
            }
            None => None,
        }
    }
}

// === ManagerRef ==============================================================

#[repr(transparent)]
pub struct ManagerRef<
    NC: InnerNodeCons<ET>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, TERMINALS>,
    RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
    MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
    const TERMINALS: usize,
>(
    Arc<
        Store<
            'static,
            NC::T<'static>,
            ET,
            TMC::T<'static>,
            RC::T<'static>,
            MDC::T<'static>,
            TERMINALS,
        >,
    >,
);

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Drop for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 2 {
            // This is the second last reference. The last reference belongs to
            // the gc thread. Terminate it.
            let gc_signal = &self.0.gc_signal;
            *gc_signal.0.lock() = GCSignal::Quit;
            gc_signal.1.notify_one();
        }
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    /// Convert `self` into a raw pointer, e.g. for usage in a foreign function
    /// interface.
    ///
    /// This method does not change any reference counters. To avoid a memory
    /// leak, use [`Self::from_raw()`] to convert the pointer back into a
    /// `ManagerRef`.
    #[inline(always)]
    pub fn into_raw(self) -> *const std::ffi::c_void {
        let this = ManuallyDrop::new(self);
        // SAFETY: we move out of `this`
        Arc::into_raw(unsafe { std::ptr::read(&this.0) }) as _
    }

    /// Convert `raw` into a `ManagerRef`
    ///
    /// # Safety
    ///
    /// `raw` must have been obtained via [`Self::into_raw()`]. This function
    /// does not change any reference counters, so calling this function
    /// multiple times for the same pointer may lead to use after free bugs
    /// depending on the usage of the returned `ManagerRef`.
    #[inline(always)]
    pub unsafe fn from_raw(raw: *const std::ffi::c_void) -> Self {
        // SAFETY: Invariants are upheld by the caller.
        Self(unsafe { Arc::from_raw(raw as _) })
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Clone for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > PartialEq for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Eq for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > PartialOrd for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Ord for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let a = &*self.0 as *const Store<_, _, _, _, _, TERMINALS>;
        let b = &*other.0 as *const _;
        a.cmp(&b)
    }
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Hash for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ptr = &*self.0 as *const Store<_, _, _, _, _, TERMINALS>;
        ptr.hash(state);
    }
}

impl<
        'a,
        'id,
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > From<&'a M<'id, NC, ET, TMC, RC, MDC, TERMINALS>>
    for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    fn from(manager: &'a M<'id, NC, ET, TMC, RC, MDC, TERMINALS>) -> Self {
        let ptr = manager as *const _ as *const M<'static, NC, ET, TMC, RC, MDC, TERMINALS>;
        // SAFETY:
        // - We just changed "identifier" lifetimes.
        // - The pointer was obtained via `Arc::into_raw()`, and since we have a
        //   `&Manager` reference, the counter is at least 1.
        unsafe {
            let manager = &*ptr;
            Arc::increment_strong_count(manager.store);
            ManagerRef(Arc::from_raw(manager.store))
        }
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > oxidd_core::ManagerRef for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    type Manager<'id> = M<'id, NC, ET, TMC, RC, MDC, TERMINALS>;

    fn with_manager_shared<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&Self::Manager<'id>) -> T,
    {
        let local_guard = self.0.prepare_local_state();
        let res = f(&self.0.manager.shared());
        drop(local_guard);
        res
    }

    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>) -> T,
    {
        let local_guard = self.0.prepare_local_state();
        let res = f(&mut self.0.manager.exclusive());
        drop(local_guard);
        res
    }
}

/// Create a new manager
pub fn new_manager<
    NC: InnerNodeCons<ET> + 'static,
    ET: Tag + Send + Sync + 'static,
    TMC: TerminalManagerCons<NC, ET, TERMINALS> + 'static,
    RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS> + 'static,
    MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS> + 'static,
    const TERMINALS: usize,
>(
    inner_node_capacity: u32,
    terminal_node_capacity: u32,
    threads: u32,
    data: MDC::T<'static>,
) -> ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS> {
    // Evaluate a few constants for assertions
    let _ = Store::<
        'static,
        NC::T<'static>,
        ET,
        TMC::T<'static>,
        RC::T<'static>,
        MDC::T<'static>,
        TERMINALS,
    >::CHECK_TERMINALS;

    let max_inner_capacity = if usize::BITS > u32::BITS {
        ((1usize << (u32::BITS - Edge::<NC::T<'static>, ET>::TAG_BITS)) - TERMINALS) as u32
    } else {
        // 32 bit address space. Every node consumes 16 byte plus at least
        // 4 byte in the unique table. Furthermore, there is the apply cache.
        // If we try to reserve space for more than `u32::MAX / 24` nodes, we
        // will most likely run out of memory.
        u32::MAX / 24
    };
    let inner_node_capacity = std::cmp::min(inner_node_capacity, max_inner_capacity);

    let gc_lwm = inner_node_capacity / 100 * 90;
    let gc_hwm = inner_node_capacity / 100 * 95;

    let arc = Arc::new(Store {
        inner_nodes: SlotSlice::new_boxed(inner_node_capacity),
        state: CachePadded::new(Mutex::new(SharedStoreState {
            next_free: Vec::new(),
            allocated: 0,
            node_count: 0,
            gc_state: if gc_lwm < gc_hwm {
                GCState::Init
            } else {
                GCState::Disabled
            },
            gc_lwm,
            gc_hwm,
        })),
        manager: RwLock::new(Manager {
            unique_table: Vec::new(),
            data: ManuallyDrop::new(data),
            store: std::ptr::null(),
            reorder_count: 0,
            gc_ongoing: TryLock::new(),
        }),
        terminal_manager: TMC::T::<'static>::with_capacity(terminal_node_capacity),
        gc_signal: (Mutex::new(GCSignal::RunGc), Condvar::new()),
        workers: crate::workers::Workers::new(threads),
    });

    let mut manager = arc.manager.exclusive();
    manager.store = Arc::as_ptr(&arc);
    drop(manager);

    let store_addr = arc.addr();
    arc.workers.pool.spawn_broadcast(move |_| {
        // The workers are dedicated to this store.
        LOCAL_STORE_STATE.with(|state| state.current_store.set(store_addr))
    });

    // spell-checker:ignore mref
    let gc_mref: ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS> = ManagerRef(arc.clone());
    std::thread::Builder::new()
        .name("oxidd mi gc".to_string())
        .spawn(move || {
            // The worker is dedicated to this store.
            LOCAL_STORE_STATE.with(|state| state.current_store.set(store_addr));

            let store = &*gc_mref.0;
            loop {
                let mut lock = store.gc_signal.0.lock();
                store.gc_signal.1.wait(&mut lock);
                if *lock == GCSignal::Quit {
                    break;
                }
                drop(lock);

                // parking_lot `Condvar`s have no spurious wakeups -> run gc now
                oxidd_core::ManagerRef::with_manager_shared(&gc_mref, |manager| {
                    oxidd_core::Manager::gc(manager);
                });

                let mut shared = store.state.lock();
                LOCAL_STORE_STATE.with(|local| {
                    if local.next_free.get() != 0 {
                        shared.node_count += local.node_count_delta.replace(0) as i64;
                        shared.next_free.push(local.next_free.replace(0));
                    }
                });

                if shared.node_count < shared.gc_lwm as i64 && shared.gc_state != GCState::Disabled
                {
                    shared.gc_state = GCState::Init;
                }
            }
        })
        .unwrap();

    ManagerRef(arc)
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag + Send + Sync,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > oxidd_core::HasWorkers for ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>
{
    type WorkerPool = crate::workers::Workers;

    #[inline]
    fn workers(&self) -> &Self::WorkerPool {
        &self.0.workers
    }
}

// === Function ================================================================

#[repr(C)]
pub struct Function<
    NC: InnerNodeCons<ET>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, TERMINALS>,
    RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
    MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
    const TERMINALS: usize,
> {
    store: ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>,
    edge: ManuallyDrop<Edge<'static, NC::T<'static>, ET>>,
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    /// Convert `self` into a raw pointer and edge value, e.g. for usage in a
    /// foreign function interface.
    ///
    /// This method does not change any reference counters. To avoid a memory
    /// leak, use [`Self::from_raw()`] to convert pointer and edge value back
    /// into a `Function`.
    #[inline(always)]
    pub fn into_raw(self) -> (*const std::ffi::c_void, u32) {
        let this = ManuallyDrop::new(self);
        // SAFETY: We forget `this`
        (
            unsafe { std::ptr::read(&this.store) }.into_raw(),
            this.edge.0,
        )
    }

    /// Convert `ptr` and `edge_val` into a `Function`
    ///
    /// # Safety
    ///
    /// `raw` and `edge_val` must have been obtained via [`Self::into_raw()`].
    /// This function does not change any reference counters, so calling this
    /// function multiple times for the same pointer may lead to use after free
    /// bugs depending on the usage of the returned `Function`.
    #[inline(always)]
    pub unsafe fn from_raw(ptr: *const std::ffi::c_void, edge_val: u32) -> Self {
        // SAFETY: Invariants are upheld by the caller.
        Self {
            store: unsafe { ManagerRef::from_raw(ptr) },
            edge: ManuallyDrop::new(Edge(edge_val, PhantomData)),
        }
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Drop for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self.edge` is never used again.
        let edge = unsafe { ManuallyDrop::take(&mut self.edge) };
        self.store.0.drop_edge(edge);
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Clone for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            edge: ManuallyDrop::new(self.store.0.clone_edge(&self.edge)),
        }
    }
}

impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > PartialEq for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.store == other.store && self.edge == other.edge
    }
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Eq for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > PartialOrd for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Ord for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.store
            .cmp(&other.store)
            .then(self.edge.cmp(&other.edge))
    }
}
impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > Hash for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.store.hash(state);
        self.edge.hash(state);
    }
}

unsafe impl<
        NC: InnerNodeCons<ET>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, TERMINALS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > oxidd_core::function::Function for Function<NC, ET, TMC, RC, MDC, TERMINALS>
{
    type Manager<'id> = M<'id, NC, ET, TMC, RC, MDC, TERMINALS>;

    type ManagerRef = ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>;

    #[inline]
    fn from_edge<'id>(manager: &Self::Manager<'id>, edge: EdgeOfFunc<'id, Self>) -> Self {
        #[allow(clippy::unnecessary_cast)] // this cast is necessary
        let ptr = manager as *const Self::Manager<'id> as *const Self::Manager<'static>;
        // SAFETY:
        // - We just changed "identifier" lifetimes.
        // - The pointer was obtained via `Arc::into_raw()`, and since we have a
        //   `&Manager` reference, the counter is at least 1.
        let store = unsafe {
            let manager = &*ptr;
            Arc::increment_strong_count(manager.store);
            ManagerRef(Arc::from_raw(manager.store))
        };
        // Avoid transmuting `edge` for changing lifetimes
        let id = edge.0;
        std::mem::forget(edge);
        Self {
            store,
            edge: ManuallyDrop::new(Edge(id, PhantomData)),
        }
    }

    #[inline]
    fn as_edge<'id>(&self, manager: &Self::Manager<'id>) -> &EdgeOfFunc<'id, Self> {
        assert!(
            crate::util::ptr_eq_untyped(self.store.0.manager.data_ptr(), manager),
            "This function does not belong to `manager`"
        );

        let ptr = &*self.edge as *const EdgeOfFunc<'static, Self> as *const EdgeOfFunc<'id, Self>;
        // SAFETY: Just changing lifetimes; we checked that `self.edge` belongs
        // to `manager` above.
        unsafe { &*ptr }
    }

    #[inline]
    fn into_edge<'id>(self, manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        assert!(
            crate::util::ptr_eq_untyped(self.store.0.manager.data_ptr(), manager),
            "This function does not belong to `manager`"
        );
        // We only want to drop the store ref (but `Function` implements `Drop`,
        // so we cannot destruct `self`).
        let mut this = ManuallyDrop::new(self);
        let edge = Edge(this.edge.0, PhantomData);
        // SAFETY: we forget `self`
        unsafe { std::ptr::drop_in_place(&mut this.store) };
        edge
    }

    #[inline]
    fn manager_ref(&self) -> Self::ManagerRef {
        self.store.clone()
    }

    fn with_manager_shared<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> T,
    {
        let local_guard = self.store.0.prepare_local_state();
        let res = f(&self.store.0.manager.shared(), &self.edge);
        drop(local_guard);
        res
    }

    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> T,
    {
        let local_guard = self.store.0.prepare_local_state();
        let res = f(&mut self.store.0.manager.exclusive(), &self.edge);
        drop(local_guard);
        res
    }
}

// === NodeSet =================================================================

/// Node set implementation using a [bit vector][BitVec]
///
/// Since nodes are stored in an array, we can use a single bit vector. This
/// reduces space consumption dramatically and increases the performance.
#[derive(Default, Clone)]
pub struct NodeSet {
    len: usize,
    data: BitVec,
}

impl PartialEq for NodeSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data == other.data
    }
}
impl Eq for NodeSet {}

impl<'id, N: NodeBase, ET: Tag> oxidd_core::util::NodeSet<Edge<'id, N, ET>> for NodeSet {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn insert(&mut self, edge: &Edge<'id, N, ET>) -> bool {
        let index = edge.node_id();
        if index < self.data.len() {
            if self.data[index] {
                return false;
            }
        } else {
            self.data.resize((index + 1).next_power_of_two(), false);
        }
        self.data.set(index, true);
        self.len += 1;
        true
    }

    #[inline]
    fn contains(&self, edge: &Edge<'id, N, ET>) -> bool {
        let index = edge.node_id();
        if index < self.data.len() {
            self.data[index]
        } else {
            false
        }
    }

    #[inline]
    fn remove(&mut self, edge: &Edge<'id, N, ET>) -> bool {
        let index = edge.node_id();
        if index < self.data.len() && self.data[index] {
            self.len -= 1;
            self.data.set(index, false);
            true
        } else {
            false
        }
    }
}

// === Additional Trait Implementations ========================================

impl<
        'id,
        N: NodeBase + InnerNode<Edge<'id, N, ET>>,
        ET: Tag,
        TM: TerminalManager<'id, N, ET, TERMINALS>,
        R: DiagramRules<Edge<'id, N, ET>, N, TM::TerminalNode>,
        MD: oxidd_core::HasApplyCache<Self, O> + GCContainer<Self> + DropWith<Edge<'id, N, ET>>,
        O: Copy,
        const TERMINALS: usize,
    > oxidd_core::HasApplyCache<Self, O> for Manager<'id, N, ET, TM, R, MD, TERMINALS>
{
    type ApplyCache = MD::ApplyCache;

    #[inline]
    fn apply_cache(&self) -> &Self::ApplyCache {
        self.data.apply_cache()
    }

    #[inline]
    fn apply_cache_mut(&mut self) -> &mut Self::ApplyCache {
        self.data.apply_cache_mut()
    }
}

impl<'id, T, N, ET, TM, R, MD, const TERMINALS: usize> AsRef<T>
    for Manager<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>> + AsRef<T>,
{
    #[inline(always)]
    fn as_ref(&self) -> &T {
        self.data.as_ref()
    }
}

impl<'id, T, N, ET, TM, R, MD, const TERMINALS: usize> AsMut<T>
    for Manager<'id, N, ET, TM, R, MD, TERMINALS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, TERMINALS>,
    MD: DropWith<Edge<'id, N, ET>> + AsMut<T>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T {
        self.data.as_mut()
    }
}
