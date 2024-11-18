//! Pointer-Based Manager Implementation
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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::{align_of, ManuallyDrop};
use std::ptr::{addr_of, addr_of_mut, NonNull};

use arcslab::{ArcSlab, ArcSlabRef, AtomicRefCounted, ExtHandle, IntHandle};
use bitvec::bitvec;
use bitvec::vec::BitVec;
use linear_hashtbl::raw::RawTable;
use parking_lot::Mutex;
use parking_lot::MutexGuard;
use rustc_hash::FxHasher;

use oxidd_core::function::EdgeOfFunc;
use oxidd_core::util::{AbortOnDrop, AllocResult, Borrowed, DropWith, GCContainer};
use oxidd_core::{DiagramRules, HasApplyCache, InnerNode, LevelNo, Node, Tag};

use crate::node::NodeBase;
use crate::terminal_manager::TerminalManager;
use crate::util;
use crate::util::rwlock::RwLock;
use crate::util::{Invariant, TryLock};

// === Type Constructors =======================================================

/// Inner node type constructor
pub trait InnerNodeCons<ET: Tag, const TAG_BITS: u32> {
    type T<'id>: NodeBase + InnerNode<Edge<'id, Self::T<'id>, ET, TAG_BITS>>;
}

/// Terminal manager type constructor
pub trait TerminalManagerCons<
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    RC: DiagramRulesCons<NC, ET, Self, MDC, PAGE_SIZE, TAG_BITS>,
    MDC: ManagerDataCons<NC, ET, Self, RC, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>: Sized
{
    type TerminalNode;
    type T<'id>: TerminalManager<
        'id,
        NC::T<'id>,
        ET,
        MDC::T<'id>,
        PAGE_SIZE,
        TAG_BITS,
        TerminalNode = Self::TerminalNode,
    >;
}

/// Diagram rules type constructor
pub trait DiagramRulesCons<
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, Self, MDC, PAGE_SIZE, TAG_BITS>,
    MDC: ManagerDataCons<NC, ET, TMC, Self, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>: Sized
{
    type T<'id>: DiagramRules<
        Edge<'id, NC::T<'id>, ET, TAG_BITS>,
        NC::T<'id>,
        <TMC::T<'id> as TerminalManager<'id, NC::T<'id>, ET, MDC::T<'id>, PAGE_SIZE, TAG_BITS>>::TerminalNode,
    >;
}

/// Manager data type constructor
pub trait ManagerDataCons<
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, RC, Self, PAGE_SIZE, TAG_BITS>,
    RC: DiagramRulesCons<NC, ET, TMC, Self, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>: Sized
{
    type T<'id>: DropWith<Edge<'id, NC::T<'id>, ET, TAG_BITS>>
        + GCContainer<
            Manager<
                'id,
                NC::T<'id>,
                ET,
                TMC::T<'id>,
                RC::T<'id>,
                Self::T<'id>,
                PAGE_SIZE,
                TAG_BITS,
            >,
        >;
}

// === Manager & Edges =========================================================

pub struct StoreInner<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    manager: RwLock<Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>,
    terminal_manager: TM,
    workers: crate::workers::Workers,
}

#[repr(transparent)]
#[must_use]
pub struct Edge<'id, N, ET, const TAG_BITS: u32>(
    /// Points to an `InnerNode` (if `ptr & (1 << TAG_BITS) == 0`) or a terminal
    /// node (`ptr & (1 << TAG_BITS) == 1`)
    ///
    /// SAFETY invariant: The pointer (as integer) is `>= 1 << (TAG_BITS + 1)`
    /// (i.e. the pointer with all tags removed is still non-null)
    NonNull<()>,
    PhantomData<(Invariant<'id>, N, ET)>,
);

unsafe impl<N: Send + Sync, ET: Send + Sync, const TAG_BITS: u32> Send
    for Edge<'_, N, ET, TAG_BITS>
{
}
unsafe impl<N: Send + Sync, ET: Send + Sync, const TAG_BITS: u32> Sync
    for Edge<'_, N, ET, TAG_BITS>
{
}

#[repr(C)]
pub struct Manager<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    unique_table: Vec<Mutex<LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>>,
    data: ManuallyDrop<MD>,
    store_inner: *const StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>,
    gc_ongoing: TryLock,
    reorder_count: u64,
    phantom: PhantomData<(TM, R)>,
}

type M<'id, NC, ET, TMC, RC, MDC, const PAGE_SIZE: usize, const TAG_BITS: u32> = Manager<
    'id,
    <NC as InnerNodeCons<ET, TAG_BITS>>::T<'id>,
    ET,
    <TMC as TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>>::T<'id>,
    <RC as DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>>::T<'id>,
    <MDC as ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>>::T<'id>,
    PAGE_SIZE,
    TAG_BITS,
>;

unsafe impl<
        'id,
        N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>> + Send + Sync,
        ET: Tag + Send + Sync,
        TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS> + Send + Sync,
        R,
        MD: DropWith<Edge<'id, N, ET, TAG_BITS>> + Send + Sync,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Send for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
{
}
unsafe impl<
        'id,
        N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>> + Send + Sync,
        ET: Tag + Send + Sync,
        TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS> + Send + Sync,
        R,
        MD: DropWith<Edge<'id, N, ET, TAG_BITS>> + Send + Sync,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Sync for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
{
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> Drop
    for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    fn drop(&mut self) {
        if !N::needs_drop() {
            // There is no need to free nodes, so we can just forget all edges
            // and let the `ArcSlab` deallocate its pages.
            unsafe { ManuallyDrop::take(&mut self.data) }.drop_with(std::mem::forget);
        } else {
            unsafe { ManuallyDrop::take(&mut self.data) }.drop_with(|edge| {
                if edge.is_inner() {
                    // SAFETY: `edge` points to an inner node
                    unsafe { edge.drop_inner() };
                } else {
                    TM::drop_edge(edge);
                }
            });

            let unique_table = std::mem::take(&mut self.unique_table);
            for level in unique_table {
                for edge in level.into_inner() {
                    unsafe { Self::drop_from_unique_table(edge) };
                }
            }
        }
    }
}

#[repr(transparent)]
pub struct ManagerRef<
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
    RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
    MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>(
    ArcSlabRef<
        NC::T<'static>,
        StoreInner<
            'static,
            NC::T<'static>,
            ET,
            TMC::T<'static>,
            RC::T<'static>,
            MDC::T<'static>,
            PAGE_SIZE,
            TAG_BITS,
        >,
        PAGE_SIZE,
    >,
);

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    unsafe fn init_in(slot: *mut Self, data: MD, threads: u32) {
        let data = RwLock::new(Manager {
            unique_table: Vec::new(),
            data: ManuallyDrop::new(data),
            store_inner: slot,
            gc_ongoing: TryLock::new(),
            reorder_count: 0,
            phantom: PhantomData,
        });
        unsafe { std::ptr::write(addr_of_mut!((*slot).manager), data) };

        unsafe { TM::new_in(addr_of_mut!((*slot).terminal_manager)) };

        let workers = crate::workers::Workers::new(threads);
        unsafe { std::ptr::write(addr_of_mut!((*slot).workers), workers) };
    }
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    #[inline(always)]
    fn from_terminal_manager_ptr(ptr: *const TM) -> *const Self {
        let byte_offset = const { std::mem::offset_of!(Self, terminal_manager) as isize };
        // SAFETY: For all uses of this function, `ptr` points to a terminal
        // manager contained in a `StoreInner` allocation
        unsafe { ptr.byte_offset(-byte_offset) as *const Self }
    }

    #[inline(always)]
    fn from_manager_ptr(
        ptr: *const RwLock<Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>,
    ) -> *const Self {
        let byte_offset = const { std::mem::offset_of!(Self, manager) as isize };
        // SAFETY: For all uses of this function, `ptr` points to the manager
        // field contained in a `StoreInner` allocation
        unsafe { ptr.byte_offset(-byte_offset) as *const Self }
    }
}

/// Add a node to `store` and return the corresponding `Edge`
///
/// The caller should ensure that no node is inserted twice
#[inline]
fn add_node<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>(
    store: &ArcSlab<N, StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>, PAGE_SIZE>,
    node: N,
) -> AllocResult<[Edge<'id, N, ET, TAG_BITS>; 2]>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    let ptr = IntHandle::into_raw(store.add_item(node)).cast();
    Ok([Edge(ptr, PhantomData), Edge(ptr, PhantomData)])
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    /// Get the pointer to `StoreInner` for this manager
    ///
    /// This method must not be called during of store and manager. After
    /// initialization, the returned pointer is safe to dereference.
    #[inline(always)]
    fn store_inner_ptr(&self) -> *const StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS> {
        let store_inner = self.store_inner;
        // We can simply get the store pointer by subtracting the offset of
        // `Manager` in `Store`. The only issue is that this violates Rust's
        // (proposed) aliasing rules. Hence, we only provide a hint that the
        // store's address can be computed without loading the value.
        if store_inner != StoreInner::from_manager_ptr(RwLock::from_data_ptr(self)) {
            // SAFETY: after initialization, the pointers are equal
            unsafe { std::hint::unreachable_unchecked() };
        }
        store_inner
    }

    /// Get the node store / `ArcSlab` for this manager
    ///
    /// Actually, this is the `ArcSlab`, in which this manager is stored. This
    /// means that it is only safe to call this method in case this `Manager`
    /// is embedded in an `ArcSlab<N, StoreInner<..>, PAGE_SIZE>`. But this
    /// holds by construction.
    ///
    /// This method must not be called during of store and manager.
    #[inline(always)]
    fn store(
        &self,
    ) -> &ArcSlab<N, StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>, PAGE_SIZE> {
        let ptr = ArcSlab::from_data_ptr(self.store_inner_ptr());
        // SAFETY: After initialization, the pointer is guaranteed to be valid
        unsafe { &*ptr }
    }

    #[inline]
    fn terminal_manager(&self) -> *const TM {
        let store_inner = self.store_inner_ptr();
        unsafe { addr_of!((*store_inner).terminal_manager) }
    }

    /// Drop the given edge
    ///
    /// This is just a wrapper around [`Edge::drop_inner_untagged`] with
    /// less type parameters.
    ///
    /// SAFETY: `edge` must be untagged and point to an inner node
    #[inline(always)]
    unsafe fn drop_from_unique_table(edge: Edge<'id, N, ET, TAG_BITS>) {
        // SAFETY: `edge` is untagged and points to an inner node, the type
        // parameters match
        unsafe { edge.drop_from_unique_table::<TM, R, MD, PAGE_SIZE>() }
    }
}

impl<'id, N: NodeBase, ET: Tag, const TAG_BITS: u32> Edge<'id, N, ET, TAG_BITS> {
    /// Mask corresponding to `TAG_BITS`
    const TAG_MASK: usize = {
        assert!(ET::MAX_VALUE < (1 << TAG_BITS), "`TAG_BITS` is too small");
        // The mask we compute below has potentially less bits set than
        // `(1 << TAG_BITS) - 1`. If `ET::MAX_VALUE + 1` is a power of two, all
        // all bit patterns after applying the mask actually correspond to valid
        // bit patterns for the `ET`, so the compiler may optimize an
        // `assert!(value <= Self::MAX_VALUE)` in the `Tag::from_usize()`
        // implementation away, which would otherwise be required for safety.
        (ET::MAX_VALUE + 1).next_power_of_two() - 1
    };

    /// `TAG_BITS` (for the user-defined edge tag) plus 1 for the distinction
    /// between inner and terminal nodes
    const ALL_TAG_BITS: u32 = {
        assert!(
            align_of::<N>() >= 1 << (TAG_BITS + 1),
            "`TAG_BITS` is too large"
        );
        TAG_BITS + 1
    };

    /// Mask corresponding to `Self::ALL_TAG_BITS`
    const ALL_TAG_MASK: usize = (1 << Self::ALL_TAG_BITS) - 1;

    #[inline(always)]
    pub fn as_ptr(&self) -> NonNull<()> {
        self.0
    }

    /// Create an edge from a raw pointer
    ///
    /// This method does neither modify the pointer nor change reference
    /// counters in any way. The user is responsible for pointer tagging etc.
    /// Hence, using this method is dangerous. Still, it is required by
    /// `TerminalManager` implementations.
    ///
    /// # Safety
    ///
    /// `ptr` must be tagged accordingly.
    ///
    /// If the pointer is tagged as referring to an inner node, then it must
    /// actually point to an inner node stored in the manager associated with
    /// the `'id` brand. Furthermore, the caller must have ownership of one
    /// reference (in terms of the reference count).
    ///
    /// If the pointer is tagged as referring to a terminal node, then the
    /// the pointer must be valid as defined by the `TerminalManager`
    /// implementation of the manager associated with the `'id` brand.
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    /// Get the address portion of the underlying pointer
    #[inline(always)]
    pub fn addr(&self) -> usize {
        sptr::Strict::addr(self.0.as_ptr())
    }

    /// Get the inner node referenced by this edge
    ///
    /// # Safety
    ///
    /// `self` must be untagged and point to an inner node
    #[inline]
    unsafe fn inner_node_unchecked(&self) -> &N {
        debug_assert_eq!(self.addr() & Self::ALL_TAG_MASK, 0);
        let ptr: NonNull<N> = self.0.cast();
        unsafe { ptr.as_ref() }
    }

    /// Drop an edge pointing to an inner node, assuming that it isn't the last
    /// one pointing to that node
    ///
    /// This assumption is fulfilled in case the node is still stored in the
    /// unique table.
    ///
    /// There is a debug assertion that checks the aforementioned assumption. In
    /// release builds, this function will simply leak the node.
    ///
    /// SAFETY: `self` must point to an inner node
    #[inline]
    unsafe fn drop_inner(self) {
        debug_assert!(self.is_inner());
        let ptr: NonNull<N> = self.all_untagged_ptr().cast();
        std::mem::forget(self);
        let _old_rc = unsafe { ptr.as_ref().release() };
        debug_assert!(_old_rc > 1);
    }

    /// Drop an edge from the unique table
    ///
    /// Dropping an edge from the unique table corresponds to dropping the last
    /// reference.
    ///
    /// SAFETY:
    /// - `self` must be untagged and point to an inner node
    /// - `TM`, `R`, `MD` and `PAGE_SIZE` must be the types/values this edge has
    ///   been created with
    #[inline]
    unsafe fn drop_from_unique_table<TM, R, MD, const PAGE_SIZE: usize>(self)
    where
        N: InnerNode<Self>,
        TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
        MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
    {
        let handle: IntHandle<
            'id,
            N,
            Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>,
            PAGE_SIZE,
        > = {
            debug_assert_eq!(self.addr() & Self::ALL_TAG_MASK, 0);
            let ptr: NonNull<N> = self.0.cast();
            std::mem::forget(self);
            unsafe { IntHandle::from_raw(ptr) }
        };
        IntHandle::drop_with(handle, |node| {
            node.drop_with(|edge| {
                if edge.is_inner() {
                    // SAFETY: `edge` points to an inner node
                    unsafe { edge.drop_inner() };
                } else {
                    TM::drop_edge(edge);
                }
            })
        })
    }

    /// Clone an edge pointing to an inner node
    ///
    /// SAFETY: `self` must be untagged and point to an inner node
    #[inline]
    unsafe fn clone_inner_unchecked(&self) -> Self {
        unsafe { self.inner_node_unchecked() }.retain();
        Self(self.0, PhantomData)
    }

    /// Returns `true` if this edge points to an inner node
    #[inline]
    pub fn is_inner(&self) -> bool {
        (self.addr() & (1 << TAG_BITS)) == 0
    }

    /// Get the underlying pointer with all tag bits set to 0
    #[inline]
    fn all_untagged_ptr(&self) -> NonNull<()> {
        let ptr = sptr::Strict::map_addr(self.0.as_ptr(), |p| p & !Self::ALL_TAG_MASK);
        // SAFETY: the (tagged) pointer is `>= (1 << ALL_TAG_BITS)`
        unsafe { NonNull::new_unchecked(ptr) }
    }

    /// Get the underlying pointer tagged with `tag`
    #[inline]
    fn retag_ptr(&self, tag: ET) -> NonNull<()> {
        let tag_val = tag.as_usize();
        debug_assert!(tag_val <= ET::MAX_VALUE);
        // Note that we assert `ET::MAX_VALUE <= Self::TAG_MASK` during the computation
        // of `Self::TAG_MASK`
        let ptr = sptr::Strict::map_addr(self.0.as_ptr(), |p| (p & !Self::TAG_MASK) | tag_val);
        // SAFETY: even an untagged pointer is non-null
        unsafe { NonNull::new_unchecked(ptr) }
    }
}

impl<N, ET, const TAG_BITS: u32> Drop for Edge<'_, N, ET, TAG_BITS> {
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

unsafe impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> oxidd_core::Manager
    for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>> + GCContainer<Self>,
{
    type Edge = Edge<'id, N, ET, TAG_BITS>;
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

    type NodeSet = NodeSet<PAGE_SIZE, TAG_BITS>;

    type LevelView<'a>
        = LevelView<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
    where
        Self: 'a;
    type LevelIterator<'a>
        = LevelIter<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
    where
        Self: 'a;

    #[inline]
    fn get_node(&self, edge: &Self::Edge) -> Node<Self> {
        if edge.is_inner() {
            let ptr: NonNull<Self::InnerNode> = edge.all_untagged_ptr().cast();
            // SAFETY: dereferencing untagged edges pointing to inner nodes is safe
            Node::Inner(unsafe { ptr.as_ref() })
        } else {
            let terminal_manager = unsafe { &*self.terminal_manager() };
            Node::Terminal(terminal_manager.deref_edge(edge))
        }
    }

    #[inline]
    fn clone_edge(&self, edge: &Self::Edge) -> Self::Edge {
        if edge.is_inner() {
            let ptr: NonNull<Self::InnerNode> = edge.all_untagged_ptr().cast();
            // SAFETY: dereferencing untagged edges pointing to inner nodes is safe
            unsafe { ptr.as_ref() }.retain();
            Edge(edge.0, PhantomData)
        } else {
            TM::clone_edge(edge)
        }
    }

    #[inline]
    fn drop_edge(&self, edge: Self::Edge) {
        if edge.is_inner() {
            // SAFETY: `edge` points to an inner node
            unsafe { edge.drop_inner() };
        } else {
            TM::drop_edge(edge);
        }
    }

    #[inline]
    fn num_inner_nodes(&self) -> usize {
        self.store().num_items()
    }

    #[inline]
    fn num_levels(&self) -> LevelNo {
        self.unique_table.len() as LevelNo
    }

    fn add_level(&mut self, f: impl FnOnce(LevelNo) -> Self::InnerNode) -> AllocResult<Self::Edge> {
        let level_no = self.unique_table.len() as LevelNo;
        assert!(level_no < LevelNo::MAX, "too many levels");
        let node = f(level_no);
        node.assert_level_matches(level_no);

        let [e1, e2] = add_node(self.store(), node)?;

        let mut set = LevelViewSet::default();
        // SAFETY: edges in unique table entries are always untagged and point
        // to inner nodes
        set.insert(e1);
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
        unsafe { TM::get(self.terminal_manager(), terminal) }
    }

    #[inline]
    fn num_terminals(&self) -> usize {
        let terminal_manager = unsafe { &*self.terminal_manager() };
        terminal_manager.len()
    }

    #[inline]
    fn terminals(&self) -> Self::TerminalIterator<'_> {
        unsafe { TM::iter(self.terminal_manager()) }
    }

    fn gc(&self) -> usize {
        if !self.gc_ongoing.try_lock() {
            // We don't want multiple garbage collections at the same time.
            return 0;
        }
        let guard = AbortOnDrop("Garbage collection panicked.");
        self.data.pre_gc(self);

        let mut collected = 0;
        for level in &self.unique_table {
            let mut level = level.lock();
            collected += level.len();
            // SAFETY: We prepared the garbage collection, hence there are no
            // "weak" edges.
            unsafe { level.gc() };
            collected -= level.len();
        }
        collected += unsafe { &*self.terminal_manager() }.gc();

        // SAFETY: We called `pre_gc()` and the garbage collection is done.
        unsafe { self.data.post_gc(self) };
        self.gc_ongoing.unlock();
        guard.defuse();
        collected
    }

    fn reorder<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let guard = AbortOnDrop("Reordering panicked.");
        self.data.pre_gc(self);
        let res = f(self);
        // SAFETY: We called `pre_gc()` and the reordering is done.
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

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> oxidd_core::HasWorkers
    for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>> + Send + Sync,
    ET: Tag + Send + Sync,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS> + Send + Sync,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>> + GCContainer<Self> + Send + Sync,
{
    type WorkerPool = crate::workers::Workers;

    #[inline]
    fn workers(&self) -> &Self::WorkerPool {
        &self.store().data().workers
    }
}

pub struct LevelIter<'a, 'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    store: &'a ArcSlab<N, StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>, PAGE_SIZE>,
    level_front: LevelNo,
    level_back: LevelNo,
    it: std::slice::Iter<'a, Mutex<LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>>,
}

impl<'a, 'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> Iterator
    for LevelIter<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    type Item = LevelView<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>;

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

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> ExactSizeIterator
    for LevelIter<'_, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    #[inline]
    fn len(&self) -> usize {
        self.it.len()
    }
}

impl<'a, 'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> FusedIterator
    for LevelIter<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
    std::slice::Iter<'a, Mutex<LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>>:
        FusedIterator,
{
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> DoubleEndedIterator
    for LevelIter<'_, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    #[inline]
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

impl<N, ET, const TAG_BITS: u32> PartialEq for Edge<'_, N, ET, TAG_BITS> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<N, ET, const TAG_BITS: u32> Eq for Edge<'_, N, ET, TAG_BITS> {}

impl<N, ET, const TAG_BITS: u32> PartialOrd for Edge<'_, N, ET, TAG_BITS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<N, ET, const TAG_BITS: u32> Ord for Edge<'_, N, ET, TAG_BITS> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<N, ET, const TAG_BITS: u32> Hash for Edge<'_, N, ET, TAG_BITS> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<N: NodeBase, ET: Tag, const TAG_BITS: u32> oxidd_core::Edge for Edge<'_, N, ET, TAG_BITS> {
    type Tag = ET;

    #[inline]
    fn borrowed(&self) -> Borrowed<'_, Self> {
        Borrowed::new(Self(self.0, PhantomData))
    }

    #[inline]
    fn with_tag(&self, tag: Self::Tag) -> Borrowed<'_, Self> {
        Borrowed::new(Self(self.retag_ptr(tag), PhantomData))
    }

    #[inline]
    fn with_tag_owned(mut self, tag: Self::Tag) -> Self {
        self.0 = self.retag_ptr(tag);
        self
    }

    #[inline]
    fn tag(&self) -> Self::Tag {
        ET::from_usize(self.addr() & Self::TAG_MASK)
    }

    #[inline]
    fn node_id(&self) -> oxidd_core::NodeID {
        self.addr() & !Self::ALL_TAG_MASK
    }
}

// === Level Views =============================================================

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
struct LevelViewSet<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>(
    RawTable<Edge<'id, N, ET, TAG_BITS>>,
    PhantomData<(TM, R, MD)>,
);

#[inline]
fn hash_node<N: NodeBase>(node: &N) -> u64 {
    let mut hasher = FxHasher::default();
    node.hash(&mut hasher);
    hasher.finish()
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
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
    unsafe fn eq(node: &N) -> impl Fn(&Edge<'id, N, ET, TAG_BITS>) -> bool + '_ {
        move |edge| unsafe { edge.inner_node_unchecked() == node }
    }

    /// Reserve space for `additional` nodes on this level
    #[inline]
    fn reserve(&mut self, additional: usize) {
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.reserve(additional)
    }

    #[inline]
    fn get(&self, node: &N) -> Option<&Edge<'id, N, ET, TAG_BITS>> {
        let hash = hash_node(node);
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.get(hash, unsafe { Self::eq(node) })
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
    fn insert(&mut self, edge: Edge<'id, N, ET, TAG_BITS>) -> bool {
        assert_eq!(
            edge.addr() & Edge::<N, ET, TAG_BITS>::ALL_TAG_MASK,
            0,
            "can only insert untagged edges pointing to inner nodes"
        );
        let edge = ManuallyDrop::new(edge);
        let node = unsafe { edge.inner_node_unchecked() };
        let hash = hash_node(node);
        // SAFETY (next 2): The hash table only contains untagged edges
        // referencing inner nodes.
        match self
            .0
            .find_or_find_insert_slot(hash, unsafe { Self::eq(node) })
        {
            Ok(_) => {
                // We need to drop `edge`. This simply amounts to decrementing
                // the reference counter, since there is still one edge
                // referencing `node` stored in here.

                // SAFETY: We forget `edge`.
                let _old_rc = unsafe { node.release() };
                debug_assert!(_old_rc > 1);

                false
            }
            Err(slot) => {
                // SAFETY: `slot` was returned by `find_or_find_insert_slot`.
                // We have exclusive access to the hash table and did not modify
                // it in between.
                unsafe {
                    self.0
                        .insert_in_slot_unchecked(hash, slot, ManuallyDrop::into_inner(edge))
                };
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
        node: N,
        insert: impl FnOnce(N) -> AllocResult<[Edge<'id, N, ET, TAG_BITS>; 2]>,
    ) -> AllocResult<Edge<'id, N, ET, TAG_BITS>> {
        let hash = hash_node(&node);
        // SAFETY (next 2): The hash table only contains untagged edges
        // referencing inner nodes.
        match self
            .0
            .find_or_find_insert_slot(hash, unsafe { Self::eq(&node) })
        {
            Ok(slot) => {
                node.drop_with(|edge| {
                    if edge.is_inner() {
                        // SAFETY: `edge` points to an inner node
                        unsafe { edge.drop_inner() };
                    } else {
                        TM::drop_edge(edge);
                    }
                });
                // SAFETY:
                // - `slot` was returned by `find_or_find_insert_slot`. We have exclusive access
                //   to the hash table and did not modify it in between.
                // - All edges in the table are untagged and refer to inner nodes.
                Ok(unsafe { self.0.get_at_slot_unchecked(slot).clone_inner_unchecked() })
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
    unsafe fn gc(&mut self) {
        self.0.retain(
            |edge| {
                // SAFETY: All edges in unique tables are untagged and point to
                // inner nodes.
                unsafe { edge.inner_node_unchecked() }.ref_count() != 0
            },
            |edge| {
                // SAFETY: All edges in unique tables are untagged and point to
                // inner nodes.
                unsafe { edge.drop_from_unique_table::<TM, R, MD, PAGE_SIZE>() };
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
    unsafe fn remove(&mut self, node: &N) -> Option<Edge<'id, N, ET, TAG_BITS>> {
        let hash = hash_node(node);
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.remove_entry(hash, unsafe { Self::eq(node) })
    }

    /// Iterate over all edges pointing to nodes in the set
    #[inline]
    fn iter(&self) -> LevelViewIter<'_, 'id, N, ET, TAG_BITS> {
        LevelViewIter(self.0.iter())
    }

    /// Iterator that consumes all [`Edge`]s in the set
    #[inline]
    fn drain(&mut self) -> linear_hashtbl::raw::Drain<Edge<'id, N, ET, TAG_BITS>> {
        self.0.drain()
    }
}

impl<N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> Drop
    for LevelViewSet<'_, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn drop(&mut self) {
        // If the nodes need drop, this is handled by the `Manager` `Drop` impl
        // or the `TakenLevelView` `Drop` impl, respectively.
        self.0.reset_no_drop();
    }
}

impl<N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> Default
    for LevelViewSet<'_, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> IntoIterator
    for LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
{
    type Item = Edge<'id, N, ET, TAG_BITS>;

    type IntoIter = linear_hashtbl::raw::IntoIter<Edge<'id, N, ET, TAG_BITS>>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        // SAFETY: We move out of `this` (and forget `this`)
        let set = unsafe { std::ptr::read(&this.0) };
        set.into_iter()
    }
}

/// Level view provided by the unique table
pub struct LevelView<'a, 'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    store: &'a ArcSlab<N, StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>, PAGE_SIZE>,
    level: LevelNo,
    set: MutexGuard<'a, LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>,
}

unsafe impl<'a, 'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    oxidd_core::LevelView<Edge<'id, N, ET, TAG_BITS>, N>
    for LevelView<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>
        + GCContainer<Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>,
{
    type Iterator<'b>
        = LevelViewIter<'b, 'id, N, ET, TAG_BITS>
    where
        Self: 'b;

    type Taken = TakenLevelView<'a, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>;

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
        self.set.reserve(additional)
    }

    #[inline]
    fn get(&self, node: &N) -> Option<&Edge<'id, N, ET, TAG_BITS>> {
        self.set.get(node)
    }

    #[inline]
    fn insert(&mut self, edge: Edge<'id, N, ET, TAG_BITS>) -> bool {
        assert_eq!(
            edge.addr() & Edge::<N, ET, TAG_BITS>::ALL_TAG_MASK,
            0,
            "can only insert untagged edges pointing to inner nodes"
        );
        unsafe { edge.inner_node_unchecked() }.assert_level_matches(self.level);
        self.set.insert(edge)
    }

    #[inline(always)]
    fn get_or_insert(&mut self, node: N) -> AllocResult<Edge<'id, N, ET, TAG_BITS>> {
        node.assert_level_matches(self.level);
        // No need to check if the children of `node` are stored in `self.store`
        // due to lifetime restrictions.
        LevelViewSet::get_or_insert(&mut *self.set, node, |node| add_node(self.store, node))
    }

    #[inline(always)]
    unsafe fn gc(&mut self) {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        unsafe { self.set.gc() };
    }

    #[inline]
    unsafe fn remove(&mut self, node: &N) -> bool {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        match unsafe { self.set.remove(node) } {
            Some(edge) => {
                // SAFETY: `edge` is untagged, the type parameters match
                unsafe { edge.drop_from_unique_table::<TM, R, MD, PAGE_SIZE>() };
                true
            }
            None => false,
        }
    }

    #[inline]
    unsafe fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut *self.set, &mut *other.set);
    }

    #[inline]
    fn iter(&self) -> Self::Iterator<'_> {
        self.set.iter()
    }

    #[inline]
    fn take(&mut self) -> Self::Taken {
        TakenLevelView {
            store: self.store,
            level: self.level,
            set: std::mem::take(&mut self.set),
        }
    }
}

/// Owned level view provided by the unique table
pub struct TakenLevelView<'a, 'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    store: &'a ArcSlab<N, StoreInner<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>, PAGE_SIZE>,
    level: LevelNo,
    set: LevelViewSet<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>,
}

unsafe impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32>
    oxidd_core::LevelView<Edge<'id, N, ET, TAG_BITS>, N>
    for TakenLevelView<'_, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>
        + GCContainer<Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>>,
{
    type Iterator<'b>
        = LevelViewIter<'b, 'id, N, ET, TAG_BITS>
    where
        Self: 'b;

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
        self.set.reserve(additional)
    }

    #[inline]
    fn get(&self, node: &N) -> Option<&Edge<'id, N, ET, TAG_BITS>> {
        self.set.get(node)
    }

    #[inline]
    fn insert(&mut self, edge: Edge<'id, N, ET, TAG_BITS>) -> bool {
        assert_eq!(
            edge.addr() & Edge::<N, ET, TAG_BITS>::ALL_TAG_MASK,
            0,
            "can only insert untagged edges pointing to inner nodes"
        );
        unsafe { edge.inner_node_unchecked() }.assert_level_matches(self.level);
        self.set.insert(edge)
    }

    #[inline(always)]
    fn get_or_insert(&mut self, node: N) -> AllocResult<Edge<'id, N, ET, TAG_BITS>> {
        node.assert_level_matches(self.level);
        // No need to check if the children of `node` are stored in `self.store`
        // due to lifetime restrictions.
        self.set
            .get_or_insert(node, |node| add_node(self.store, node))
    }

    #[inline]
    unsafe fn remove(&mut self, node: &N) -> bool {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        match unsafe { self.set.remove(node) } {
            Some(edge) => {
                // SAFETY: `edge` is untagged, the type parameters match
                unsafe { edge.drop_from_unique_table::<TM, R, MD, PAGE_SIZE>() };
                true
            }
            None => false,
        }
    }

    #[inline]
    unsafe fn gc(&mut self) {
        // SAFETY: Called from inside the closure of `Manager::reorder()`, hence
        // there are no "weak" edges.
        unsafe { self.set.gc() };
    }

    #[inline]
    unsafe fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.set, &mut other.set);
    }

    #[inline]
    fn iter(&self) -> Self::Iterator<'_> {
        self.set.iter()
    }

    #[inline]
    fn take(&mut self) -> Self::Taken {
        TakenLevelView {
            store: self.store,
            level: self.level,
            set: std::mem::take(&mut self.set),
        }
    }
}

impl<'id, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> Drop
    for TakenLevelView<'_, 'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>>,
{
    fn drop(&mut self) {
        for edge in self.set.drain() {
            // SAFETY: `edge` is untagged and points to an inner node, the type
            // parameters match
            unsafe { edge.drop_from_unique_table::<TM, R, MD, PAGE_SIZE>() };
        }
    }
}

// --- Level Views: Iterator ---------------------------------------------------

/// Iterator over entries (as [`Edge`]s) of a level view
pub struct LevelViewIter<'a, 'id, N, ET, const TAG_BITS: u32>(
    linear_hashtbl::raw::Iter<'a, Edge<'id, N, ET, TAG_BITS>>,
);

impl<'a, 'id, InnerNode, ET, const TAG_BITS: u32> Iterator
    for LevelViewIter<'a, 'id, InnerNode, ET, TAG_BITS>
{
    type Item = &'a Edge<'id, InnerNode, ET, TAG_BITS>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<N, ET, const TAG_BITS: u32> ExactSizeIterator for LevelViewIter<'_, '_, N, ET, TAG_BITS> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, 'id, N, ET, const TAG_BITS: u32> FusedIterator for LevelViewIter<'a, 'id, N, ET, TAG_BITS> where
    linear_hashtbl::raw::Iter<'a, Edge<'id, N, ET, TAG_BITS>, u32>: FusedIterator
{
}

// === Node Set ================================================================

/// Node set implementation using [bit vectors][BitVec]
///
/// Since nodes are stored on large pages, we can use one bit vector per page.
/// This reduces space consumption dramatically and increases the performance.
pub struct NodeSet<const PAGE_SIZE: usize, const TAG_BITS: u32> {
    len: usize,
    data: HashMap<usize, BitVec, BuildHasherDefault<FxHasher>>,
}

impl<const PAGE_SIZE: usize, const TAG_BITS: u32> NodeSet<PAGE_SIZE, TAG_BITS> {
    const NODES_PER_PAGE: usize = PAGE_SIZE >> (TAG_BITS + 1);

    #[inline]
    fn page_offset<InnerNode, ET>(edge: &Edge<'_, InnerNode, ET, TAG_BITS>) -> (usize, usize) {
        let node_id = sptr::Strict::addr(edge.0.as_ptr()) >> TAG_BITS;
        let page = node_id / Self::NODES_PER_PAGE;
        let offset = node_id % Self::NODES_PER_PAGE;
        (page, offset)
    }
}

impl<const PAGE_SIZE: usize, const TAG_BITS: u32> Default for NodeSet<PAGE_SIZE, TAG_BITS> {
    fn default() -> Self {
        Self {
            len: 0,
            data: Default::default(),
        }
    }
}

impl<const PAGE_SIZE: usize, const TAG_BITS: u32> Clone for NodeSet<PAGE_SIZE, TAG_BITS> {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            data: self.data.clone(),
        }
    }
}

impl<const PAGE_SIZE: usize, const TAG_BITS: u32> PartialEq for NodeSet<PAGE_SIZE, TAG_BITS> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data == other.data
    }
}
impl<const PAGE_SIZE: usize, const TAG_BITS: u32> Eq for NodeSet<PAGE_SIZE, TAG_BITS> {}

impl<'id, InnerNode, ET, const PAGE_SIZE: usize, const TAG_BITS: u32>
    oxidd_core::util::NodeSet<Edge<'id, InnerNode, ET, TAG_BITS>> for NodeSet<PAGE_SIZE, TAG_BITS>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    fn insert(&mut self, edge: &Edge<'id, InnerNode, ET, TAG_BITS>) -> bool {
        let (page, offset) = Self::page_offset(edge);
        match self.data.entry(page) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let page = e.get_mut();
                if page[offset] {
                    false
                } else {
                    page.set(offset, true);
                    self.len += 1;
                    true
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut page = bitvec![0; Self::NODES_PER_PAGE];
                page.set(offset, true);
                e.insert(page);
                self.len += 1;
                true
            }
        }
    }

    #[inline]
    fn contains(&self, edge: &Edge<'id, InnerNode, ET, TAG_BITS>) -> bool {
        let (page, offset) = Self::page_offset(edge);
        match self.data.get(&page) {
            Some(page) => page[offset],
            None => false,
        }
    }

    fn remove(&mut self, edge: &Edge<'id, InnerNode, ET, TAG_BITS>) -> bool {
        let (page, offset) = Self::page_offset(edge);
        match self.data.get_mut(&page) {
            Some(page) => {
                if page[offset] {
                    page.set(offset, false);
                    self.len -= 1;
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }
}

// === `ManagerRef` & Creation =================================================

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    /// Convert `self` into a raw pointer, e.g. for usage in a foreign function
    /// interface.
    ///
    /// This method does not change any reference counters. To avoid a memory
    /// leak, use [`Self::from_raw()`] to convert the pointer back into a
    /// `ManagerRef`.
    #[inline(always)]
    pub fn into_raw(self) -> *const std::ffi::c_void {
        let ptr = ArcSlabRef::into_raw(self.0);
        ptr.as_ptr() as _
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
        let ptr = NonNull::new(raw as *mut _).expect("expected a non-null pointer");
        // SAFETY: Invariants are upheld by the caller.
        Self(unsafe { ArcSlabRef::from_raw(ptr) })
    }
}

impl<
        'a,
        'id,
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > From<&'a M<'id, NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>>
    for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn from(manager: &'a M<'id, NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>) -> Self {
        manager.store().retain();
        let store_ptr =
            ArcSlab::<NC::T<'id>, _, PAGE_SIZE>::from_data_ptr(manager.store_inner_ptr()) as *mut _;
        Self(unsafe { ArcSlabRef::from_raw(NonNull::new_unchecked(store_ptr)) })
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > oxidd_core::ManagerRef for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    type Manager<'id> = M<'id, NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>;

    #[inline]
    fn with_manager_shared<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&Self::Manager<'id>) -> T,
    {
        f(&*self.0.data().manager.shared())
    }

    #[inline]
    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>) -> T,
    {
        f(&mut *self.0.data().manager.exclusive())
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Clone for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > PartialEq for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Eq for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Hash for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > PartialOrd for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Ord for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag + Sync + Send,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > oxidd_core::HasWorkers for ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
where
    NC::T<'static>: Send + Sync,
    TMC::T<'static>: Send + Sync,
    MDC::T<'static>: Send + Sync,
{
    type WorkerPool = crate::workers::Workers;

    #[inline]
    fn workers(&self) -> &Self::WorkerPool {
        &self.0.data().workers
    }
}

/// Create a new manager
pub fn new_manager<
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
    RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
    MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>(
    data: MDC::T<'static>,
    threads: u32,
) -> ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS> {
    // Evaluate a few constants for assertions
    let _ = Edge::<'static, NC::T<'static>, ET, TAG_BITS>::TAG_MASK;
    let _ = Edge::<'static, NC::T<'static>, ET, TAG_BITS>::ALL_TAG_BITS;

    ManagerRef(unsafe { ArcSlab::new_with(|slot| StoreInner::init_in(slot, data, threads)) })
}

// === Functions ===============================================================

#[repr(transparent)]
pub struct Function<
    NC: InnerNodeCons<ET, TAG_BITS>,
    ET: Tag,
    TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
    RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
    MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>(NonNull<()>, PhantomData<(NC, ET, TMC, RC, MDC)>);

unsafe impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Send for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
where
    for<'id> NC::T<'id>: Send + Sync,
    for<'id> TMC::T<'id>: Send + Sync,
    for<'id> MDC::T<'id>: Send + Sync,
{
}

unsafe impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Sync for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
where
    for<'id> NC::T<'id>: Send + Sync,
    for<'id> TMC::T<'id>: Send + Sync,
    for<'id> MDC::T<'id>: Send + Sync,
{
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn store(
        &self,
    ) -> NonNull<
        ArcSlab<
            NC::T<'static>,
            StoreInner<
                'static,
                NC::T<'static>,
                ET,
                TMC::T<'static>,
                RC::T<'static>,
                MDC::T<'static>,
                PAGE_SIZE,
                TAG_BITS,
            >,
            PAGE_SIZE,
        >,
    > {
        let edge: ManuallyDrop<Edge<'static, NC::T<'static>, ET, TAG_BITS>> =
            ManuallyDrop::new(Edge(self.0, PhantomData));
        if edge.is_inner() {
            let ptr: NonNull<NC::T<'static>> = edge.all_untagged_ptr().cast();
            let handle = ManuallyDrop::new(unsafe { ExtHandle::from_raw(ptr) });
            NonNull::from(ExtHandle::slab(&*handle))
        } else {
            let ptr = ArcSlab::from_data_ptr(StoreInner::from_terminal_manager_ptr(
                TerminalManager::terminal_manager(&*edge).as_ptr(),
            ));
            unsafe { NonNull::new_unchecked(ptr.cast_mut()) }
        }
    }

    /// Convert `self` into a raw pointer, e.g. for usage in a foreign function
    /// interface.
    ///
    /// This method does not change any reference counters. To avoid a memory
    /// leak, use [`Self::from_raw()`] to convert the pointer back into a
    /// `Function`.
    #[inline(always)]
    pub fn into_raw(self) -> *const std::ffi::c_void {
        let ptr = self.0;
        std::mem::forget(self);
        ptr.as_ptr() as _
    }

    /// Convert `raw` into a `Function`
    ///
    /// # Safety
    ///
    /// `raw` must have been obtained via [`Self::into_raw()`]. This function
    /// does not change any reference counters, so calling this function
    /// multiple times for the same pointer may lead to use after free bugs
    /// depending on the usage of the returned `Function`.
    #[inline(always)]
    pub unsafe fn from_raw(raw: *const std::ffi::c_void) -> Self {
        let ptr = NonNull::new(raw as *mut ()).expect("expected a non-null pointer");
        Self(ptr, PhantomData)
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Clone for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn clone(&self) -> Self {
        let mut edge: ManuallyDrop<Edge<'static, NC::T<'static>, ET, TAG_BITS>> =
            ManuallyDrop::new(Edge(self.0, PhantomData));
        if edge.is_inner() {
            edge.0 = edge.all_untagged_ptr();
            unsafe { edge.inner_node_unchecked() }.retain();
        } else {
            std::mem::forget(TMC::T::<'static>::clone_edge(&*edge));
        }
        let store = self.store();
        unsafe { store.as_ref() }.retain();
        Self(self.0, PhantomData)
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Drop for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    fn drop(&mut self) {
        let store = self.store();
        let edge: Edge<'static, NC::T<'static>, ET, TAG_BITS> = Edge(self.0, PhantomData);
        if edge.is_inner() {
            // SAFETY: `edge` points to an inner node
            unsafe { edge.drop_inner() };
        } else {
            TMC::T::<'static>::drop_edge(edge);
        }
        // We own a reference count, `store` is valid, and we do not use the
        // `store` pointer afterwards.
        unsafe { ArcSlab::release(store) };
    }
}

impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > PartialEq for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Eq for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
}
impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > PartialOrd for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}
impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Ord for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}
impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > Hash for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

unsafe impl<
        NC: InnerNodeCons<ET, TAG_BITS>,
        ET: Tag,
        TMC: TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > oxidd_core::function::Function for Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    type Manager<'id> =
        Manager<'id, NC::T<'id>, ET, TMC::T<'id>, RC::T<'id>, MDC::T<'id>, PAGE_SIZE, TAG_BITS>;

    type ManagerRef = ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>;

    #[inline]
    fn from_edge<'id>(manager: &Self::Manager<'id>, edge: EdgeOfFunc<'id, Self>) -> Self {
        manager.store().retain();
        Self(ManuallyDrop::new(edge).0, PhantomData)
    }

    #[inline]
    fn as_edge<'id>(&self, manager: &Self::Manager<'id>) -> &EdgeOfFunc<'id, Self> {
        assert!(util::ptr_eq_untyped(self.store().as_ptr(), manager.store()));
        // SAFETY: `Function` and `Edge` have the same representation
        unsafe { std::mem::transmute(self) }
    }

    #[inline]
    fn into_edge<'id>(self, manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        let store = manager.store();
        assert!(util::ptr_eq_untyped(self.store().as_ptr(), store));
        unsafe { ArcSlab::release(NonNull::from(store)) };
        Edge(ManuallyDrop::new(self).0, PhantomData)
    }

    #[inline]
    fn manager_ref(&self) -> Self::ManagerRef {
        let store_ptr = self.store();
        unsafe { store_ptr.as_ref() }.retain();
        ManagerRef(unsafe { ArcSlabRef::from_raw(store_ptr) })
    }

    #[inline]
    fn with_manager_shared<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> T,
    {
        let edge = ManuallyDrop::new(Edge(self.0, PhantomData));
        let store_ptr = self.store();
        f(
            &*unsafe { store_ptr.as_ref() }.data().manager.shared(),
            &*edge,
        )
    }

    #[inline]
    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> T,
    {
        let edge = ManuallyDrop::new(Edge(self.0, PhantomData));
        let store_ptr = self.store();
        f(
            &mut *unsafe { store_ptr.as_ref() }.data().manager.exclusive(),
            &*edge,
        )
    }
}

// === Additional Trait Implementations ========================================

impl<
        'id,
        N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
        ET: Tag,
        TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
        R: DiagramRules<Edge<'id, N, ET, TAG_BITS>, N, TM::TerminalNode>,
        MD: HasApplyCache<Self, O> + GCContainer<Self> + DropWith<Edge<'id, N, ET, TAG_BITS>>,
        O: Copy,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > HasApplyCache<Self, O> for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
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

impl<'id, T, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> AsRef<T>
    for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>> + AsRef<T>,
{
    #[inline(always)]
    fn as_ref(&self) -> &T {
        self.data.as_ref()
    }
}

impl<'id, T, N, ET, TM, R, MD, const PAGE_SIZE: usize, const TAG_BITS: u32> AsMut<T>
    for Manager<'id, N, ET, TM, R, MD, PAGE_SIZE, TAG_BITS>
where
    N: NodeBase + InnerNode<Edge<'id, N, ET, TAG_BITS>>,
    ET: Tag,
    TM: TerminalManager<'id, N, ET, MD, PAGE_SIZE, TAG_BITS>,
    MD: DropWith<Edge<'id, N, ET, TAG_BITS>> + AsMut<T>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T {
        self.data.as_mut()
    }
}
