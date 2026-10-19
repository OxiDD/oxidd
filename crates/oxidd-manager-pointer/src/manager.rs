//! Pointer-Based Manager Implementation

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, align_of};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::{Acquire, Relaxed};

use arcslab::{ArcSlab, ArcSlabRef, AtomicRefCounted, ExtHandle, IntHandle, PageSizeInfo};
use derive_where::derive_where;
use fixedbitset::FixedBitSet;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHasher;

use oxidd_core::error::DuplicateVarName;
use oxidd_core::function::{EdgeOfFunc, OwnEdgeOfFunc};
use oxidd_core::util::{AbortOnDrop, AllocResult, DropWith, Own, Ref, VarNameMap};
use oxidd_core::{
    Countable, HasApplyCache, InnerNode, LevelNo, ManagerEventSubscriber, Node, VarNo,
};

use crate::DiagramInfo;
use crate::node::NodeBase;
use crate::terminal_manager::TerminalManager;
use crate::util::{Invariant, TryLock, VarLevelMap, rwlock::RwLock};

#[cfg(not(feature = "hugealloc"))]
type RawTable<T> = linear_hashtbl::raw::RawTable<T, usize>;
#[cfg(feature = "hugealloc")]
type RawTable<T> = linear_hashtbl::raw::RawTable<T, usize, hugealloc::HugeAlloc>;

// === Manager & Edges =========================================================

pub struct StoreInner<'id, DI: DiagramInfo> {
    manager: RwLock<Manager<'id, DI>>,
    terminal_manager: DI::TerminalManager<'id>,
    workers: crate::workers::Workers,
}

type StoreSlab<'id, DI> = ArcSlab<
    <DI as DiagramInfo>::InnerNode<'id>,
    StoreInner<'id, DI>,
    <DI as DiagramInfo>::PageSizeInfo,
>;

#[repr(transparent)]
#[must_use]
#[derive_where(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge<'id, DI: DiagramInfo>(
    /// Points to an `InnerNode` (if `ptr & (1 << TAG_BITS) == 0`) or a terminal
    /// node (`ptr & (1 << TAG_BITS) == 1`)
    ///
    /// SAFETY invariant: The pointer (as integer) is `>= 1 << (TAG_BITS + 1)`
    /// (i.e. the pointer with all tags removed is still non-null)
    NonNull<()>,
    #[derive_where(skip)] PhantomData<(Invariant<'id>, DI)>,
);

unsafe impl<DI: DiagramInfo> Send for Edge<'_, DI> {}
unsafe impl<DI: DiagramInfo> Sync for Edge<'_, DI> {}

#[repr(C)]
pub struct Manager<'id, DI: DiagramInfo> {
    unique_table: Vec<Mutex<LevelViewSet<'id, DI>>>,
    var_level_map: VarLevelMap,
    var_name_map: VarNameMap,
    data: ManuallyDrop<DI::ManagerData<'id>>,
    store_inner: *const StoreInner<'id, DI>,
    gc_count: AtomicU64,
    reorder_count: u64,
    gc_ongoing: TryLock,
    reorder_gc_prepared: bool,
}

unsafe impl<'id, DI: DiagramInfo> Send for Manager<'id, DI> {}
unsafe impl<'id, DI: DiagramInfo> Sync for Manager<'id, DI> {}

impl<'id, DI: DiagramInfo> Drop for Manager<'id, DI> {
    fn drop(&mut self) {
        if !DI::InnerNode::<'id>::needs_drop() {
            // There is no need to free nodes, so we can just forget all edges
            // and let the `ArcSlab` deallocate its pages.
            unsafe { ManuallyDrop::take(&mut self.data) }.drop_with(std::mem::forget);
        } else {
            unsafe { ManuallyDrop::take(&mut self.data) }.drop_with(Self::drop_edge_common);

            let unique_table = std::mem::take(&mut self.unique_table);
            for level in unique_table {
                for edge in level.into_inner() {
                    unsafe { Self::drop_edge_from_unique_table(edge) };
                }
            }
        }
    }
}

#[repr(transparent)]
#[derive_where(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManagerRef<DI: DiagramInfo>(
    ArcSlabRef<DI::InnerNode<'static>, StoreInner<'static, DI>, DI::PageSizeInfo>,
);

impl<'id, DI: DiagramInfo> StoreInner<'id, DI> {
    unsafe fn init_in(slot: *mut Self, data: DI::ManagerData<'id>, threads: u32) {
        let data = RwLock::new(Manager {
            unique_table: Vec::new(),
            var_level_map: VarLevelMap::new(),
            var_name_map: VarNameMap::new(),
            data: ManuallyDrop::new(data),
            store_inner: slot,
            gc_count: AtomicU64::new(0),
            reorder_count: 0,
            gc_ongoing: TryLock::new(),
            reorder_gc_prepared: false,
        });
        unsafe { std::ptr::write(&raw mut ((*slot).manager), data) };

        unsafe { DI::TerminalManager::<'id>::new_in(&raw mut ((*slot).terminal_manager)) };

        let workers = crate::workers::Workers::new(threads);
        unsafe { std::ptr::write(&raw mut ((*slot).workers), workers) };
    }

    #[inline(always)]
    fn from_terminal_manager_ptr(ptr: *const DI::TerminalManager<'id>) -> *const Self {
        let byte_offset = const { std::mem::offset_of!(Self, terminal_manager) as isize };
        // SAFETY: For all uses of this function, `ptr` points to a terminal
        // manager contained in a `StoreInner` allocation
        unsafe { ptr.byte_offset(-byte_offset) }.cast()
    }

    #[inline(always)]
    fn from_manager_ptr(ptr: *const RwLock<Manager<'id, DI>>) -> *const Self {
        let byte_offset = const { std::mem::offset_of!(Self, manager) as isize };
        // SAFETY: For all uses of this function, `ptr` points to the manager
        // field contained in a `StoreInner` allocation
        unsafe { ptr.byte_offset(-byte_offset) }.cast()
    }
}

/// Add a node to `store` and return the corresponding `Edge`
///
/// The caller should ensure that no node is inserted twice
#[inline]
fn add_node<'id, DI: DiagramInfo>(
    store: &StoreSlab<'id, DI>,
    node: DI::InnerNode<'id>,
) -> AllocResult<[Own<Edge<'id, DI>>; 2]> {
    debug_assert!(node.load_rc(Relaxed) >= 2);
    let ptr = IntHandle::into_raw(store.add_item(node)).cast();
    let raw = Edge(ptr, PhantomData);
    // SAFETY: the initial reference count is (at least) 2
    Ok(unsafe { [Own::from_raw(raw), Own::from_raw(raw)] })
}

impl<'id, DI: DiagramInfo> Manager<'id, DI> {
    /// Get the pointer to `StoreInner` for this manager
    ///
    /// This method must not be called during of store and manager. After
    /// initialization, the returned pointer is safe to dereference.
    #[inline(always)]
    fn store_inner_ptr(&self) -> *const StoreInner<'id, DI> {
        let store_inner = self.store_inner;
        // We can simply get the store pointer by subtracting the offset of
        // `Manager` in `Store`. The only issue is that this violates Rust's
        // (proposed) aliasing rules. Hence, we only provide a hint that the
        // store's address can be computed without loading the value.
        let offset_ptr = StoreInner::from_manager_ptr(RwLock::from_data_ptr(self));
        // SAFETY: after initialization, the pointers are equal
        unsafe { std::hint::assert_unchecked(std::ptr::eq(store_inner, offset_ptr)) };
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
    fn store(&self) -> &StoreSlab<'id, DI> {
        let ptr = ArcSlab::from_data_ptr(self.store_inner_ptr());
        // SAFETY: After initialization, the pointer is guaranteed to be valid
        unsafe { &*ptr }
    }

    #[inline]
    fn terminal_manager(&self) -> *const DI::TerminalManager<'id> {
        let store_inner = self.store_inner_ptr();
        unsafe { &raw const ((*store_inner).terminal_manager) }
    }

    #[inline]
    pub(crate) fn drop_edge_common(edge: Own<Edge<'id, DI>>) {
        if edge.raw().is_inner() {
            // SAFETY: `edge` points to an inner node
            unsafe { Edge::drop_inner(edge) };
        } else {
            DI::TerminalManager::<'id>::drop_edge(edge);
        }
    }

    /// Forcibly drop the last edge, i.e., one that comes from the unique table
    ///
    /// # Safety
    ///
    /// - `self` must be untagged and point to an inner node
    /// - `self` must be the last reference to the node. Beware of relaxed
    ///   memory (e.g., use [`Acquire`] ordering to check that `this` is the
    ///   last reference).
    #[inline]
    unsafe fn force_drop_edge(edge: Own<Edge<'id, DI>>) {
        let handle: IntHandle<'id, DI::InnerNode<'id>, Self, DI::PageSizeInfo> = {
            let raw = edge.into_raw();
            debug_assert_eq!(raw.tag_bits(), 0);
            let ptr: NonNull<DI::InnerNode<'id>> = raw.0.cast();
            // SAFETY: By the type invariant, `ptr` was created from an
            // `IntHandle`, the caller ensures that the type/const arguments
            // of `IntHandle` match. Due to lifetime restrictions the `ArcSlab`
            // outlives the `IntHandle` we create.
            unsafe { IntHandle::from_raw(ptr) }
        };
        // SAFETY: `handle` is the last reference
        unsafe { IntHandle::force_into_inner(handle) }.drop_with(Self::drop_edge_common)
    }

    /// Drop the given edge
    ///
    /// # Safety
    ///
    /// `edge` must be untagged and point to an inner node
    #[inline(always)]
    unsafe fn drop_edge_from_unique_table(edge: Own<Edge<'id, DI>>) {
        let handle: IntHandle<'id, DI::InnerNode<'id>, Self, DI::PageSizeInfo> = {
            debug_assert_eq!(edge.raw().tag_bits(), 0);
            let ptr: NonNull<DI::InnerNode<'id>> = edge.into_raw().0.cast();
            // SAFETY: By the type invariant, `ptr` was created from an
            // `IntHandle`, the caller ensures that the type/const arguments
            // of `IntHandle` match. Due to lifetime restrictions the `ArcSlab`
            // outlives the `IntHandle` we create.
            unsafe { IntHandle::from_raw(ptr) }
        };
        IntHandle::drop_with(handle, |node| node.drop_with(Self::drop_edge_common))
    }
}

impl<'id, DI: DiagramInfo> Edge<'id, DI> {
    /// Mask corresponding to `TAG_BITS`
    const TAG_MASK: usize = {
        assert!(
            DI::EdgeTag::MAX_VALUE < (1 << DI::TAG_BITS),
            "`TAG_BITS` is too small"
        );
        // The mask we compute below has potentially less bits set than
        // `(1 << TAG_BITS) - 1`. If `ET::MAX_VALUE + 1` is a power of two, all
        // all bit patterns after applying the mask actually correspond to valid
        // bit patterns for the `ET`, so the compiler may optimize an
        // `assert!(value <= Self::MAX_VALUE)` in the `Tag::from_usize()`
        // implementation away, which would otherwise be required for safety.
        (DI::EdgeTag::MAX_VALUE + 1).next_power_of_two() - 1
    };

    /// `TAG_BITS` (for the user-defined edge tag) plus 1 for the distinction
    /// between inner and terminal nodes
    const ALL_TAG_BITS: u32 = {
        assert!(
            align_of::<DI::InnerNode<'id>>() >= 1 << (DI::TAG_BITS + 1),
            "`TAG_BITS` is too large"
        );
        DI::TAG_BITS + 1
    };

    /// Mask corresponding to `Self::ALL_TAG_BITS`
    const ALL_TAG_MASK: usize = (1 << Self::ALL_TAG_BITS) - 1;

    #[inline(always)]
    pub fn as_ptr(self) -> NonNull<()> {
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
    pub unsafe fn from_ptr(ptr: NonNull<()>) -> Own<Self> {
        // SAFETY: the caller guarantees permission
        unsafe { Own::from_raw(Self(ptr, PhantomData)) }
    }

    /// Get the address portion of the underlying pointer
    #[inline(always)]
    pub fn addr(&self) -> usize {
        self.0.as_ptr().addr()
    }

    #[inline]
    fn tag_bits(self) -> usize {
        self.addr() & Self::ALL_TAG_MASK
    }

    /// Get the inner node referenced by this edge
    ///
    /// # Safety
    ///
    /// `self` must be untagged and point to an inner node
    #[inline]
    unsafe fn inner_node_unchecked(this: Ref<'_, Self>) -> &DI::InnerNode<'id> {
        let raw = this.raw();
        debug_assert_eq!(raw.tag_bits(), 0);
        let ptr: NonNull<DI::InnerNode<'id>> = raw.0.cast();
        unsafe { ptr.as_ref() }
    }

    /// Clone an edge pointing to an inner node
    ///
    /// SAFETY: `edge` must be untagged and point to an inner node
    #[inline]
    unsafe fn clone_inner_unchecked(edge: Ref<'_, Self>) -> Own<Self> {
        unsafe { Self::inner_node_unchecked(edge) }.retain();
        unsafe { Own::from_raw(Self(edge.raw().0, PhantomData)) }
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
    /// SAFETY: `edge` must point to an inner node
    #[inline]
    unsafe fn drop_inner(edge: Own<Edge<'id, DI>>) {
        let edge = edge.into_raw();
        debug_assert!(edge.is_inner());
        let ptr: NonNull<DI::InnerNode<'id>> = edge.all_untagged_ptr().cast();
        // SAFETY: `self` points to an inner node and by the type invariant, we
        // have shared access. Also, `self` forgotten now.
        let _old_rc = unsafe { ptr.as_ref().release() };
        debug_assert!(_old_rc > 1);
    }

    /// Returns `true` if this edge points to an inner node
    #[inline]
    pub fn is_inner(self) -> bool {
        (self.addr() & (1 << DI::TAG_BITS)) == 0
    }

    /// Get the underlying pointer with all tag bits set to 0
    #[inline]
    fn all_untagged_ptr(self) -> NonNull<()> {
        let ptr = self.0.as_ptr().map_addr(|p| p & !Self::ALL_TAG_MASK);
        // SAFETY: the (tagged) pointer is `>= (1 << ALL_TAG_BITS)`
        unsafe { NonNull::new_unchecked(ptr) }
    }

    /// Get the underlying pointer tagged with `tag`
    #[inline]
    fn retag_ptr(self, tag: DI::EdgeTag) -> NonNull<()> {
        let tv = tag.as_usize();
        debug_assert!(tv <= DI::EdgeTag::MAX_VALUE);
        // Note that we assert `ET::MAX_VALUE <= Self::TAG_MASK` during the
        // computation of `Self::TAG_MASK`
        let ptr = self.0.as_ptr().map_addr(|p| (p & !Self::TAG_MASK) | tv);
        // SAFETY: even an untagged pointer is non-null
        unsafe { NonNull::new_unchecked(ptr) }
    }
}

unsafe impl<'id, DI: DiagramInfo> oxidd_core::Manager for Manager<'id, DI> {
    type Edge = Edge<'id, DI>;
    type EdgeTag = DI::EdgeTag;
    type InnerNode = DI::InnerNode<'id>;
    type InnerNodeValue = DI::InnerNodeValue;
    type Terminal = DI::TerminalNode;
    type TerminalRef<'a>
        = <DI::TerminalManager<'id> as TerminalManager<'id, DI>>::TerminalNodeRef<'a>
    where
        Self: 'a;
    type Rules = DI::Rules;

    type TerminalIterator<'a>
        = <DI::TerminalManager<'id> as TerminalManager<'id, DI>>::Iterator<'a>
    where
        Self: 'a;

    type NodeSet = NodeSet<DI::PageSizeInfo>;

    type LevelView<'a>
        = LevelView<'a, 'id, DI>
    where
        Self: 'a;
    type LevelIterator<'a>
        = LevelIter<'a, 'id, DI>
    where
        Self: 'a;

    #[inline]
    fn get_node<'a>(&'a self, edge: Ref<'a, Self::Edge>) -> Node<'a, Self> {
        let raw = edge.raw();
        if raw.is_inner() {
            let ptr: NonNull<Self::InnerNode> = raw.all_untagged_ptr().cast();
            // SAFETY: dereferencing untagged edges pointing to inner nodes is
            // safe
            Node::Inner(unsafe { ptr.as_ref() })
        } else {
            let terminal_manager = unsafe { &*self.terminal_manager() };
            Node::Terminal(terminal_manager.deref_edge(edge))
        }
    }

    #[inline]
    fn clone_edge(&self, edge: Ref<'_, Self::Edge>) -> Own<Self::Edge> {
        let raw = edge.raw();
        if raw.is_inner() {
            let ptr: NonNull<Self::InnerNode> = raw.all_untagged_ptr().cast();
            // SAFETY: dereferencing untagged edges pointing to inner nodes is
            // safe
            unsafe { ptr.as_ref() }.retain();
            // SAFETY: we have incremented the reference counter above
            unsafe { Own::from_raw(raw) }
        } else {
            DI::TerminalManager::<'id>::clone_edge(edge)
        }
    }

    #[inline]
    fn drop_edge(&self, edge: Own<Self::Edge>) {
        Self::drop_edge_common(edge);
    }

    #[track_caller]
    fn try_remove_node(&self, edge: Own<Self::Edge>, level: LevelNo) -> bool {
        if !edge.raw().is_inner() {
            DI::TerminalManager::<'id>::drop_edge(edge);
            debug_assert_eq!(level, LevelNo::MAX, "`level` does not match");
            return false;
        }

        let node_ptr: NonNull<Self::InnerNode> = edge.into_raw().all_untagged_ptr().cast();
        // SAFETY: `node_ptr` points to an inner node, and by the type invariant
        // of `Edge`, we have shared access.
        let node = unsafe { node_ptr.as_ref() };
        // SAFETY: `edge` is forgotten
        let old_rc = unsafe { node.release() };

        debug_assert_ne!(level, LevelNo::MAX, "`level` does not match");
        debug_assert!(
            (level as usize) < self.unique_table.len(),
            "`level` out of range"
        );
        debug_assert!(node.check_level(|l| l == level), "`level` does not match");
        debug_assert!(old_rc > 1);

        if old_rc != 2 || !self.reorder_gc_prepared {
            return false;
        }

        let Some(set) = self.unique_table.get(level as usize) else {
            return false;
        };
        let mut set = set.lock();

        // Read the reference count again: Another thread may have created an
        // edge between our `node.release()` call and `set.lock()`.
        let rc = node.load_rc(Acquire);
        debug_assert_ne!(rc, 0);
        if rc != 1 {
            return false;
        }

        let Some(edge) = set.remove(node) else {
            return false;
        };

        // SAFETY: Since `rc` is 1, this is the last reference. We use `Acquire`
        // order above and `Release` order when decrementing reference counters,
        // so we have exclusive node access now. Additionally, `edge` is
        // untagged and points to an inner node.
        unsafe { Self::force_drop_edge(edge) };

        true
    }

    #[inline]
    fn num_inner_nodes(&self) -> usize {
        self.store().num_items()
    }

    #[inline(always)]
    fn num_levels(&self) -> LevelNo {
        self.unique_table.len() as LevelNo
    }

    #[inline(always)]
    fn num_named_vars(&self) -> VarNo {
        self.var_name_map.named_count() as VarNo
    }

    #[track_caller]
    fn add_vars(&mut self, additional: VarNo) -> Range<VarNo> {
        let len = self.unique_table.len() as VarNo;
        let new_len = len.checked_add(additional).expect("too many variables");
        let range = len as VarNo..new_len as VarNo;

        self.data.pre_reorder(self);
        DI::ManagerData::<'id>::pre_reorder_mut(self);

        self.unique_table
            .resize_with(new_len as usize, || Mutex::new(LevelViewSet::default()));
        self.var_level_map.extend(additional);
        self.var_name_map.add_unnamed(additional);

        debug_assert_eq!(new_len as usize, self.unique_table.len());
        debug_assert_eq!(new_len as usize, self.var_level_map.len());
        debug_assert_eq!(new_len, self.var_name_map.len());

        self.data.post_reorder(self);
        DI::ManagerData::<'id>::post_reorder_mut(self);

        range
    }

    #[track_caller]
    fn add_named_vars<S: Into<String>>(
        &mut self,
        names: impl IntoIterator<Item = S>,
    ) -> Result<Range<VarNo>, DuplicateVarName> {
        self.data.pre_reorder(self);
        DI::ManagerData::<'id>::pre_reorder_mut(self);

        let len = self.var_name_map.len();
        let mut guard = scopeguard::guard(self, |this| {
            // This block is executed whenever `guard` gets dropped, i.e., even
            // if iterating over `names` or converting a value of type `S` into
            // a `String` panics. This way, we ensure that the manager's state
            // remains consistent.
            let new_len = this.var_name_map.len();
            this.unique_table
                .resize_with(new_len as usize, || Mutex::new(LevelViewSet::default()));
            this.var_level_map.extend((new_len - len) as VarNo);

            debug_assert_eq!(new_len as usize, this.unique_table.len());
            debug_assert_eq!(new_len as usize, this.var_level_map.len());
            debug_assert_eq!(new_len, this.var_name_map.len());

            this.data.post_reorder(this);
            DI::ManagerData::<'id>::post_reorder_mut(this);
        });

        let mut names = names.into_iter();
        let range = guard.var_name_map.add_named(names.by_ref())?;
        drop(guard);

        if names.next().is_some() {
            // important: panic only after dropping `guard`
            panic!("too many variables");
        }

        Ok(range)
    }

    #[track_caller]
    fn add_named_vars_from_map(
        &mut self,
        map: VarNameMap,
    ) -> Result<Range<VarNo>, DuplicateVarName> {
        if !self.var_name_map.is_empty() {
            return self.add_named_vars(map.into_names_iter());
        }

        self.data.pre_reorder(self);
        DI::ManagerData::<'id>::pre_reorder_mut(self);

        let n = map.len();
        self.unique_table
            .resize_with(n as usize, || Mutex::new(LevelViewSet::default()));
        self.var_level_map.extend(n);
        self.var_name_map = map;

        debug_assert_eq!(n as usize, self.unique_table.len());
        debug_assert_eq!(n as usize, self.var_level_map.len());
        debug_assert_eq!(n, self.var_name_map.len());

        self.data.post_reorder(self);
        DI::ManagerData::<'id>::post_reorder_mut(self);

        Ok(0..n)
    }

    #[track_caller]
    #[inline(always)]
    fn var_name(&self, var: VarNo) -> &str {
        self.var_name_map.var_name(var)
    }

    #[track_caller]
    #[inline(always)]
    fn set_var_name(
        &mut self,
        var: VarNo,
        name: impl Into<String>,
    ) -> Result<(), DuplicateVarName> {
        self.var_name_map.set_var_name(var, name)
    }

    #[inline(always)]
    fn name_to_var(&self, name: impl AsRef<str>) -> Option<VarNo> {
        self.var_name_map.name_to_var(name)
    }

    #[track_caller]
    #[inline(always)]
    fn var_to_level(&self, var: VarNo) -> LevelNo {
        self.var_level_map.var_to_level(var)
    }

    #[track_caller]
    #[inline(always)]
    fn level_to_var(&self, level: LevelNo) -> VarNo {
        self.var_level_map.level_to_var(level)
    }

    #[track_caller]
    #[inline(always)]
    fn level(&self, no: LevelNo) -> Self::LevelView<'_> {
        LevelView {
            store: self.store(),
            var_level_map: &self.var_level_map,
            allow_node_removal: self.reorder_gc_prepared,
            level: no,
            set: self.unique_table[no as usize].lock(),
        }
    }

    #[inline(always)]
    unsafe fn level_unchecked(&self, no: LevelNo) -> Self::LevelView<'_> {
        LevelView {
            store: self.store(),
            var_level_map: &self.var_level_map,
            allow_node_removal: self.reorder_gc_prepared,
            level: no,
            // SAFETY: ensured by caller
            set: unsafe { self.unique_table.get_unchecked(no as usize) }.lock(),
        }
    }

    #[inline]
    fn levels(&self) -> Self::LevelIterator<'_> {
        LevelIter {
            store: self.store(),
            var_level_map: &self.var_level_map,
            allow_node_removal: self.reorder_gc_prepared,
            level_front: 0,
            level_back: self.unique_table.len() as LevelNo,
            it: self.unique_table.iter(),
        }
    }

    #[inline]
    fn get_terminal(&self, terminal: Self::Terminal) -> AllocResult<Own<Self::Edge>> {
        unsafe { DI::TerminalManager::<'id>::get(self.terminal_manager(), terminal) }
    }

    #[inline]
    fn num_terminals(&self) -> usize {
        let terminal_manager = unsafe { &*self.terminal_manager() };
        terminal_manager.len()
    }

    #[inline]
    fn terminals(&self) -> Self::TerminalIterator<'_> {
        unsafe { DI::TerminalManager::<'id>::iter(self.terminal_manager()) }
    }

    fn gc(&self) -> usize {
        if !self.gc_ongoing.try_lock() {
            // We don't want multiple garbage collections at the same time.
            return 0;
        }
        self.gc_count.fetch_add(1, Relaxed);
        let guard = AbortOnDrop("Garbage collection panicked.");
        if !self.reorder_gc_prepared {
            self.data.pre_gc(self);
        }

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

        if !self.reorder_gc_prepared {
            // SAFETY: We called `pre_gc`, the garbage collection is done.
            unsafe { self.data.post_gc(self) };
        }
        self.gc_ongoing.unlock();
        guard.defuse();
        collected
    }

    fn reorder<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        if self.reorder_gc_prepared {
            // Reordering was already prepared (nested `reorder` call).
            // Important: We must return here to not finalize the reordering
            // before the parent `reorder` closure returns.
            return f(self);
        }
        let guard = AbortOnDrop("Reordering panicked.");

        self.data.pre_gc(self);
        self.reorder_gc_prepared = true;
        self.data.pre_reorder(self);
        DI::ManagerData::<'id>::pre_reorder_mut(self);

        let res = f(self);

        self.data.post_reorder(self);
        DI::ManagerData::<'id>::post_reorder_mut(self);
        self.reorder_gc_prepared = false;
        // SAFETY: We called `pre_gc()` and the reordering is done.
        unsafe { self.data.post_gc(self) };

        guard.defuse();
        // Depending on the reordering implementation, garbage collections are
        // preformed, but not necessarily through `Self::gc`. So we increment
        // the GC count here.
        *self.gc_count.get_mut() += 1;
        self.reorder_count += 1;
        res
    }

    #[inline]
    fn gc_count(&self) -> u64 {
        self.gc_count.load(Relaxed)
    }

    #[inline]
    fn reorder_count(&self) -> u64 {
        self.reorder_count
    }
}

impl<'id, DI: DiagramInfo> oxidd_core::HasWorkers for Manager<'id, DI> {
    type WorkerPool = crate::workers::Workers;

    #[inline]
    fn workers(&self) -> &Self::WorkerPool {
        &self.store().data().workers
    }
}

pub struct LevelIter<'a, 'id, DI: DiagramInfo> {
    store: &'a StoreSlab<'id, DI>,
    var_level_map: &'a VarLevelMap,
    /// SAFETY invariant: If set to true, a garbage collection is prepared
    /// (i.e., there are no "weak" edges).
    allow_node_removal: bool,
    level_front: LevelNo,
    level_back: LevelNo,
    it: std::slice::Iter<'a, Mutex<LevelViewSet<'id, DI>>>,
}

impl<'a, 'id, DI: DiagramInfo> Iterator for LevelIter<'a, 'id, DI> {
    type Item = LevelView<'a, 'id, DI>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let mutex = self.it.next()?;
        let level = self.level_front;
        self.level_front += 1;
        Some(LevelView {
            store: self.store,
            var_level_map: self.var_level_map,
            allow_node_removal: self.allow_node_removal,
            level,
            set: mutex.lock(),
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'id, DI: DiagramInfo> ExactSizeIterator for LevelIter<'_, 'id, DI> {
    #[inline]
    fn len(&self) -> usize {
        self.it.len()
    }
}

impl<'a, 'id, DI: DiagramInfo> FusedIterator for LevelIter<'a, 'id, DI>
where
    DI: DiagramInfo,
    std::slice::Iter<'a, Mutex<LevelViewSet<'id, DI>>>: FusedIterator,
{
}

impl<'id, DI: DiagramInfo> DoubleEndedIterator for LevelIter<'_, 'id, DI> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let mutex = self.it.next_back()?;
        self.level_back -= 1;
        Some(LevelView {
            store: self.store,
            var_level_map: self.var_level_map,
            allow_node_removal: self.allow_node_removal,
            level: self.level_back,
            set: mutex.lock(),
        })
    }
}

impl<DI: DiagramInfo> oxidd_core::Edge for Edge<'_, DI> {
    type Tag = DI::EdgeTag;

    #[inline]
    fn with_tag(self, tag: Self::Tag) -> Self {
        Self(self.retag_ptr(tag), PhantomData)
    }

    #[inline]
    fn tag(self) -> Self::Tag {
        DI::EdgeTag::from_usize(self.addr() & Self::TAG_MASK)
    }

    #[inline]
    fn node_id(self) -> oxidd_core::NodeID {
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
struct LevelViewSet<'id, DI: DiagramInfo>(RawTable<Own<Edge<'id, DI>>>);

#[inline]
fn hash_node<N: NodeBase>(node: &N) -> u64 {
    let mut hasher = FxHasher::default();
    node.hash(&mut hasher);
    hasher.finish()
}

impl<'id, DI: DiagramInfo> LevelViewSet<'id, DI> {
    /// Get the number of nodes on this level
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Reserve space for `additional` nodes on this level
    #[inline]
    fn reserve(&mut self, additional: usize) {
        // SAFETY: The hash table only contains untagged edges referencing inner
        // nodes.
        self.0.reserve(additional)
    }

    #[inline]
    fn get(
        &self,
        children: &[Ref<'_, Edge<'id, DI>>],
        value: &DI::InnerNodeValue,
    ) -> Option<Ref<'_, Edge<'id, DI>>> {
        let mut hasher = FxHasher::default();
        crate::util::hash_children(children, &mut hasher);
        let hash = hasher.finish();

        let edge = self.0.get(hash, move |edge| {
            let edge = edge.borrowed();
            // SAFETY: The hash table only contains untagged edges referencing
            // inner nodes.
            let node = unsafe { Edge::inner_node_unchecked(edge) };
            node.children().len() == children.len()
                && node.children().zip(children).all(|(a, b)| a == *b)
                && node.value() == value
        })?;
        Some(edge.borrowed())
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
    unsafe fn insert<const CHECKS: bool>(
        &mut self,
        edge: Own<Edge<'id, DI>>,
        expected_level: LevelNo,
    ) -> bool {
        let mut raw = edge.into_raw();
        if const { CHECKS || cfg!(debug_assertions) } {
            assert_eq!(
                raw.tag_bits(),
                0,
                "can only insert untagged edges pointing to inner nodes"
            );
        } else {
            raw.0 = raw.all_untagged_ptr();
        }

        // SAFETY: we potentially change `edge`'s tag and forget the original
        // `edge`
        let edge = ManuallyDrop::new(unsafe { Own::from_raw(raw) });
        let borrowed = edge.borrowed();

        // SAFETY: `edge` is now untagged. It refers to an inner node as ensured
        // by the caller or asserted, depending on `CHECKS`.
        let node = unsafe { Edge::inner_node_unchecked(borrowed) };
        if CHECKS {
            node.assert_level_matches(expected_level);
        }

        let hash = hash_node(node);
        // SAFETY (next 2): The hash table only contains untagged edges
        // referencing inner nodes.
        match self
            .0
            .find_or_find_insert_slot(hash, |e| std::ptr::eq(e.raw().0.as_ptr().cast(), node))
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
                let edge = ManuallyDrop::into_inner(edge);
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
    #[inline(always)]
    fn get_or_insert(
        &mut self,
        node: DI::InnerNode<'id>,
        insert: impl FnOnce(DI::InnerNode<'id>) -> AllocResult<[Own<Edge<'id, DI>>; 2]>,
    ) -> AllocResult<Own<Edge<'id, DI>>> {
        let hash = hash_node(&node);
        match self.0.find_or_find_insert_slot(hash, |edge| {
            // SAFETY: The hash table only contains untagged edges referencing
            // inner nodes.
            *unsafe { Edge::inner_node_unchecked(edge.borrowed()) } == node
        }) {
            Ok(slot) => {
                node.drop_with(Manager::<'id, DI>::drop_edge_common);
                // SAFETY: `slot` was returned by `find_or_find_insert_slot`. We
                // have exclusive access to the hash table and
                // did not modify it in between.
                let edge = unsafe { self.0.get_at_slot_unchecked(slot) }.borrowed();
                // SAFETY: All edges in the table are untagged and refer to
                // inner nodes.
                Ok(unsafe { Edge::clone_inner_unchecked(edge) })
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
                let edge = edge.borrowed();
                // SAFETY: All edges in unique tables are untagged and point to
                // inner nodes.
                unsafe { Edge::inner_node_unchecked(edge) }.load_rc(Acquire) != 1
            },
            |edge| {
                // SAFETY: Since `rc` is 1, this is the last reference. We use
                // `Acquire` order above and `Release` order when decrementing
                // reference counters, so we have exclusive node access now.
                // Additionally, `edge` is untagged and points to an inner node.
                // The type/const arguments match the ones from the manager.
                unsafe { Manager::<DI>::force_drop_edge(edge) };
            },
        );
    }

    /// Remove `node` from the level view
    ///
    /// Note that `node` must refer to the node in the node store, this method
    /// uses referential equality.
    ///
    /// Returns the edge from the unique table entry if `node` was found.
    #[inline]
    fn remove(&mut self, node: &DI::InnerNode<'id>) -> Option<Own<Edge<'id, DI>>> {
        self.0.remove_entry(hash_node(node), |e| {
            std::ptr::eq(e.raw().0.as_ptr().cast(), node)
        })
    }

    /// Remove the node referenced by `edge` if it is contained in this level
    /// view and `edge` is the only edge besides the one in the level view
    /// referring to this node.
    ///
    /// Returns whether the node was removed
    ///
    /// # Safety
    ///
    /// `edge` must refer to an inner node. Further, there must not be any
    /// "weak" edges, i.e., edges where the reference count is not materialized
    /// (apply cache implementations exploit this).
    #[inline]
    unsafe fn try_remove(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        let node_ptr: NonNull<DI::InnerNode<'id>> = edge.into_raw().all_untagged_ptr().cast();
        // SAFETY: `node_ptr` points to an inner node, and by the type invariant
        // of `Own<Edge>`, we have shared access.
        let node = unsafe { node_ptr.as_ref() };
        // SAFETY: `edge` is forgotten
        let old_rc = unsafe { node.release() };
        debug_assert!(old_rc > 1);
        if old_rc != 2 {
            return false;
        }

        let Some(edge) = self.remove(node) else {
            return false;
        };

        std::sync::atomic::fence(Acquire);

        // SAFETY: Since `rc` is 1, this is the last reference. We use `Acquire`
        // order above and `Release` order when decrementing reference
        // counters, so we have exclusive node access now. Additionally,
        // the caller guarantees that there are no weak edges.
        unsafe { Manager::<DI>::force_drop_edge(edge) };

        true
    }

    /// Iterate over all edges pointing to nodes in the set
    #[inline]
    fn iter(&self) -> LevelViewIter<'_, 'id, DI> {
        LevelViewIter(self.0.iter())
    }

    /// Iterator that consumes all [`Edge`]s in the set
    #[inline]
    fn drain(&mut self) -> linear_hashtbl::raw::Drain<'_, Own<Edge<'id, DI>>> {
        self.0.drain()
    }
}

impl<DI: DiagramInfo> Drop for LevelViewSet<'_, DI> {
    #[inline]
    fn drop(&mut self) {
        // If the nodes need drop, this is handled by the `Manager` `Drop` impl
        // or the `TakenLevelView` `Drop` impl, respectively.
        self.0.reset_no_drop();
    }
}

impl<DI: DiagramInfo> Default for LevelViewSet<'_, DI> {
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<'id, DI: DiagramInfo> IntoIterator for LevelViewSet<'id, DI> {
    type Item = Own<Edge<'id, DI>>;

    #[cfg(not(feature = "hugealloc"))]
    type IntoIter = linear_hashtbl::raw::IntoIter<Own<Edge<'id, DI>>, usize>;
    #[cfg(feature = "hugealloc")]
    type IntoIter = linear_hashtbl::raw::IntoIter<Own<Edge<'id, DI>>, usize, hugealloc::HugeAlloc>;

    fn into_iter(self) -> Self::IntoIter {
        let this = ManuallyDrop::new(self);
        // SAFETY: We move out of `this` (and forget `this`)
        let set = unsafe { std::ptr::read(&this.0) };
        set.into_iter()
    }
}

/// Level view provided by the unique table
pub struct LevelView<'a, 'id, DI: DiagramInfo> {
    store: &'a StoreSlab<'id, DI>,
    var_level_map: &'a VarLevelMap,
    /// SAFETY invariant: If set to true, a garbage collection is prepared
    /// (i.e., there are no "weak" edges).
    allow_node_removal: bool,
    level: LevelNo,
    set: MutexGuard<'a, LevelViewSet<'id, DI>>,
}

unsafe impl<'a, 'id, DI: DiagramInfo> oxidd_core::LevelView<Edge<'id, DI>, DI::InnerNode<'id>>
    for LevelView<'a, 'id, DI>
{
    type Iterator<'b>
        = LevelViewIter<'b, 'id, DI>
    where
        Self: 'b;

    type Taken = TakenLevelView<'a, 'id, DI>;

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
    fn get(
        &self,
        children: &[Ref<'_, Edge<'id, DI>>],
        value: &DI::InnerNodeValue,
    ) -> Option<Ref<'_, Edge<'id, DI>>> {
        self.set.get(children, value)
    }

    #[inline]
    fn insert(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        // SAFETY: `CHECKS` is true
        unsafe { self.set.insert::<true>(edge, self.level) }
    }

    #[inline]
    unsafe fn insert_unchecked(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        // SAFETY: the caller ensures that `edge` points to an inner node at the
        // respective level
        unsafe { self.set.insert::<false>(edge, self.level) }
    }

    #[inline(always)]
    fn get_or_insert(&mut self, node: DI::InnerNode<'id>) -> AllocResult<Own<Edge<'id, DI>>> {
        node.assert_level_matches(self.level);
        // SAFETY: we checked the level as required
        unsafe { self.get_or_insert_unchecked(node) }
    }

    unsafe fn get_or_insert_unchecked(
        &mut self,
        node: DI::InnerNode<'id>,
    ) -> AllocResult<Own<Edge<'id, DI>>> {
        // No need to check if the children of `node` are stored in `self.store`
        // due to lifetime restrictions.
        LevelViewSet::get_or_insert(&mut *self.set, node, |node| add_node(self.store, node))
    }

    #[inline]
    fn gc(&mut self) {
        if self.allow_node_removal {
            // SAFETY: By invariant, node removal is allowed.
            unsafe { self.set.gc() };
        }
    }

    #[inline]
    fn try_remove(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        if !edge.raw().is_inner() {
            DI::TerminalManager::<'id>::drop_edge(edge);
            return false;
        }
        if !self.allow_node_removal {
            // SAFETY: `edge` points to an inner node
            unsafe { Edge::drop_inner(edge) };
            return false;
        }
        // SAFETY: `edge` refers to an inner node. By invariant, node removal is
        // allowed.
        unsafe { self.set.try_remove(edge) }
    }

    #[inline]
    unsafe fn swap(&mut self, other: &mut Self) {
        self.var_level_map.swap_levels(self.level, other.level);
        std::mem::swap(&mut *self.set, &mut *other.set);
    }

    #[inline]
    fn iter(&self) -> Self::Iterator<'_> {
        self.set.iter()
    }

    #[inline(always)]
    fn take(&mut self) -> Option<Self::Taken> {
        if self.allow_node_removal {
            Some(TakenLevelView {
                store: self.store,
                var_level_map: self.var_level_map,
                level: self.level,
                set: std::mem::take(&mut self.set),
            })
        } else {
            None
        }
    }
}

/// Owned level view provided by the unique table
//
// SAFETY invariant: whenever a TakenLevelView is live, a garbage collection is
// prepared (i.e., there are no "weak" edges).
//
// Note that due to lifetime restrictions there is no way to have a
// non-empty `TakenLevelView` when the associated `Manager` has
// `reorder_gc_prepared` set to `false`.
pub struct TakenLevelView<'a, 'id, DI: DiagramInfo> {
    store: &'a StoreSlab<'id, DI>,
    var_level_map: &'a VarLevelMap,
    level: LevelNo,
    set: LevelViewSet<'id, DI>,
}

unsafe impl<'id, DI: DiagramInfo> oxidd_core::LevelView<Edge<'id, DI>, DI::InnerNode<'id>>
    for TakenLevelView<'_, 'id, DI>
{
    type Iterator<'b>
        = LevelViewIter<'b, 'id, DI>
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
    fn get(
        &self,
        children: &[Ref<'_, Edge<'id, DI>>],
        value: &DI::InnerNodeValue,
    ) -> Option<Ref<'_, Edge<'id, DI>>> {
        self.set.get(children, value)
    }

    #[inline]
    fn insert(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        // SAFETY: `CHECKS` is true
        unsafe { self.set.insert::<true>(edge, self.level) }
    }

    #[inline]
    unsafe fn insert_unchecked(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        // SAFETY: the caller ensures that `edge` points to an inner node at the
        // respective level
        unsafe { self.set.insert::<false>(edge, self.level) }
    }

    #[inline(always)]
    fn get_or_insert(&mut self, node: DI::InnerNode<'id>) -> AllocResult<Own<Edge<'id, DI>>> {
        node.assert_level_matches(self.level);
        // SAFETY: we checked the level as required
        unsafe { self.get_or_insert_unchecked(node) }
    }

    unsafe fn get_or_insert_unchecked(
        &mut self,
        node: DI::InnerNode<'id>,
    ) -> AllocResult<Own<Edge<'id, DI>>> {
        // No need to check if the children of `node` are stored in `self.store`
        // due to lifetime restrictions.
        self.set
            .get_or_insert(node, |node| add_node(self.store, node))
    }

    #[inline]
    fn gc(&mut self) {
        // SAFETY: By invariant, node removal is allowed.
        unsafe { self.set.gc() };
    }

    #[inline]
    fn try_remove(&mut self, edge: Own<Edge<'id, DI>>) -> bool {
        if !edge.raw().is_inner() {
            DI::TerminalManager::<'id>::drop_edge(edge);
            return false;
        }
        // SAFETY: `edge` refers to an inner node. Called from inside the
        // closure of `Manager::reorder()`, hence there are no "weak"
        // edges.
        unsafe { self.set.try_remove(edge) }
    }

    #[inline]
    unsafe fn swap(&mut self, other: &mut Self) {
        self.var_level_map.swap_levels(self.level, other.level);
        std::mem::swap(&mut self.set, &mut other.set);
    }

    #[inline]
    fn iter(&self) -> Self::Iterator<'_> {
        self.set.iter()
    }

    #[inline(always)]
    fn take(&mut self) -> Option<Self::Taken> {
        Some(Self {
            store: self.store,
            var_level_map: self.var_level_map,
            level: self.level,
            set: std::mem::take(&mut self.set),
        })
    }
}

impl<'id, DI: DiagramInfo> Drop for TakenLevelView<'_, 'id, DI> {
    fn drop(&mut self) {
        for edge in self.set.drain() {
            // SAFETY: `edge` is untagged and points to an inner node, the type
            // parameters match
            unsafe { Manager::<'id, DI>::drop_edge_from_unique_table(edge) };
        }
    }
}

// --- Level Views: Iterator ---------------------------------------------------

/// Iterator over entries (as [`Edge`]s) of a level view
pub struct LevelViewIter<'a, 'id, DI: DiagramInfo>(
    linear_hashtbl::raw::Iter<'a, Own<Edge<'id, DI>>>,
);

impl<'a, 'id, DI: DiagramInfo> Iterator for LevelViewIter<'a, 'id, DI> {
    type Item = Ref<'a, Edge<'id, DI>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.borrowed())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, 'id, DI: DiagramInfo> ExactSizeIterator for LevelViewIter<'a, 'id, DI> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, 'id, DI> FusedIterator for LevelViewIter<'a, 'id, DI>
where
    DI: DiagramInfo,
    linear_hashtbl::raw::Iter<'a, Edge<'id, DI>, u32>: FusedIterator,
{
}

// === Node Set ================================================================

/// Node set implementation using [bit sets][FixedBitSet]
///
/// Since nodes are stored on large pages, we can use one bit vector per page.
/// This reduces space consumption dramatically and increases the performance.
#[derive_where(Clone, PartialEq, Eq, Default)]
pub struct NodeSet<P: PageSizeInfo> {
    len: usize,
    data: HashMap<usize, FixedBitSet, BuildHasherDefault<FxHasher>>,
    phantom: PhantomData<P>,
}

impl<P: PageSizeInfo> NodeSet<P> {
    const fn nodes_per_page<DI: DiagramInfo>() -> usize {
        P::PAGE_SIZE >> (DI::TAG_BITS + 1)
    }

    #[inline]
    fn page_offset<DI: DiagramInfo>(edge: Edge<'_, DI>) -> (usize, usize) {
        let nodes_per_page = const { Self::nodes_per_page::<DI>() };
        let node_id = edge.0.as_ptr().addr() >> DI::TAG_BITS;
        let page = node_id / nodes_per_page;
        let offset = node_id % nodes_per_page;
        (page, offset)
    }
}

impl<'id, DI: DiagramInfo, P: PageSizeInfo> oxidd_core::util::NodeSet<Edge<'id, DI>>
    for NodeSet<P>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    fn insert(&mut self, edge: Ref<'_, Edge<'id, DI>>) -> bool {
        let (page, offset) = Self::page_offset(edge.raw());
        match self.data.entry(page) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let page = e.get_mut();
                if page.contains(offset) {
                    false
                } else {
                    page.insert(offset);
                    self.len += 1;
                    true
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut page = FixedBitSet::with_capacity(const { Self::nodes_per_page::<DI>() });
                page.insert(offset);
                e.insert(page);
                self.len += 1;
                true
            }
        }
    }

    #[inline]
    fn contains(&self, edge: Ref<'_, Edge<'id, DI>>) -> bool {
        let (page, offset) = Self::page_offset(edge.raw());
        match self.data.get(&page) {
            Some(page) => page.contains(offset),
            None => false,
        }
    }

    fn remove(&mut self, edge: Ref<'_, Edge<'id, DI>>) -> bool {
        let (page, offset) = Self::page_offset(edge.raw());
        match self.data.get_mut(&page) {
            Some(page) if page.contains(offset) => {
                page.remove(offset);
                self.len -= 1;
                true
            }
            _ => false,
        }
    }
}

// === `ManagerRef` & Creation =================================================

impl<DI: DiagramInfo> ManagerRef<DI> {
    /// Convert `self` into a raw pointer, e.g. for usage in a foreign function
    /// interface.
    ///
    /// This method does not change any reference counters. To avoid a memory
    /// leak, use [`Self::from_raw()`] to convert the pointer back into a
    /// `ManagerRef`.
    #[inline(always)]
    pub fn into_raw(self) -> *const std::ffi::c_void {
        ArcSlabRef::into_raw(self.0).as_ptr().cast()
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
        let ptr = NonNull::new(raw.cast_mut().cast()).expect("expected a non-null pointer");
        // SAFETY: Invariants are upheld by the caller.
        Self(unsafe { ArcSlabRef::from_raw(ptr) })
    }
}

impl<'a, 'id, DI: DiagramInfo> From<&'a Manager<'id, DI>> for ManagerRef<DI> {
    #[inline]
    fn from(manager: &'a Manager<'id, DI>) -> Self {
        manager.store().retain();
        let store_ptr: *const StoreSlab<'id, DI> =
            StoreSlab::<'id, DI>::from_data_ptr(manager.store_inner_ptr());
        let store_ptr: *mut StoreSlab<'static, DI> = store_ptr.cast_mut().cast();
        Self(unsafe { ArcSlabRef::from_raw(NonNull::new_unchecked(store_ptr)) })
    }
}

impl<DI: DiagramInfo> oxidd_core::ManagerRef for ManagerRef<DI> {
    type Manager<'id> = Manager<'id, DI>;

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

impl<DI: DiagramInfo> oxidd_core::HasWorkers for ManagerRef<DI> {
    type WorkerPool = crate::workers::Workers;

    #[inline]
    fn workers(&self) -> &Self::WorkerPool {
        &self.0.data().workers
    }
}

/// Create a new manager
pub fn new_manager<DI: DiagramInfo>(
    data: DI::ManagerData<'static>,
    threads: u32,
) -> ManagerRef<DI> {
    // Evaluate a few constants for assertions
    let _ = Edge::<'static, DI>::TAG_MASK;
    let _ = Edge::<'static, DI>::ALL_TAG_BITS;

    let arc = unsafe { ArcSlab::new_with(|slot| StoreInner::init_in(slot, data, threads)) };

    // initialize the manager data
    {
        let guard = &mut arc.data().manager.exclusive();
        let manager = &mut *guard;
        DI::ManagerData::<'static>::init(&manager.data, manager);
        DI::ManagerData::<'static>::init_mut(manager);
    }

    ManagerRef(arc)
}

// === Functions ===============================================================

#[repr(transparent)]
#[derive_where(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Function<DI: DiagramInfo>(NonNull<()>, PhantomData<DI>);

unsafe impl<DI: DiagramInfo> Send for Function<DI> {}

unsafe impl<DI: DiagramInfo> Sync for Function<DI> {}

impl<DI: DiagramInfo> Function<DI> {
    #[inline]
    fn store(&self) -> NonNull<StoreSlab<'static, DI>> {
        let edge: Edge<'static, DI> = Edge(self.0, PhantomData);
        if edge.is_inner() {
            let ptr: NonNull<DI::InnerNode<'static>> = edge.all_untagged_ptr().cast();
            let handle = ManuallyDrop::new(unsafe { ExtHandle::from_raw(ptr) });
            NonNull::from(ExtHandle::slab(&*handle))
        } else {
            let ptr = ArcSlab::from_data_ptr(StoreInner::from_terminal_manager_ptr(
                TerminalManager::terminal_manager(unsafe { Ref::from_raw(edge) }).as_ptr(),
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
        ptr.as_ptr().cast()
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
        let ptr = NonNull::new(raw.cast_mut().cast()).expect("expected a non-null pointer");
        Self(ptr, PhantomData)
    }
}

impl<DI: DiagramInfo> Clone for Function<DI> {
    #[inline]
    fn clone(&self) -> Self {
        let edge: Edge<'static, DI> = Edge(self.0, PhantomData);
        if edge.is_inner() {
            let ptr: NonNull<DI::InnerNode<'static>> = edge.all_untagged_ptr().cast();
            // SAFETY: dereferencing untagged edges pointing to inner nodes is
            // safe
            unsafe { ptr.as_ref() }.retain();
        } else {
            std::mem::forget(DI::TerminalManager::<'static>::clone_edge(unsafe {
                Ref::from_raw(edge)
            }));
        }
        let store = self.store();
        unsafe { store.as_ref() }.retain();
        Self(self.0, PhantomData)
    }
}

impl<DI: DiagramInfo> Drop for Function<DI> {
    fn drop(&mut self) {
        let store = self.store();
        // SAFETY: we are dropping `self`, so we can take an owned edge out
        unsafe {
            Manager::<'static, DI>::drop_edge_common(Own::from_raw(Edge(self.0, PhantomData)))
        };

        // We own a reference count, `store` is valid, and we do not use the
        // `store` pointer afterwards.
        unsafe { ArcSlab::release(store) };
    }
}

unsafe impl<DI: DiagramInfo> oxidd_core::function::Function for Function<DI> {
    const REPR_ID: &str = "<none>";

    type Manager<'id> = Manager<'id, DI>;
    type ManagerRef = ManagerRef<DI>;

    #[inline]
    fn from_edge<'id>(manager: &Self::Manager<'id>, edge: OwnEdgeOfFunc<'id, Self>) -> Self {
        manager.store().retain();
        Self(edge.into_raw().0, PhantomData)
    }

    #[inline]
    fn as_edge<'id>(&self, manager: &Self::Manager<'id>) -> Ref<'_, EdgeOfFunc<'id, Self>> {
        assert!(std::ptr::eq(self.store().as_ptr().cast(), manager.store()));
        // SAFETY: a `Function` provides shared permission for the referenced
        // node and the lifetime of the returned `Ref` is bounded by the
        // lifetime of `self`
        unsafe { Ref::from_raw(Edge(self.0, PhantomData)) }
    }

    #[inline]
    fn into_edge<'id>(self, manager: &Self::Manager<'id>) -> OwnEdgeOfFunc<'id, Self> {
        let store = manager.store();
        assert!(std::ptr::eq(self.store().as_ptr().cast(), store));
        let this = ManuallyDrop::new(self);
        // SAFETY: we are dropping the manager reference part of `this`
        unsafe { ArcSlab::release(NonNull::from(store)) };
        // SAFETY: we are converting the owned `Function` into an `Own<Edge>`
        unsafe { Own::from_raw(Edge(this.0, PhantomData)) }
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
        F: for<'id> FnOnce(&Self::Manager<'id>, Ref<'_, EdgeOfFunc<'id, Self>>) -> T,
    {
        let store_ptr = self.store();
        f(
            &*unsafe { store_ptr.as_ref() }.data().manager.shared(),
            // SAFETY: a `Function` provides shared permission for the referenced
            // node
            unsafe { Ref::from_raw(Edge(self.0, PhantomData)) },
        )
    }

    #[inline]
    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>, Ref<'_, EdgeOfFunc<'id, Self>>) -> T,
    {
        let store_ptr = self.store();
        f(
            &mut *unsafe { store_ptr.as_ref() }.data().manager.exclusive(),
            // SAFETY: a `Function` provides shared permission for the referenced
            // node
            unsafe { Ref::from_raw(Edge(self.0, PhantomData)) },
        )
    }
}

// === Additional Trait Implementations ========================================

impl<'id, DI, O: Copy> HasApplyCache<Self, O> for Manager<'id, DI>
where
    DI: DiagramInfo,
    DI::ManagerData<'id>: HasApplyCache<Self, O>,
{
    type ApplyCache = <DI::ManagerData<'id> as HasApplyCache<Self, O>>::ApplyCache;

    #[inline]
    fn apply_cache(&self) -> &Self::ApplyCache {
        self.data.apply_cache()
    }

    #[inline]
    fn apply_cache_mut(&mut self) -> &mut Self::ApplyCache {
        self.data.apply_cache_mut()
    }
}

impl<'id, DI, T> AsRef<T> for Manager<'id, DI>
where
    DI: DiagramInfo,
    DI::ManagerData<'id>: AsRef<T>,
{
    #[inline(always)]
    fn as_ref(&self) -> &T {
        self.data.as_ref()
    }
}

impl<'id, DI, T> AsMut<T> for Manager<'id, DI>
where
    DI: DiagramInfo,
    DI::ManagerData<'id>: AsMut<T>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T {
        self.data.as_mut()
    }
}
