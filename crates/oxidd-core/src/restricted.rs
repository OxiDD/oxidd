//! Restricted views on a manager, enabling the implementation of
//! [`LevelSwapper`] without unsafe code

use std::{fmt, marker::PhantomData, mem::ManuallyDrop};

use derive_where::derive_where;

use crate::util::{AbortOnDrop, AllocResult, Own, Ref};
use crate::{HasLevel, InnerNode, LevelNo, LevelSwapper, LevelView, Manager, Node};

/// Rules for fixing upwards-pointing edges as part of a level swap in a
/// decision diagram
///
/// All `SafeLevelSwapper` implementations have a blanket [`LevelSwapper`]
/// implementation.
pub trait SafeLevelSwapper<M: Manager> {
    /// Fixup procedure applied to every node at the lower level after a level
    /// swap
    ///
    /// `main` is the node under consideration. It was previously at the upper
    /// level. The node may have outgoing edges pointing to the upper level,
    /// which is forbidden in most kinds of decision diagrams. If so, the fixup
    /// procedure typically resolves the issue by keeping the node to the upper
    /// level and adjusting its children, preserving the node's original
    /// semantics. If the node does not have outgoing edges pointing to the
    /// upper level, it should be moved to the lower level.
    fn swap_fixup<'a, 'v>(
        manager: RestrictedManager<'a, M>,
        main: MutNode<'a, M>,
        upper: &mut UpperLevelView<'v, M>,
        lower: &mut LowerLevelView<'a, 'v, M>,
    ) -> AllocResult<Inserted>;
}

unsafe impl<T: SafeLevelSwapper<M>, M: Manager> LevelSwapper<M> for T {
    unsafe fn swap(
        manager: &M,
        upper_no: LevelNo,
        lower_no: LevelNo,
        upper_no_pre: LevelNo,
        lower_no_pre: LevelNo,
    ) {
        debug_assert!(upper_no < lower_no);
        debug_assert!(lower_no < manager.num_levels());
        debug_assert!(upper_no_pre < manager.num_levels());
        debug_assert!(lower_no_pre < manager.num_levels());

        // Note that the `Manager::level_unchecked()` may or may not acquire a
        // lock.
        let mut upper = unsafe { manager.level_unchecked(upper_no) };
        let mut lower = unsafe { manager.level_unchecked(lower_no) };

        let abort_on_panic = AbortOnDrop("Some operation in `oxidd_reorder::level_down` panicked.");
        unsafe { upper.swap(&mut lower) };

        // `lower` now refers to the new lower level, `upper` to the new upper
        // level. Note that nodes on the new lower level may refer to the new
        // upper level, which is not allowed. We also fix this below.
        #[cfg(debug_assertions)]
        upper.iter().for_each(|e| {
            manager
                .get_node(e)
                .unwrap_inner()
                .assert_level_matches(lower_no_pre)
        });

        let old_upper = LevelView::take(&mut lower);
        // SAFETY: reordering is in progress
        let old_upper = unsafe { old_upper.unwrap_unchecked() };
        lower.reserve(old_upper.len());

        let mut upper = UpperLevelView {
            view: upper,
            pre_level: lower_no_pre,
            phantom: PhantomData,
        };
        let mut lower = LowerLevelView {
            view: lower,
            old_upper: &old_upper,
            pre_level: upper_no_pre,
            mut_node_id: usize::MAX,
            phantom: PhantomData,
        };

        for e in old_upper.iter() {
            let Node::Inner(node) = manager.get_node(e) else {
                // SAFETY: all edges in level views refer to inner nodes
                unsafe { std::hint::unreachable_unchecked() };
            };
            #[cfg(debug_assertions)]
            node.assert_level_matches(upper_no_pre);

            lower.mut_node_id = e.node_id();

            Self::swap_fixup(
                RestrictedManager(manager),
                MutNode {
                    edge: e,
                    node,
                    no_drop: MustInsert,
                },
                &mut upper,
                &mut lower,
            )
            .unwrap(); // TODO: rollback if out of memory
        }

        abort_on_panic.defuse();
    }
}

/// A resource at level `L` during a level swap
#[repr(transparent)]
#[derive_where(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash; T)]
pub struct At<L, T>(#[derive_where(skip)] PhantomData<L>, T);

impl<L, T> At<L, T> {
    /// Unwrap the contained resource
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.1
    }
}
impl<L, T> std::ops::Deref for At<L, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.1
    }
}
impl<L, T> std::ops::DerefMut for At<L, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.1
    }
}

impl<L, T: fmt::Debug> fmt::Debug for At<L, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "At{}({:?})", std::any::type_name::<L>(), self.1)
    }
}

/// Marker type for the lower level during a level swap
pub struct Lower;
/// Marker type for the upper level during a level swap
pub struct Upper;

/// A resource at the lower level during a level swap
pub type AtLower<T> = At<Lower, T>;
/// A resource at the upper level during a level swap
pub type AtUpper<T> = At<Upper, T>;

/// Restricted reference to a [`Manager`]
///
/// By invariant, the manager is prepared for reordering. In context of
/// concurrent level swaps, only nodes at the two levels subject to the
/// respective swap may be accessed. To enforce this, a `RestrictedManager` on
/// its own does not provide a [`get_node()`][Manager::get_node] method.
/// Instead, there are [`UpperLevelView::node()`], [`LowerLevelView::node()`],
/// and [`RestrictedManager::get_terminal_node()`].
#[derive_where(Clone, Copy)]
pub struct RestrictedManager<'a, M>(&'a M);

impl<'a, M: Manager> RestrictedManager<'a, M> {
    /// Clone `edge`
    #[inline(always)]
    pub fn clone_edge(self, edge: Ref<'_, M::Edge>) -> Own<M::Edge> {
        self.0.clone_edge(edge)
    }
    /// Clone `edge`, retaining the [`At`] marker
    #[inline(always)]
    pub fn clone_edge_at<L>(self, edge: At<L, Ref<'_, M::Edge>>) -> At<L, Own<M::Edge>> {
        At(PhantomData, self.0.clone_edge(edge.1))
    }

    /// Drop `edge`
    #[inline(always)]
    pub fn drop_edge(self, edge: Own<M::Edge>) {
        self.0.drop_edge(edge);
    }

    /// Get a reference to the terminal node referenced by `edge`, or `None` if
    /// `edge` refers to an inner node
    #[inline]
    pub fn get_terminal_node(self, edge: Ref<'a, M::Edge>) -> Option<M::TerminalRef<'a>> {
        match self.0.get_node(edge) {
            Node::Inner(_) => None,
            Node::Terminal(t) => Some(t),
        }
    }

    /// Check if `edge` refers to `terminal`
    #[inline]
    pub fn refers_to_terminal(self, edge: Ref<'_, M::Edge>, terminal: &M::Terminal) -> bool {
        self.0.get_node(edge).is_terminal(terminal)
    }

    /// Get an edge for the given terminal
    ///
    /// See [`Manager::get_terminal()`] for more details.
    #[inline(always)]
    pub fn get_terminal(self, terminal: M::Terminal) -> AllocResult<Own<M::Edge>> {
        self.0.get_terminal(terminal)
    }
}

/// View on the lower level during a level swap
///
/// This is a wrapper around the lower [`LevelView`] and also contains a
/// reference to the old upper [`LevelView`].
///
/// There are two lifetimes involved here: `'v` is the lifetime of the level
/// views. Since the level views are generic associated types of `M`,
/// `LowerLevelView` is invariant in this lifetime. Given that the taken-out old
/// upper level view does not live as long, we need a separate lifetime `'a` for
/// the reference to it.
pub struct LowerLevelView<'a, 'v, M: Manager + 'v> {
    view: M::LevelView<'v>,
    old_upper: &'a <M::LevelView<'v> as LevelView<M::Edge, M::InnerNode>>::Taken,
    /// Level prior to reordering
    pre_level: LevelNo,
    mut_node_id: usize,
    phantom: PhantomData<M>,
}

impl<'a, 'v, M: Manager> LowerLevelView<'a, 'v, M> {
    /// Insert the given edge into the unique table at the lower level, assuming
    /// that the referenced node is already stored in the associated manager.
    ///
    /// Returns `true` if the edge was inserted, `false` if it was already
    /// present.
    ///
    /// Safe zero-cost wrapper around [`LevelView::insert_unchecked()`]. See the
    /// documentation there for details on the panicking behavior.
    #[inline(always)]
    pub fn insert(&mut self, edge: AtLower<Own<M::Edge>>) -> bool {
        // SAFETY: `edge` points to a node at the lower level by `AtLower`'s
        // type invariant
        unsafe { self.view.insert_unchecked(edge.1) }
    }

    /// Get a reference to the node to which `edge` points
    ///
    /// Returns `None` if `edge` refers to the main node of
    /// [`SafeLevelSwapper::swap_fixup()`]. This is to allow safe modification
    /// of the main node.
    ///
    /// This is a wrapper around [`Manager::get_node()`]. You may want to use
    /// [`Self::node_checked()`] in case the level referred to by `edge`
    /// is not statically known but the node type implements [`HasLevel`].
    #[inline]
    pub fn node(
        &self,
        manager: RestrictedManager<'a, M>,
        edge: AtLower<Ref<'a, M::Edge>>,
    ) -> Option<&'a M::InnerNode> {
        if edge.node_id() == self.mut_node_id {
            return None;
        }
        match manager.0.get_node(edge.1) {
            Node::Inner(n) => Some(n),
            // SAFETY: by invariant, edge points to an inner node
            Node::Terminal(_) => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Get an edge pointing to a node at this level with `children` if present,
    /// or insert such a node.
    ///
    /// This method checks both the old upper level and the (current) lower
    /// level for present nodes.
    #[inline]
    pub fn get_or_insert(
        &mut self,
        manager: RestrictedManager<'a, M>,
        children: &[Ref<'_, M::Edge>],
    ) -> AllocResult<Own<M::Edge>> {
        if let Some(edge) = self.old_upper.get(children) {
            Ok(manager.clone_edge(edge))
        } else {
            let node = M::InnerNode::new(
                self.pre_level,
                children.iter().map(|e| manager.clone_edge(*e)),
            );
            unsafe { self.view.get_or_insert_unchecked(node) }
        }
    }

    /// Try to remove the node referred to by `edge` from (this level of) the
    /// manager
    ///
    /// Returns whether the node was present at this level and has been removed.
    #[inline(always)]
    pub fn try_remove(&mut self, edge: Own<M::Edge>) -> bool {
        self.view.try_remove(edge)
    }

    /// Get the level number of this level, i.e., the distance to the top-most
    /// level
    #[inline(always)]
    pub fn level_no(&self) -> LevelNo {
        self.view.level_no()
    }

    /// Get the level number prior to reordering. If the node type implements
    /// [`HasLevel`], then all nodes at this level are labeled with this level
    /// number.
    #[inline(always)]
    pub fn previous_level_no(&self) -> LevelNo {
        self.pre_level
    }
}
impl<'a, 'v, M: Manager> LowerLevelView<'a, 'v, M>
where
    M::InnerNode: HasLevel,
{
    /// Get a reference to the node to which `edge` points
    ///
    /// Returns `None` if `edge` refers to a node at a different level (i.e.,
    /// neither the lower nor the old upper level) or to the main node of
    /// [`SafeLevelSwapper::swap_fixup()`]. The latter is to allow safe
    /// modification of the main node.
    ///
    /// Like [`Self::node()`], this is a wrapper around [`Manager::get_node()`].
    /// Use [`Self::node()`] instead if you have the [`AtLower`] predicate for
    /// `edge`.
    #[inline]
    pub fn node_checked(
        &self,
        manager: RestrictedManager<'a, M>,
        edge: Ref<'a, M::Edge>,
    ) -> Option<&'a M::InnerNode> {
        if edge.node_id() == self.mut_node_id {
            return None;
        }
        match manager.0.get_node(edge) {
            Node::Inner(n) if n.level() == self.pre_level => Some(n),
            _ => None,
        }
    }
}

/// View on the upper level during a level swap
///
/// This is a wrapper around the upper [`LevelView`].
pub struct UpperLevelView<'v, M: Manager + 'v> {
    view: M::LevelView<'v>,
    /// Level prior to reordering
    pre_level: LevelNo,
    phantom: PhantomData<M>,
}

impl<'v, M: Manager> UpperLevelView<'v, M> {
    /// Insert the given edge into the unique table at the upper level, assuming
    /// that the referenced node is already stored in the associated manager.
    ///
    /// Returns `true` if the edge was inserted, `false` if it was already
    /// present.
    ///
    /// Safe zero-cost wrapper around [`LevelView::insert_unchecked()`]. See the
    /// documentation there for details on the panicking behavior.
    #[inline(always)]
    pub fn insert(&mut self, edge: AtUpper<Own<M::Edge>>) -> bool {
        // SAFETY: `edge` points to a node at the upper level by `AtUpper`'s
        // type invariant
        unsafe { self.view.insert_unchecked(edge.1) }
    }

    /// Get a reference to the node to which `edge` points
    ///
    /// This is a zero-cost wrapper around [`Manager::get_node()`]. You may want
    /// to use [`Self::node_checked()`] in case the level referred to by `edge`
    /// is not statically known but the node type implements [`HasLevel`].
    #[inline]
    pub fn node(
        &self,
        manager: RestrictedManager<'v, M>,
        edge: AtUpper<Ref<'v, M::Edge>>,
    ) -> &'v M::InnerNode {
        match manager.0.get_node(edge.1) {
            Node::Inner(n) => n,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    /// Get an edge pointing to a node at this level with `children` if present,
    /// or insert such a node.
    ///
    /// Safe wrapper around [`InnerNode::new()`] and
    /// [`LevelView::get_or_insert_unchecked()`].
    #[inline]
    pub fn get_or_insert(
        &mut self,
        children: impl IntoIterator<Item = Own<M::Edge>>,
    ) -> AllocResult<Own<M::Edge>> {
        let node = M::InnerNode::new(self.pre_level, children);
        // SAFETY: `node` has the desired level
        unsafe { self.view.get_or_insert_unchecked(node) }
    }

    /// Try to remove the node referred to by `edge` from (this level of) the
    /// manager
    ///
    /// Returns whether the node was present at this level and has been removed.
    #[inline(always)]
    pub fn try_remove(&mut self, edge: Own<M::Edge>) -> bool {
        self.view.try_remove(edge)
    }

    /// Get the level number of this level, i.e., the distance to the top-most
    /// level
    #[inline(always)]
    pub fn level_no(&self) -> LevelNo {
        self.view.level_no()
    }

    /// Get the level number prior to reordering. If the node type implements
    /// [`HasLevel`], then all nodes at this level are labeled with this level
    /// number.
    #[inline(always)]
    pub fn previous_level_no(&self) -> LevelNo {
        self.pre_level
    }
}
impl<'v, M: Manager> UpperLevelView<'v, M>
where
    M::InnerNode: HasLevel,
{
    /// Get a reference to the node to which `edge` points
    ///
    /// Returns `None` if `edge` refers to a node at a different level.
    ///
    /// Like [`Self::node()`], this is a wrapper around [`Manager::get_node()`].
    /// Use [`Self::node()`] instead if you have the [`AtUpper`] predicate for
    /// `edge`.
    #[inline]
    pub fn node_checked<'a>(
        &self,
        manager: RestrictedManager<'a, M>,
        edge: Ref<'a, M::Edge>,
    ) -> Option<&'a M::InnerNode> {
        match manager.0.get_node(edge) {
            Node::Inner(n) if n.level() == self.pre_level => Some(n),
            _ => None,
        }
    }
}

/// Drop guard for owned edges to ensure that they are not leaked
///
/// Unlike [`util::EdgeDropGuard`][crate::util::EdgeDropGuard], this
/// implementation works with a [`RestrictedManager`].
pub struct EdgeDropGuard<'a, M: Manager> {
    manager: &'a M,
    edge: ManuallyDrop<Own<M::Edge>>,
}

impl<'a, M: Manager> std::ops::Deref for EdgeDropGuard<'a, M> {
    type Target = Own<M::Edge>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.edge
    }
}
impl<'a, M: Manager> std::ops::DerefMut for EdgeDropGuard<'a, M> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.edge
    }
}

impl<'a, M: Manager> EdgeDropGuard<'a, M> {
    /// Create a new drop guard
    #[inline]
    pub fn new(manager: RestrictedManager<'a, M>, edge: Own<M::Edge>) -> Self {
        Self {
            manager: manager.0,
            edge: ManuallyDrop::new(edge),
        }
    }

    /// Convert `this` into the contained edge
    #[inline]
    pub fn into_edge(mut self) -> Own<M::Edge> {
        // SAFETY: `this.edge` is never used again, we drop `self` below
        let edge = unsafe { ManuallyDrop::take(&mut self.edge) };
        std::mem::forget(self);
        edge
    }
}

impl<'a, M: Manager> Drop for EdgeDropGuard<'a, M> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self.edge` is never used again.
        self.manager
            .drop_edge(unsafe { ManuallyDrop::take(&mut self.edge) });
    }
}

struct MustInsert;

impl Drop for MustInsert {
    #[track_caller]
    fn drop(&mut self) {
        panic!("The `MutNode` must be inserted back into the unique table and not be dropped");
    }
}

/// Marker type to ensure that the main node of
/// [`SafeLevelSwapper::swap_fixup()`] gets inserted at either of the levels
pub struct Inserted(());

/// Mutable node reference, allowing safe modification of its children
pub struct MutNode<'a, M: Manager> {
    // Could be `pub`, but must not be marked as `AtLower<_>`: While `edge`
    // initially refers to a node at old upper / new lower level, this is not
    // true anymore after calling `MutNode::insert_at_upper_level()`.
    edge: Ref<'a, M::Edge>,
    // Must not be `pub`: when calling `set_child`, we obtain the old child and
    // can remove the node. Had we exposed a node reference with lifetime `'a`,
    // then one would be able to obtain a `Ref<'a, M::Edge>` to the child and
    // have a use after free access to the child node. By having a dedicated
    // `node` method, we make sure that references to child nodes obtained from
    // this node only live until `set_child()` is called.
    node: &'a M::InnerNode,
    no_drop: MustInsert,
}

impl<'a, M: Manager> MutNode<'a, M> {
    /// Get a plain inner node reference
    // Soundness: The lifetime of the returned reference must be bounded by the
    // lifetime of the `self` reference, see the comment on the `node` field.
    #[inline(always)]
    pub fn node<'b>(&'b self) -> &'b M::InnerNode {
        self.node
    }

    /// Set the `n`-th child of this node
    ///
    /// Returns the previous `n`-th child. Panics if the node does not have an
    /// `n`-th child.
    ///
    /// Safe zero-cost wrapper around [`InnerNode::set_child()`]. See the
    /// documentation there for more details.
    // Soundness: `set_child` must take a mutable reference, see the comment on
    // the `node` field.
    #[inline(always)]
    pub fn set_child(&mut self, n: usize, child: Own<M::Edge>) -> Own<M::Edge> {
        unsafe { self.node.set_child(n, child) }
    }

    /// Insert this node at the lower level
    ///
    /// The node must not be present at the lower level yet—this is checked via
    /// a debug assertion. Note that in the [`LevelSwapper`] implementation for
    /// a [`SafeLevelSwapper`], the lower level is initially empty, i.e., does
    /// not contain the nodes of the old upper level, and is successively filled
    /// by iterating over the old upper level and calling
    /// [`SafeLevelSwapper::swap_fixup()`].
    #[inline]
    pub fn insert_at_lower_level(
        self,
        manager: RestrictedManager<'_, M>,
        level: &mut LowerLevelView<'_, '_, M>,
    ) -> Inserted {
        std::mem::forget(self.no_drop);
        let edge = manager.0.clone_edge(self.edge);
        let _inserted = unsafe { level.view.insert_unchecked(edge) };
        debug_assert!(_inserted);
        Inserted(())
    }
}

impl<'a, M: Manager> MutNode<'a, M>
where
    M::InnerNode: HasLevel,
{
    /// Insert this node at the upper level, adjusting the node's level label
    ///
    /// The node must not be present at the upper level yet—this is checked via
    /// a debug assertion.
    #[inline]
    pub fn insert_at_upper_level(
        self,
        manager: RestrictedManager<'_, M>,
        level: &mut UpperLevelView<'_, M>,
    ) -> Inserted {
        std::mem::forget(self.no_drop);
        unsafe { self.node.set_level(level.pre_level) };
        let edge = manager.0.clone_edge(self.edge);
        let _inserted = unsafe { level.view.insert_unchecked(edge) };
        debug_assert!(_inserted);
        Inserted(())
    }
}
