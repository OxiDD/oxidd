//! Rules and other basic definitions for zero-suppressed binary decision
//! diagrams
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![forbid(unsafe_code)]
// `'id` lifetimes may make the code easier to understand
#![allow(clippy::needless_lifetimes)]

use std::fmt;
use std::hash::Hash;

use oxidd_core::util::{AllocResult, Borrowed, DropWith};
use oxidd_core::{
    DiagramRules, Edge, HasLevel, InnerNode, LevelNo, LevelView, Manager, ManagerEventSubscriber,
    ReducedOrNew,
};
use oxidd_derive::Countable;

// spell-checker:ignore symm

mod apply_rec;
mod recursor;

// --- Reduction Rules ---------------------------------------------------------

/// [`DiagramRules`] for simple binary decision diagrams
pub struct ZBDDRules;

impl<E: Edge, N: InnerNode<E>> DiagramRules<E, N, ZBDDTerminal> for ZBDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline]
    fn reduce<M: Manager<Edge = E, InnerNode = N, Terminal = ZBDDTerminal>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let hi = it.next().unwrap();
        let lo = it.next().unwrap();
        debug_assert!(it.next().is_none());

        if manager.get_node(&hi).is_terminal(&ZBDDTerminal::Empty) {
            manager.drop_edge(hi);
            return ReducedOrNew::Reduced(lo);
        }
        ReducedOrNew::New(N::new(level, [hi, lo]), Default::default())
    }

    #[inline]
    fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        node.children()
    }
}

#[inline(always)]
fn reduce<M>(
    manager: &M,
    level: LevelNo,
    hi: M::Edge,
    lo: M::Edge,
    op: ZBDDOp,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal>,
{
    // We do not use `DiagramRules::reduce()` here, as the iterator is
    // apparently not fully optimized away.
    if manager.get_node(&hi).is_terminal(&ZBDDTerminal::Empty) {
        stat!(reduced op);
        manager.drop_edge(hi);
        return Ok(lo);
    }
    oxidd_core::LevelView::get_or_insert(
        &mut manager.level(level),
        M::InnerNode::new(level, [hi, lo]),
    )
}

#[inline(always)]
fn reduce_borrowed<M>(
    manager: &M,
    level: LevelNo,
    hi: Borrowed<M::Edge>,
    lo: M::Edge,
    op: ZBDDOp,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal>,
{
    let _ = op;
    if manager.get_node(&hi).is_terminal(&ZBDDTerminal::Empty) {
        stat!(reduced op);
        return Ok(lo);
    }
    ReducedOrNew::New(
        M::InnerNode::new(level, [manager.clone_edge(&hi), lo]),
        Default::default(),
    )
    .then_insert(manager, level)
}

// --- Terminal Type -----------------------------------------------------------

/// Terminal nodes in simple binary decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
pub enum ZBDDTerminal {
    /// Represents the empty set (∅)
    Empty,
    /// Represents the set containing exactly the empty set ({∅})
    Base,
}

impl<Tag: Default> oxidd_dump::ParseTagged<Tag> for ZBDDTerminal {
    fn parse(s: &str) -> Option<(Self, Tag)> {
        let val = match s {
            "e" | "E" | "empty" | "Empty" | "EMPTY" | "∅" | "0" => ZBDDTerminal::Empty,
            "b" | "B" | "base" | "Base" | "BASE" | "{∅}" => ZBDDTerminal::Base,
            _ => return None,
        };
        Some((val, Tag::default()))
    }
}

impl oxidd_dump::AsciiDisplay for ZBDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            ZBDDTerminal::Empty => f.write_str("E"),
            ZBDDTerminal::Base => f.write_str("B"),
        }
    }
}

impl fmt::Display for ZBDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZBDDTerminal::Empty => f.write_str("∅"),
            ZBDDTerminal::Base => f.write_str("{∅}"),
        }
    }
}

// --- ZBDD Cache --------------------------------------------------------------

pub struct ZBDDCache<E> {
    tautologies: Vec<E>,
}

impl<M> ManagerEventSubscriber<M> for ZBDDCache<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasZBDDCache<M::Edge>,
{
    #[inline(always)]
    fn init_mut(manager: &mut M) {
        Self::post_reorder_mut(manager);
    }

    fn pre_reorder_mut(manager: &mut M) {
        // clear the cache top down
        let mut ts = std::mem::take(&mut manager.zbdd_cache_mut().tautologies);
        let mut level = 0;
        while ts.len() > 1 {
            if !manager.try_remove_node(ts.pop().unwrap(), level) {
                break;
            }
            level += 1;
        }
        for e in ts.into_iter().rev() {
            manager.drop_edge(e);
        }
    }

    fn post_reorder_mut(manager: &mut M) {
        // Build the tautologies bottom up
        //
        // Storing the edge for `ZBDDTerminal::Base` as well enables us to return
        // `&E` instead of `E` in `Self::tautology()`, so we don't need as many
        // clone/drop operations.
        let mut tautologies = Vec::with_capacity(1 + manager.num_levels() as usize);
        tautologies.push(manager.get_terminal(ZBDDTerminal::Base).unwrap());
        for mut view in manager.levels().rev() {
            let level = view.level_no();
            let hi = manager.clone_edge(tautologies.last().unwrap());
            let lo = manager.clone_edge(&hi);
            let Ok(edge) = view.get_or_insert(M::InnerNode::new(level, [hi, lo])) else {
                eprintln!("Out of memory");
                std::process::abort();
            };
            tautologies.push(edge);
        }

        manager.zbdd_cache_mut().tautologies = tautologies;
    }
}

pub trait HasZBDDCache<E: Edge> {
    fn zbdd_cache(&self) -> &ZBDDCache<E>;
    fn zbdd_cache_mut(&mut self) -> &mut ZBDDCache<E>;
}
impl<E: Edge, T: AsRef<ZBDDCache<E>> + AsMut<ZBDDCache<E>>> HasZBDDCache<E> for T {
    #[inline(always)]
    fn zbdd_cache(&self) -> &ZBDDCache<E> {
        self.as_ref()
    }
    #[inline(always)]
    fn zbdd_cache_mut(&mut self) -> &mut ZBDDCache<E> {
        self.as_mut()
    }
}

impl<E: Edge> DropWith<E> for ZBDDCache<E> {
    fn drop_with(self, drop_edge: impl Fn(E)) {
        for e in self.tautologies.into_iter().rev() {
            drop_edge(e)
        }
    }
}

impl<E: Edge> ZBDDCache<E> {
    /// Create a new `ZBDDCache`
    pub fn new() -> Self {
        Self {
            tautologies: Vec::new(),
        }
    }

    /// Get the tautology for the set of variables at `level` and below
    #[inline]
    fn tautology(&self, level: LevelNo) -> &E {
        // The vector contains one entry for each level including the terminals.
        // The terminal level comes first, the top-most level last.
        let len = self.tautologies.len() as u32;
        debug_assert!(
            len > 0,
            "ZBDDCache is empty. This is an OxiDD-internal error."
        );
        let rev_idx = std::cmp::min(len - 1, level);
        &self.tautologies[(len - 1 - rev_idx) as usize]
    }
}

impl<E: Edge> Default for ZBDDCache<E> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Operations & Apply Implementation ---------------------------------------

/// Native operations of this ZBDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum ZBDDOp {
    Subset0,
    Subset1,
    Change,
    Union,
    Intsec,
    Diff,
    SymmDiff,

    /// If-then-else
    Ite,

    /// Make a new node
    ///
    /// Only used by [`make_node()`] for statistical purposes.
    MkNode,
}

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<'_, E>, Borrowed<'_, E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = node.children();
    let hi = it.next().unwrap();
    let lo = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (hi, lo)
}

/// Get the level number of the singleton set referenced by `edge`
///
/// Panics if `edge` does not point to a singleton set.
#[inline]
#[track_caller]
fn singleton_level<M: Manager<Terminal = ZBDDTerminal>>(manager: &M, edge: &M::Edge) -> LevelNo
where
    M::InnerNode: HasLevel,
{
    let node = manager
        .get_node(edge)
        .expect_inner("expected a singleton set, got a terminal");
    debug_assert!(
        {
            let (hi, lo) = collect_children(node);
            manager.get_node(&*hi).is_terminal(&ZBDDTerminal::Base)
                && manager.get_node(&*lo).is_terminal(&ZBDDTerminal::Empty)
        },
        "expected a singleton set, but the children are not the respective terminals"
    );
    node.level()
}

/// Create a set that corresponds to a ZBDD node at the level of `var` with
/// the given `hi` and `lo` edges
///
/// `var` must be a singleton set, and `var`'s level must be above `hi`'s
/// and `lo`'s levels. If one of these conditions is violated, the result is
/// unspecified. Ideally the method panics.
///
/// The set semantics of this new node is `lo ∪ {x ∪ {var} | x ∈ hi}`, the
/// logical equivalent is `lo ∨ (var ∧ hi|ᵥₐᵣ₌₀)`.
pub fn make_node<M>(manager: &M, var: &M::Edge, hi: M::Edge, lo: M::Edge) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal>,
    M::InnerNode: HasLevel,
{
    let level = singleton_level(manager, var);
    reduce(manager, level, hi, lo, ZBDDOp::MkNode)
}

/// Get the Boolean function v for the singleton set {v} (given by `singleton`)
///
/// Panics if `singleton` is not a singleton set
#[deprecated = "use `BooleanFunction::var` instead"]
pub fn var_boolean_function<M>(manager: &M, singleton: &M::Edge) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasZBDDCache<M::Edge>,
    M::InnerNode: HasLevel,
{
    let level = singleton_level(manager, singleton);
    let hi = manager.clone_edge(manager.zbdd_cache().tautology(level + 1));
    let lo = manager.get_terminal(ZBDDTerminal::Empty).unwrap();
    let mut edge = oxidd_core::LevelView::get_or_insert(
        &mut manager.level(level),
        InnerNode::new(level, [hi, lo]),
    )?;

    // Build the chain bottom up. We need to skip the newly created level.
    let levels = manager.levels().rev();
    // skip -> for level 0, we are already done
    for mut view in levels.skip((manager.num_levels() - level) as usize) {
        // only use `oxidd_core::LevelView` here to mitigate confusion of Rust Analyzer
        use oxidd_core::LevelView;

        let level = view.level_no();
        let edge2 = manager.clone_edge(&edge);
        edge = view.get_or_insert(InnerNode::new(level, [edge, edge2]))?;
    }

    Ok(edge)
}

// --- Function Interface ------------------------------------------------------

#[cfg(feature = "multi-threading")]
pub use apply_rec::mt::ZBDDFunctionMT;
pub use apply_rec::ZBDDFunction;

// --- Statistics --------------------------------------------------------------

#[cfg(feature = "statistics")]
struct StatCounters {
    calls: std::sync::atomic::AtomicI64,
    cache_queries: std::sync::atomic::AtomicI64,
    cache_hits: std::sync::atomic::AtomicI64,
    reduced: std::sync::atomic::AtomicI64,
}

#[cfg(feature = "statistics")]
impl StatCounters {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: StatCounters = StatCounters {
        calls: std::sync::atomic::AtomicI64::new(0),
        cache_queries: std::sync::atomic::AtomicI64::new(0),
        cache_hits: std::sync::atomic::AtomicI64::new(0),
        reduced: std::sync::atomic::AtomicI64::new(0),
    };

    fn print(counters: &[Self]) {
        // spell-checker:ignore ctrs
        for (i, ctrs) in counters.iter().enumerate() {
            let calls = ctrs.calls.swap(0, std::sync::atomic::Ordering::Relaxed);
            let cache_queries = ctrs
                .cache_queries
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            let cache_hits = ctrs
                .cache_hits
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            let reduced = ctrs.reduced.swap(0, std::sync::atomic::Ordering::Relaxed);

            if calls == 0 {
                continue;
            }

            let terminal_percent = (calls - cache_queries) as f32 / calls as f32 * 100.0;
            let cache_hit_percent = cache_hits as f32 / cache_queries as f32 * 100.0;
            let op = <ZBDDOp as oxidd_core::Countable>::from_usize(i);
            eprintln!("  {op:?}: calls: {calls}, cache queries: {cache_queries} ({terminal_percent} % terminal cases), cache hits: {cache_hits} ({cache_hit_percent} %), reduced: {reduced}");
        }
    }
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [crate::StatCounters; <ZBDDOp as oxidd_core::Countable>::MAX_VALUE + 1] =
    [crate::StatCounters::INIT; <ZBDDOp as oxidd_core::Countable>::MAX_VALUE + 1];

#[cfg(feature = "statistics")]
/// Print statistics to stderr
pub fn print_stats() {
    eprintln!("[oxidd_rules_zbdd]");
    crate::StatCounters::print(&STAT_COUNTERS);
}

macro_rules! stat {
    (call $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .calls
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (cache_query $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .cache_queries
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (cache_hit $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .cache_hits
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
    (reduced $op:expr) => {
        let _ = $op as usize;
        #[cfg(feature = "statistics")]
        STAT_COUNTERS[$op as usize]
            .reduced
            .fetch_add(1, ::std::sync::atomic::Ordering::Relaxed);
    };
}

pub(crate) use stat;
