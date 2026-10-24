//! List decision diagrams (LDDs) for OxiDD.

use std::hash::Hash;

use oxidd_core::{
    Edge, HasApplyCache, InnerNode, Manager,
    util::{AllocResult, Own, Ref},
};
use oxidd_derive::Countable;

mod apply_rec;
mod recursor;

#[cfg(feature = "statistics")]
pub use apply_rec::print_stats;

/// Terminal nodes of a list decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
pub enum LDDTerminal {
    /// This represents the empty set, also denoted by `false`.
    Empty,
    /// This represents the set containing only the empty list, also denoted by
    /// `true`.
    True,
}

/// For LDDs it is essential that values are ordered and can be cloned.
trait LDDValue: Clone + Ord + Eq + Hash {
    /// The value used to indicate "included" in a projection meta-LDD.
    fn true_value() -> Self;

    /// The value used to indicate "excluded" in a projection meta-LDD, and
    /// "skip" (neither read nor write) in a relation-product meta-LDD.
    fn false_value() -> Self;

    /// Value encoding a read-only position in a relation-product meta-LDD.
    fn read_only_value() -> Self;

    /// Value encoding a write-only position in a relation-product meta-LDD.
    fn write_only_value() -> Self;

    /// Value encoding the *read* half of a read+write position pair in a
    /// relation-product meta-LDD.
    fn read_of_pair_value() -> Self;

    /// Value encoding the *write* half of a read+write position pair in a
    /// relation-product meta-LDD.
    fn write_of_pair_value() -> Self;
}

// The default LDDValue to be used.
impl LDDValue for u32 {
    #[inline(always)]
    fn true_value() -> Self {
        1
    }

    #[inline(always)]
    fn false_value() -> Self {
        0
    }

    #[inline(always)]
    fn read_only_value() -> Self {
        1
    }

    #[inline(always)]
    fn write_only_value() -> Self {
        2
    }

    #[inline(always)]
    fn read_of_pair_value() -> Self {
        3
    }

    #[inline(always)]
    fn write_of_pair_value() -> Self {
        4
    }
}

/// Index of the down branch in a node
#[allow(unused)] // for later usage
const DOWN: usize = 0;
/// Index of the right branch in a node
const RIGHT: usize = 1;

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Ref<'_, E>, Ref<'_, E>) {
    const { assert!(N::ARITY == 2) };
    let mut it = node.children();
    let f_down = it.next().unwrap();
    let f_right = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (f_down, f_right)
}

/// Rules for list decision diagrams
pub struct LDDRules;

/// Create a node in `manager` if necessary
#[inline(always)]
fn make_node<M: LDDManager>(
    manager: &M,
    value: &<M as Manager>::InnerNodeValue,
    down: Own<M::Edge>,
    right: Own<M::Edge>,
) -> AllocResult<Own<M::Edge>> {
    manager.get_or_insert(0, [down, right], value.clone())
}

/// Native operators of this LDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Ord, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum LDDOp {
    Union,
    Intsec,
    Diff,

    Project,

    RelationalProduct,
    RelationalPredecessor,
}

trait LDDManager:
    Manager<Terminal = LDDTerminal, InnerNodeValue: LDDValue> + HasApplyCache<Self, LDDOp>
{
}
impl<M> LDDManager for M
where
    M: Manager<Terminal = LDDTerminal> + HasApplyCache<M, LDDOp>,
    M::InnerNodeValue: LDDValue,
{
}

/// Result of [`LDDFunction::relation_product_meta`].
///
/// Bundles the meta-LDD encoding the read/write projection together with the
/// positions of the read and write variables in that encoding.
pub struct RelationProductMeta<E> {
    /// The meta-LDD encoding the read/write projection.
    pub meta: E,
    /// The positions of the read variables in the meta encoding.
    pub read_positions: Vec<usize>,
    /// The positions of the write variables in the meta encoding.
    pub write_positions: Vec<usize>,
}

// --- Function Interface ------------------------------------------------------

pub use apply_rec::LDDFunction;
#[cfg(feature = "multi-threading")]
pub use apply_rec::mt::LDDFunctionMT;

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

    fn print<O: oxidd_core::Countable + std::fmt::Debug>(counters: &[Self]) {
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
            let op = <O as oxidd_core::Countable>::from_usize(i);
            eprintln!(
                "  {op:?}: calls: {calls}, cache queries: {cache_queries} ({terminal_percent} % terminal cases), cache hits: {cache_hits} ({cache_hit_percent} %), reduced: {reduced}"
            );
        }
    }
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [StatCounters; <LDDOp as oxidd_core::Countable>::MAX_VALUE + 1] =
    [StatCounters::INIT; <LDDOp as oxidd_core::Countable>::MAX_VALUE + 1];

/// Print statistics to stderr
#[cfg(feature = "statistics")]
pub fn print_stats() {
    eprintln!("[oxidd_rules_ldd]");
    StatCounters::print::<LDDOp>(&STAT_COUNTERS);
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
