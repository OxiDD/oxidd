//! Rules and other basic definitions for binary decision diagrams
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
// `'id` lifetimes may make the code easier to understand
#![allow(clippy::needless_lifetimes)]

use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Manager;

#[cfg(feature = "complement-edge")]
pub mod complement_edge;
#[cfg(feature = "simple")]
pub mod simple;

/// Remove all variables from `set` which are above level `until`
///
/// `set` is the conjunction of the variables. This means that popping a
/// variable resorts to taking its “true” child.
#[inline]
fn set_pop<'a, M: Manager>(
    manager: &'a M,
    set: Borrowed<'a, M::Edge>,
    until: LevelNo,
) -> Borrowed<'a, M::Edge>
where
    M::InnerNode: HasLevel,
{
    match manager.get_node(&set) {
        oxidd_core::Node::Inner(n) => {
            if n.level() >= until {
                set
            } else {
                // It seems like we cannot simply implement this as a loop due
                // to lifetime restrictions. But the compiler should perform
                // tail call optimization.
                set_pop(manager, n.child(0), until)
            }
        }
        oxidd_core::Node::Terminal(_) => set,
    }
}

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
    const INIT: StatCounters = StatCounters {
        calls: std::sync::atomic::AtomicI64::new(0),
        cache_queries: std::sync::atomic::AtomicI64::new(0),
        cache_hits: std::sync::atomic::AtomicI64::new(0),
        reduced: std::sync::atomic::AtomicI64::new(0),
    };

    fn print(counters: &[Self], labels: &[&str]) {
        // spell-checker:ignore ctrs
        for (ctrs, op) in counters.iter().zip(labels) {
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
            eprintln!("  {op}: calls: {calls}, cache queries: {cache_queries} ({terminal_percent} % terminal cases), cache hits: {cache_hits} ({cache_hit_percent} %), reduced: {reduced}");
        }
    }
}

#[cfg(not(feature = "statistics"))]
macro_rules! stat {
    (call $op:expr) => {};
    (cache_query $op:expr) => {};
    (cache_hit $op:expr) => {};
    (reduced $op:expr) => {};
}

#[cfg(feature = "statistics")]
macro_rules! stat {
    (call $op:expr) => {
        STAT_COUNTERS[$op as usize]
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
    (cache_query $op:expr) => {
        STAT_COUNTERS[$op as usize]
            .cache_queries
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
    (cache_hit $op:expr) => {
        STAT_COUNTERS[$op as usize]
            .cache_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
    (reduced $op:expr) => {
        STAT_COUNTERS[$op as usize]
            .reduced
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
}

use oxidd_core::util::Borrowed;
pub(crate) use stat;
