//! Rules and other basic definitions for multi-terminal binary decision
//! diagrams
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![forbid(unsafe_code)]
// `'id` lifetimes may make the code easier to understand
#![allow(clippy::needless_lifetimes)]

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::Hash;

use oxidd_core::function::NumberBase;
use oxidd_core::util::{AllocResult, Borrowed};
use oxidd_core::{DiagramRules, Edge, InnerNode, LevelNo, Manager, Node, ReducedOrNew};
use oxidd_derive::Countable;

pub mod terminal;

mod apply_rec;

// --- Reduction Rules ---------------------------------------------------------

/// [`DiagramRules`] for (multi-terminal) binary decision diagrams
pub struct MTBDDRules;

impl<E: Edge, N: InnerNode<E>, T> DiagramRules<E, N, T> for MTBDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline]
    #[must_use]
    fn reduce<M: Manager<Edge = E, InnerNode = N, Terminal = T>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let t = it.next().unwrap();
        let e = it.next().unwrap();
        debug_assert!(it.next().is_none());

        if t == e {
            manager.drop_edge(e);
            ReducedOrNew::Reduced(t)
        } else {
            ReducedOrNew::New(N::new(level, [t, e]), Default::default())
        }
    }

    #[inline]
    #[must_use]
    fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        node.children()
    }
}

#[inline(always)]
fn reduce<M: Manager>(
    manager: &M,
    level: LevelNo,
    t: M::Edge,
    e: M::Edge,
    op: MTBDDOp,
) -> AllocResult<M::Edge> {
    let _ = op;
    let tmp = <MTBDDRules as DiagramRules<_, _, _>>::reduce(manager, level, [t, e]);
    if let ReducedOrNew::Reduced(..) = &tmp {
        stat!(reduced op);
    }
    tmp.then_insert(manager, level)
}

// --- Operations & Apply Implementation ---------------------------------------

/// Native operations of this MTBDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum MTBDDOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,

    /// If-then-else
    Ite,
}

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<E>, Borrowed<E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = node.children();
    let t = it.next().unwrap();
    let e = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (t, e)
}

enum Operation<'a, E: 'a + Edge> {
    Binary(MTBDDOp, Borrowed<'a, E>, Borrowed<'a, E>),
    Done(E),
}

/// Terminal case for binary operators
#[inline]
fn terminal_bin<'a, M: Manager<Terminal = T>, T: NumberBase, const OP: u8>(
    m: &M,
    f: &'a M::Edge,
    g: &'a M::Edge,
) -> AllocResult<Operation<'a, M::Edge>> {
    use Node::*;
    use Operation::*;

    Ok(if OP == MTBDDOp::Add as u8 {
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(tf), Terminal(tg)) => {
                let val = tf.borrow().add(tg.borrow());
                Done(m.get_terminal(val)?)
            }
            (Terminal(t), _) if t.borrow().is_zero() => Done(m.clone_edge(g)),
            (_, Terminal(t)) if t.borrow().is_zero() => Done(m.clone_edge(f)),
            (Terminal(t), _) | (_, Terminal(t)) if t.borrow().is_nan() => {
                Done(m.get_terminal(T::nan())?)
            }
            _ if f > g => Binary(MTBDDOp::Add, g.borrowed(), f.borrowed()),
            _ => Binary(MTBDDOp::Add, f.borrowed(), g.borrowed()),
        }
    } else if OP == MTBDDOp::Sub as u8 {
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(tf), Terminal(tg)) => {
                let val = tf.borrow().sub(tg.borrow());
                Done(m.get_terminal(val)?)
            }
            (Terminal(t), _) if t.borrow().is_zero() => Done(m.clone_edge(g)),
            (_, Terminal(t)) if t.borrow().is_zero() => Done(m.clone_edge(f)),
            (Terminal(t), _) | (_, Terminal(t)) if t.borrow().is_nan() => {
                Done(m.get_terminal(T::nan())?)
            }
            _ => Binary(MTBDDOp::Sub, f.borrowed(), g.borrowed()),
        }
    } else if OP == MTBDDOp::Mul as u8 {
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(tf), Terminal(tg)) => {
                let val = tf.borrow().mul(tg.borrow());
                Done(m.get_terminal(val)?)
            }
            (Terminal(t), _) if t.borrow().is_one() => Done(m.clone_edge(g)),
            (_, Terminal(t)) if t.borrow().is_one() => Done(m.clone_edge(f)),
            (Terminal(t), _) | (_, Terminal(t)) if t.borrow().is_nan() => {
                Done(m.get_terminal(T::nan())?)
            }
            // Don't optimize the case where one of the operands is 0. 0 * NaN
            // is still NaN.
            _ if f > g => Binary(MTBDDOp::Mul, g.borrowed(), f.borrowed()),
            _ => Binary(MTBDDOp::Mul, f.borrowed(), g.borrowed()),
        }
    } else if OP == MTBDDOp::Div as u8 {
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(tf), Terminal(tg)) => {
                let val = tf.borrow().div(tg.borrow());
                Done(m.get_terminal(val)?)
            }
            (_, Terminal(t)) if t.borrow().is_one() => Done(m.clone_edge(f)),
            (Terminal(t), _) | (_, Terminal(t)) if t.borrow().is_nan() => {
                Done(m.get_terminal(T::nan())?)
            }
            _ => Binary(MTBDDOp::Div, f.borrowed(), g.borrowed()),
        }
    } else if OP == MTBDDOp::Min as u8 {
        if f == g {
            return Ok(Done(m.clone_edge(f)));
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(tf), Terminal(tg)) => Done(match tf.borrow().partial_cmp(tg.borrow()) {
                Some(Ordering::Less | Ordering::Equal) => m.clone_edge(f),
                Some(Ordering::Greater) => m.clone_edge(g),
                None => m.get_terminal(T::nan())?,
            }),
            (Terminal(t), _) | (_, Terminal(t)) if t.borrow().is_nan() => {
                Done(m.get_terminal(T::nan())?)
            }
            _ if f > g => Binary(MTBDDOp::Min, g.borrowed(), f.borrowed()),
            _ => Binary(MTBDDOp::Min, f.borrowed(), g.borrowed()),
        }
    } else if OP == MTBDDOp::Max as u8 {
        if f == g {
            return Ok(Done(m.clone_edge(f)));
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(tf), Terminal(tg)) => Done(match tf.borrow().partial_cmp(tg.borrow()) {
                Some(Ordering::Greater | Ordering::Equal) => m.clone_edge(f),
                Some(Ordering::Less) => m.clone_edge(g),
                None => m.get_terminal(T::nan())?,
            }),
            (Terminal(t), _) | (_, Terminal(t)) if t.borrow().is_nan() => {
                Done(m.get_terminal(T::nan())?)
            }
            _ if f > g => Binary(MTBDDOp::Min, g.borrowed(), f.borrowed()),
            _ => Binary(MTBDDOp::Min, f.borrowed(), g.borrowed()),
        }
    } else {
        unreachable!("invalid binary operator")
    })
}

// --- Function Interface ------------------------------------------------------

//#[cfg(feature = "multi-threading")]
//pub use apply_rec::mt::MTBDDFunctionMT;
pub use apply_rec::MTBDDFunction;

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
            let op = <MTBDDOp as oxidd_core::Countable>::from_usize(i);
            eprintln!("  {op:?}: calls: {calls}, cache queries: {cache_queries} ({terminal_percent} % terminal cases), cache hits: {cache_hits} ({cache_hit_percent} %), reduced: {reduced}");
        }
    }
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [crate::StatCounters; <MTBDDOp as oxidd_core::Countable>::MAX_VALUE + 1] =
    [crate::StatCounters::INIT; <MTBDDOp as oxidd_core::Countable>::MAX_VALUE + 1];

#[cfg(feature = "statistics")]
/// Print statistics to stderr
pub fn print_stats() {
    eprintln!("[oxidd_rules_mtbdd]");
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
