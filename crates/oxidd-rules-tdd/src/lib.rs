//! Rules and other basic definitions for general ternary decision diagrams
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![forbid(unsafe_code)]
// `'id` lifetimes may make the code easier to understand
#![allow(clippy::needless_lifetimes)]

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use oxidd_core::util::{AllocResult, Borrowed};
use oxidd_core::{DiagramRules, Edge, InnerNode, LevelNo, Manager, Node, ReducedOrNew};
use oxidd_derive::Countable;
use oxidd_dump::dddmp::AsciiDisplay;

mod apply_rec;

// --- Reduction Rules ---------------------------------------------------------

/// [`DiagramRules`] for ternary decision diagrams
pub struct TDDRules;

impl<E: Edge, N: InnerNode<E>> DiagramRules<E, N, TDDTerminal> for TDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline]
    #[must_use]
    fn reduce<M: Manager<Edge = E, InnerNode = N, Terminal = TDDTerminal>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let t = it.next().unwrap();
        let u = it.next().unwrap();
        let e = it.next().unwrap();
        debug_assert!(it.next().is_none());

        if t == u && u == e {
            manager.drop_edge(u);
            manager.drop_edge(e);
            ReducedOrNew::Reduced(t)
        } else {
            ReducedOrNew::New(N::new(level, [t, u, e]), Default::default())
        }
    }

    #[inline]
    #[must_use]
    fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        node.children()
    }
}

#[inline(always)]
fn reduce<M>(
    manager: &M,
    level: LevelNo,
    t: M::Edge,
    u: M::Edge,
    e: M::Edge,
    op: TDDOp,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = TDDTerminal>,
{
    let _ = op;
    let tmp = <TDDRules as DiagramRules<_, _, _>>::reduce(manager, level, [t, u, e]);
    if let ReducedOrNew::Reduced(..) = &tmp {
        stat!(reduced op);
    }
    tmp.then_insert(manager, level)
}

// --- Terminal Type -----------------------------------------------------------

/// Terminal nodes in ternary decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum TDDTerminal {
    False,
    Unknown,
    True,
}

impl std::ops::Not for TDDTerminal {
    type Output = TDDTerminal;

    fn not(self) -> Self::Output {
        match self {
            TDDTerminal::False => TDDTerminal::True,
            TDDTerminal::Unknown => TDDTerminal::Unknown,
            TDDTerminal::True => TDDTerminal::False,
        }
    }
}

/// Error returned when parsing a [`TDDTerminal`] from string fails
#[derive(Debug, PartialEq, Eq)]
pub struct ParseTerminalErr;

impl std::str::FromStr for TDDTerminal {
    type Err = ParseTerminalErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f" | "F" | "false" | "False" | "FALSE" | "⊥" => Ok(TDDTerminal::False),
            "u" | "U" | "unknown" | "Unknown" | "UNKNOWN" => Ok(TDDTerminal::Unknown),
            "t" | "T" | "true" | "True" | "TRUE" | "⊤" => Ok(TDDTerminal::True),
            _ => Err(ParseTerminalErr),
        }
    }
}

impl AsciiDisplay for TDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            TDDTerminal::False => f.write_str("F"),
            TDDTerminal::Unknown => f.write_str("U"),
            TDDTerminal::True => f.write_str("T"),
        }
    }
}

impl fmt::Display for TDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TDDTerminal::False => f.write_str("⊥"),
            TDDTerminal::Unknown => f.write_str("U"),
            TDDTerminal::True => f.write_str("⊤"),
        }
    }
}

// --- Operations & Apply Implementation ---------------------------------------

/// Native operations of this TDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum TDDOp {
    Not,
    And,
    Or,
    Nand,
    Nor,
    Xor,
    Equiv,
    Imp,
    ImpStrict,

    /// If-then-else
    Ite,
}

/// Collect the two children of a ternary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<E>, Borrowed<E>, Borrowed<E>) {
    debug_assert_eq!(N::ARITY, 3);
    let mut it = node.children();
    let t = it.next().unwrap();
    let u = it.next().unwrap();
    let e = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (t, u, e)
}

enum Operation<'a, E: 'a + Edge> {
    Binary(TDDOp, Borrowed<'a, E>, Borrowed<'a, E>),
    Not(Borrowed<'a, E>),
    Done(E),
}

/// Terminal case for binary operators
#[inline]
fn terminal_bin<'a, M: Manager<Terminal = TDDTerminal>, const OP: u8>(
    m: &M,
    f: &'a M::Edge,
    g: &'a M::Edge,
) -> Operation<'a, M::Edge> {
    use Node::*;
    use Operation::*;
    use TDDTerminal::*;

    if OP == TDDOp::And as u8 {
        if f == g {
            return Done(m.clone_edge(f));
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == False => {
                Done(m.get_terminal(False).unwrap())
            }
            (Terminal(t), _) if *t.borrow() == True => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == True => Done(m.clone_edge(f)),
            // Both terminal U is handled above
            // One terminal U or both inner
            _ if f > g => Binary(TDDOp::And, g.borrowed(), f.borrowed()),
            _ => Binary(TDDOp::And, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::Or as u8 {
        if f == g {
            return Done(m.clone_edge(f));
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == True => {
                Done(m.get_terminal(True).unwrap())
            }
            (Terminal(t), _) if *t.borrow() == False => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == False => Done(m.clone_edge(f)),
            _ if f > g => Binary(TDDOp::Or, g.borrowed(), f.borrowed()),
            _ => Binary(TDDOp::Or, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::Nand as u8 {
        if f == g {
            return Not(f.borrowed());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == False => {
                Done(m.get_terminal(True).unwrap())
            }
            (Terminal(t), _) if *t.borrow() == True => Not(g.borrowed()),
            (_, Terminal(t)) if *t.borrow() == True => Not(f.borrowed()),
            _ if f > g => Binary(TDDOp::Nand, g.borrowed(), f.borrowed()),
            _ => Binary(TDDOp::Nand, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::Nor as u8 {
        if f == g {
            return Not(f.borrowed());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == True => {
                Done(m.get_terminal(False).unwrap())
            }
            (Terminal(t), _) if *t.borrow() == False => Not(g.borrowed()),
            (_, Terminal(t)) if *t.borrow() == False => Not(f.borrowed()),
            _ if f > g => Binary(TDDOp::Nor, g.borrowed(), f.borrowed()),
            _ => Binary(TDDOp::Nor, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::Xor as u8 {
        if f == g {
            return Done(m.get_terminal(False).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) if *t.borrow() == False => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == False => Done(m.clone_edge(f)),
            (Terminal(t), _) if *t.borrow() == True => Not(g.borrowed()),
            (_, Terminal(t)) if *t.borrow() == True => Not(f.borrowed()),
            _ if f > g => Binary(TDDOp::Xor, g.borrowed(), f.borrowed()),
            _ => Binary(TDDOp::Xor, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::Equiv as u8 {
        if f == g {
            return Done(m.get_terminal(True).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) if *t.borrow() == True => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == True => Done(m.clone_edge(f)),
            (Terminal(t), _) if *t.borrow() == False => Not(g.borrowed()),
            (_, Terminal(t)) if *t.borrow() == False => Not(f.borrowed()),
            _ if f > g => Binary(TDDOp::Equiv, g.borrowed(), f.borrowed()),
            _ => Binary(TDDOp::Equiv, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::Imp as u8 {
        if f == g {
            return Done(m.get_terminal(True).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) if *t.borrow() == False => Done(m.get_terminal(True).unwrap()),
            (_, Terminal(t)) if *t.borrow() == True => Done(m.get_terminal(True).unwrap()),
            (Terminal(t), _) if *t.borrow() == True => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == False => Not(f.borrowed()),
            _ => Binary(TDDOp::Imp, f.borrowed(), g.borrowed()),
        }
    } else if OP == TDDOp::ImpStrict as u8 {
        if f == g {
            return Done(m.get_terminal(False).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Terminal(t), _) if *t.borrow() == True => Done(m.get_terminal(False).unwrap()),
            (_, Terminal(t)) if *t.borrow() == False => Done(m.get_terminal(False).unwrap()),
            (Terminal(t), _) if *t.borrow() == False => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == True => Not(f.borrowed()),
            _ => Binary(TDDOp::ImpStrict, f.borrowed(), g.borrowed()),
        }
    } else {
        unreachable!("invalid binary operator")
    }
}

// --- Function Interface ------------------------------------------------------

//#[cfg(feature = "multi-threading")]
//pub use apply_rec::mt::TDDFunctionMT;
pub use apply_rec::TDDFunction;

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
            let op = <TDDOp as oxidd_core::Countable>::from_usize(i);
            eprintln!("  {op:?}: calls: {calls}, cache queries: {cache_queries} ({terminal_percent} % terminal cases), cache hits: {cache_hits} ({cache_hit_percent} %), reduced: {reduced}");
        }
    }
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [crate::StatCounters; <TDDOp as oxidd_core::Countable>::MAX_VALUE + 1] =
    [crate::StatCounters::INIT; <TDDOp as oxidd_core::Countable>::MAX_VALUE + 1];

#[cfg(feature = "statistics")]
/// Print statistics to stderr
pub fn print_stats() {
    eprintln!("[oxidd_rules_tdd]");
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
