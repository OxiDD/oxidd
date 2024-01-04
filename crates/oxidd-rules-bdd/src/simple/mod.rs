//! Simple binary decision diagrams (i.e. no complemented edges)

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use oxidd_core::util::AllocResult;
use oxidd_core::util::Borrowed;
use oxidd_core::DiagramRules;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Manager;
use oxidd_core::Node;
use oxidd_core::ReducedOrNew;
use oxidd_derive::Countable;
use oxidd_dump::dddmp::AsciiDisplay;

use crate::stat;

#[cfg(feature = "multi-threading")]
mod apply_rec_mt;
mod apply_rec_st;

// --- Reduction Rules ---------------------------------------------------------

/// [`DiagramRules`] for simple binary decision diagrams
pub struct BDDRules;

impl<E: Edge, N: InnerNode<E>> DiagramRules<E, N, BDDTerminal> for BDDRules {
    type Cofactors<'a> = N::ChildrenIter<'a> where N: 'a, E: 'a;

    #[inline(always)]
    #[must_use]
    fn reduce<M: Manager<Edge = E, InnerNode = N>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let f0 = it.next().unwrap();
        let f1 = it.next().unwrap();
        debug_assert!(it.next().is_none());

        if f0 == f1 {
            manager.drop_edge(f1);
            ReducedOrNew::Reduced(f0)
        } else {
            ReducedOrNew::New(N::new(level, [f0, f1]), Default::default())
        }
    }

    #[inline(always)]
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
    e: M::Edge,
    _op: BDDOp,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal>,
{
    let tmp = <BDDRules as DiagramRules<_, _, _>>::reduce(manager, level, [t, e]);
    if let ReducedOrNew::Reduced(..) = &tmp {
        stat!(reduced _op);
    }
    tmp.then_insert(manager, level)
}

// --- Terminal Type -----------------------------------------------------------

/// Terminal nodes in simple binary decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
#[repr(u8)]
pub enum BDDTerminal {
    #[allow(missing_docs)]
    False,
    #[allow(missing_docs)]
    True,
}

/// Error returned when parsing a [`BDDTerminal`] from string fails
#[derive(Debug, PartialEq, Eq)]
pub struct ParseTerminalErr;

impl std::str::FromStr for BDDTerminal {
    type Err = ParseTerminalErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "t" | "T" | "true" | "True" | "TRUE" | "⊤" => Ok(BDDTerminal::True),
            "f" | "F" | "false" | "False" | "FALSE" | "⊥" => Ok(BDDTerminal::False),
            _ => Err(ParseTerminalErr),
        }
    }
}

impl AsciiDisplay for BDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            BDDTerminal::False => f.write_str("F"),
            BDDTerminal::True => f.write_str("T"),
        }
    }
}

impl fmt::Display for BDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BDDTerminal::False => f.write_str("⊥"),
            BDDTerminal::True => f.write_str("⊤"),
        }
    }
}

impl std::ops::Not for BDDTerminal {
    type Output = BDDTerminal;

    #[inline]
    fn not(self) -> BDDTerminal {
        match self {
            BDDTerminal::False => BDDTerminal::True,
            BDDTerminal::True => BDDTerminal::False,
        }
    }
}

// --- Operations & Apply Implementation ---------------------------------------

/// Native operators of this BDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Ord, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum BDDOp {
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

    /// Forall quantification
    Forall,
    /// Existential quantification
    Exist,
    /// Unique quantification
    Unique,
}

enum Operation<'a, E: 'a + Edge> {
    Binary(BDDOp, Borrowed<'a, E>, Borrowed<'a, E>),
    Not(Borrowed<'a, E>),
    Done(E),
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [crate::StatCounters; 13] = [crate::StatCounters::INIT; 13];

#[cfg(feature = "statistics")]
/// Print statistics to stderr
pub fn print_stats() {
    eprintln!("[oxidd_rules_bdd::simple]");
    // FIXME: we should auto generate the labels
    crate::StatCounters::print(
        &STAT_COUNTERS,
        &[
            "Not",
            "And",
            "Or",
            "Nand",
            "Nor",
            "Xor",
            "Equiv",
            "Imp",
            "ImpStrict",
            "Ite",
            "Forall",
            "Exist",
            "Unique",
        ],
    );
}

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<E>, Borrowed<E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = node.children();
    let f0 = it.next().unwrap();
    let f1 = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (f0, f1)
}

/// Terminal case for binary operators
#[inline]
fn terminal_bin<'a, M: Manager<Terminal = BDDTerminal>, const OP: u8>(
    m: &M,
    f: &'a M::Edge,
    g: &'a M::Edge,
) -> Operation<'a, M::Edge> {
    use BDDTerminal::*;
    use Node::*;
    use Operation::*;

    if OP == BDDOp::And as u8 {
        if f == g {
            return Done(m.clone_edge(f));
        }
        match (m.get_node(f), m.get_node(g)) {
            // Unique representation of {f, g} for commutative functions
            (Inner(_), Inner(_)) if f > g => Binary(BDDOp::And, g.borrowed(), f.borrowed()),
            (Inner(_), Inner(_)) => Binary(BDDOp::And, g.borrowed(), f.borrowed()),
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == False => {
                Done(m.get_terminal(False).unwrap())
            }
            (Terminal(_), _) => Done(m.clone_edge(g)),
            (_, Terminal(_)) => Done(m.clone_edge(f)),
        }
    } else if OP == BDDOp::Or as u8 {
        if f == g {
            return Done(m.clone_edge(f));
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) if f > g => Binary(BDDOp::Or, g.borrowed(), f.borrowed()),
            (Inner(_), Inner(_)) => Binary(BDDOp::Or, g.borrowed(), f.borrowed()),
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == True => {
                Done(m.get_terminal(True).unwrap())
            }
            (Terminal(_), _) => Done(m.clone_edge(g)),
            (_, Terminal(_)) => Done(m.clone_edge(f)),
        }
    } else if OP == BDDOp::Nand as u8 {
        if f == g {
            return Not(f.borrowed());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) if f > g => Binary(BDDOp::Nand, g.borrowed(), f.borrowed()),
            (Inner(_), Inner(_)) => Binary(BDDOp::Nand, g.borrowed(), f.borrowed()),
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == False => {
                Done(m.get_terminal(True).unwrap())
            }
            (Terminal(_), _) => Not(g.borrowed()),
            (_, Terminal(_)) => Not(f.borrowed()),
        }
    } else if OP == BDDOp::Nor as u8 {
        if f == g {
            return Not(f.borrowed());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) if f > g => Binary(BDDOp::Nor, g.borrowed(), f.borrowed()),
            (Inner(_), Inner(_)) => Binary(BDDOp::Nor, g.borrowed(), f.borrowed()),
            (Terminal(t), _) | (_, Terminal(t)) if *t.borrow() == True => {
                Done(m.get_terminal(False).unwrap())
            }
            (Terminal(_), _) => Not(g.borrowed()),
            (_, Terminal(_)) => Not(f.borrowed()),
        }
    } else if OP == BDDOp::Xor as u8 {
        if f == g {
            return Done(m.get_terminal(False).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) if f > g => Binary(BDDOp::Xor, g.borrowed(), f.borrowed()),
            (Inner(_), Inner(_)) => Binary(BDDOp::Xor, f.borrowed(), g.borrowed()),
            (Terminal(t), _) if *t.borrow() == False => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == False => Done(m.clone_edge(f)),
            (Terminal(_), _) => Not(g.borrowed()),
            (_, Terminal(_)) => Not(f.borrowed()),
        }
    } else if OP == BDDOp::Equiv as u8 {
        if f == g {
            return Done(m.get_terminal(True).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) if f > g => Binary(BDDOp::Equiv, g.borrowed(), f.borrowed()),
            (Inner(_), Inner(_)) => Binary(BDDOp::Equiv, f.borrowed(), g.borrowed()),
            (Terminal(t), _) if *t.borrow() == True => Done(m.clone_edge(g)),
            (_, Terminal(t)) if *t.borrow() == True => Done(m.clone_edge(f)),
            (Terminal(_), _) => Not(g.borrowed()),
            (_, Terminal(_)) => Not(f.borrowed()),
        }
    } else if OP == BDDOp::Imp as u8 {
        if f == g {
            return Done(m.get_terminal(True).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) => Binary(BDDOp::Imp, f.borrowed(), g.borrowed()),
            (Terminal(t), _) if *t.borrow() == False => Done(m.get_terminal(True).unwrap()),
            (_, Terminal(t)) if *t.borrow() == True => Done(m.get_terminal(True).unwrap()),
            (Terminal(_), _) => Done(m.clone_edge(g)),
            (_, Terminal(_)) => Not(f.borrowed()),
        }
    } else if OP == BDDOp::ImpStrict as u8 {
        if f == g {
            return Done(m.get_terminal(False).unwrap());
        }
        match (m.get_node(f), m.get_node(g)) {
            (Inner(_), Inner(_)) => Binary(BDDOp::ImpStrict, f.borrowed(), g.borrowed()),
            (Terminal(t), _) if *t.borrow() == True => Done(m.get_terminal(False).unwrap()),
            (_, Terminal(t)) if *t.borrow() == False => Done(m.get_terminal(False).unwrap()),
            (Terminal(_), _) => Done(m.clone_edge(g)),
            (_, Terminal(_)) => Not(f.borrowed()),
        }
    } else {
        unreachable!("invalid binary operator")
    }
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasBDDOpApplyCache<M: Manager>: HasApplyCache<M, Operator = BDDOp> {}
impl<M: Manager + HasApplyCache<M, Operator = BDDOp>> HasBDDOpApplyCache<M> for M {}

#[cfg(feature = "multi-threading")]
pub use apply_rec_mt::BDDFunctionMT;
pub use apply_rec_st::BDDFunction;
