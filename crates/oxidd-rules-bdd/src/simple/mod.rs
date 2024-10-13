//! Simple binary decision diagrams (i.e. no complemented edges)

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use oxidd_core::util::{AllocResult, Borrowed};
use oxidd_core::{DiagramRules, Edge, HasLevel, InnerNode, LevelNo, Manager, Node, ReducedOrNew};
use oxidd_derive::Countable;
use oxidd_dump::dddmp::AsciiDisplay;

use crate::stat;

mod apply_rec;

// --- Reduction Rules ---------------------------------------------------------

/// [`DiagramRules`] for simple binary decision diagrams
pub struct BDDRules;

impl<E: Edge, N: InnerNode<E>> DiagramRules<E, N, BDDTerminal> for BDDRules {
    type Cofactors<'a>
        = N::ChildrenIter<'a>
    where
        N: 'a,
        E: 'a;

    #[inline(always)]
    #[must_use]
    fn reduce<M: Manager<Edge = E, InnerNode = N>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let f_then = it.next().unwrap();
        let f_else = it.next().unwrap();
        debug_assert!(it.next().is_none());

        if f_then == f_else {
            manager.drop_edge(f_else);
            ReducedOrNew::Reduced(f_then)
        } else {
            ReducedOrNew::New(N::new(level, [f_then, f_else]), Default::default())
        }
    }

    #[inline(always)]
    #[must_use]
    fn cofactors(_tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        node.children()
    }

    #[inline(always)]
    fn cofactor(_tag: E::Tag, node: &N, n: usize) -> Borrowed<E> {
        node.child(n)
    }
}

/// Apply the reduction rules, creating a node in `manager` if necessary
#[inline(always)]
fn reduce<M>(manager: &M, level: LevelNo, t: M::Edge, e: M::Edge, op: BDDOp) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = BDDTerminal>,
{
    // We do not use `DiagramRules::reduce()` here, as the iterator is
    // apparently not fully optimized away.
    if t == e {
        stat!(reduced op);
        manager.drop_edge(e);
        return Ok(t);
    }
    oxidd_core::LevelView::get_or_insert(
        &mut manager.level(level),
        M::InnerNode::new(level, [t, e]),
    )
}

/// Collect the two children of a binary node
#[inline]
#[must_use]
fn collect_children<E: Edge, N: InnerNode<E>>(node: &N) -> (Borrowed<E>, Borrowed<E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = node.children();
    let f_then = it.next().unwrap();
    let f_else = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (f_then, f_else)
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
            (Inner(_), Inner(_)) => Binary(BDDOp::And, f.borrowed(), g.borrowed()),
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
            (Inner(_), Inner(_)) => Binary(BDDOp::Or, f.borrowed(), g.borrowed()),
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
            (Inner(_), Inner(_)) => Binary(BDDOp::Nand, f.borrowed(), g.borrowed()),
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
            (Inner(_), Inner(_)) => Binary(BDDOp::Nor, f.borrowed(), g.borrowed()),
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

// --- Operations & Apply Implementation ---------------------------------------

/// Native operators of this BDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Ord, Countable, Debug)]
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

    Substitute,

    Restrict,
    /// Forall quantification
    Forall,
    /// Existential quantification
    Exist,
    /// Unique quantification
    Unique,

    ForallAnd,
    ForallOr,
    ForallNand,
    ForallNor,
    ForallXor,
    ForallEquiv,
    ForallImp,
    ForallImpStrict,

    ExistAnd,
    ExistOr,
    ExistNand,
    ExistNor,
    ExistXor,
    ExistEquiv,
    ExistImp,
    ExistImpStrict,

    UniqueAnd,
    UniqueOr,
    UniqueNand,
    UniqueNor,
    UniqueXor,
    UniqueEquiv,
    UniqueImp,
    UniqueImpStrict,
}

impl BDDOp {
    const fn from_apply_quant(q: u8, op: u8) -> Self {
        if q == BDDOp::And as u8 {
            match () {
                _ if op == BDDOp::And as u8 => BDDOp::ForallAnd,
                _ if op == BDDOp::Or as u8 => BDDOp::ForallOr,
                _ if op == BDDOp::Nand as u8 => BDDOp::ForallNand,
                _ if op == BDDOp::Nor as u8 => BDDOp::ForallNor,
                _ if op == BDDOp::Xor as u8 => BDDOp::ForallXor,
                _ if op == BDDOp::Equiv as u8 => BDDOp::ForallEquiv,
                _ if op == BDDOp::Imp as u8 => BDDOp::ForallImp,
                _ if op == BDDOp::ImpStrict as u8 => BDDOp::ForallImpStrict,
                _ => panic!("invalid OP"),
            }
        } else if q == BDDOp::Or as u8 {
            match () {
                _ if op == BDDOp::And as u8 => BDDOp::ExistAnd,
                _ if op == BDDOp::Or as u8 => BDDOp::ExistOr,
                _ if op == BDDOp::Nand as u8 => BDDOp::ExistNand,
                _ if op == BDDOp::Nor as u8 => BDDOp::ExistNor,
                _ if op == BDDOp::Xor as u8 => BDDOp::ExistXor,
                _ if op == BDDOp::Equiv as u8 => BDDOp::ExistEquiv,
                _ if op == BDDOp::Imp as u8 => BDDOp::ExistImp,
                _ if op == BDDOp::ImpStrict as u8 => BDDOp::ExistImpStrict,
                _ => panic!("invalid OP"),
            }
        } else if q == BDDOp::Xor as u8 {
            match () {
                _ if op == BDDOp::And as u8 => BDDOp::UniqueAnd,
                _ if op == BDDOp::Or as u8 => BDDOp::UniqueOr,
                _ if op == BDDOp::Nand as u8 => BDDOp::UniqueNand,
                _ if op == BDDOp::Nor as u8 => BDDOp::UniqueNor,
                _ if op == BDDOp::Xor as u8 => BDDOp::UniqueXor,
                _ if op == BDDOp::Equiv as u8 => BDDOp::UniqueEquiv,
                _ if op == BDDOp::Imp as u8 => BDDOp::UniqueImp,
                _ if op == BDDOp::ImpStrict as u8 => BDDOp::UniqueImpStrict,
                _ => panic!("invalid OP"),
            }
        } else {
            panic!("invalid quantifier");
        }
    }
}

enum Operation<'a, E: 'a + Edge> {
    Binary(BDDOp, Borrowed<'a, E>, Borrowed<'a, E>),
    Not(Borrowed<'a, E>),
    Done(E),
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [crate::StatCounters; <BDDOp as oxidd_core::Countable>::MAX_VALUE + 1] =
    [crate::StatCounters::INIT; <BDDOp as oxidd_core::Countable>::MAX_VALUE + 1];

#[cfg(feature = "statistics")]
/// Print statistics to stderr
pub fn print_stats() {
    eprintln!("[oxidd_rules_bdd::simple]");
    crate::StatCounters::print::<BDDOp>(&STAT_COUNTERS);
}

// --- Utility Functions -------------------------------------------------------

#[inline]
fn is_var<M>(manager: &M, node: &M::InnerNode) -> bool
where
    M: Manager<Terminal = BDDTerminal>,
{
    let t = node.child(0);
    let e = node.child(1);
    manager.get_node(&t).is_terminal(&BDDTerminal::True)
        && manager.get_node(&e).is_terminal(&BDDTerminal::False)
}

#[inline]
#[track_caller]
fn var_level<M>(manager: &M, e: Borrowed<M::Edge>) -> LevelNo
where
    M: Manager<Terminal = BDDTerminal>,
    M::InnerNode: HasLevel,
{
    let node = manager
        .get_node(&e)
        .expect_inner("Expected a variable but got a terminal node");
    debug_assert!(is_var(manager, node));
    node.level()
}

// --- Function Interface ------------------------------------------------------

#[cfg(feature = "multi-threading")]
pub use apply_rec::mt::BDDFunctionMT;
pub use apply_rec::BDDFunction;
