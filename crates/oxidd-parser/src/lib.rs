//! Collection of parsers for various problem formats
//!
//! ## Example
//!
//! ```no_run
//! # use oxidd_parser::load_file::load_file;
//! # use oxidd_parser::*;
//! let parse_options = ParseOptionsBuilder::default().build().unwrap();
//! let Some(problem) = load_file("foo.dimacs", &parse_options) else {
//!     return; // an error message has been printed to stderr
//! };
//! match problem {
//!     Problem::CNF { clauses, .. } => println!("{clauses:?}"),
//!     Problem::Prop { ast, .. } => println!("{ast:?}"),
//!     _ => todo!("problem kind not yet supported"),
//! }
//! ```
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::type_complexity)]

use std::fmt;
use std::num::NonZeroU32;
use std::num::NonZeroUsize;

use derive_builder::Builder;

pub mod dimacs;
mod util;

#[cfg(feature = "load-file")]
pub mod load_file;

/// Variable type
///
/// This is a `NonZeroU32` and not a `u32` to allow space optimizations
pub type Var = NonZeroU32;

/// Variable order including variable names
///
/// An entry `(1337, "Foo")` at index 42 means that the variable with number
/// 1337 in the problem should be placed at index 42.
pub type VarOrder = Vec<(Var, String)>;

/// Different problem kinds that may be returned by the problem parsers
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Problem {
    /// Conjunctive normal form
    CNF {
        /// Number of variables
        num_vars: u32,
        /// Variable order
        var_order: VarOrder,
        /// The clauses. Each clause is a list of literals, where the boolean
        /// field is true iff the literal is negated.
        clauses: Vec<Vec<(Var, bool)>>,

        /// Pre-linearized binary tree representing the clause order, if
        /// non-empty
        clause_order: Vec<ClauseOrderNode>,
    },
    /// Propositional formula
    Prop {
        /// Number of variables
        num_vars: u32,
        /// Variable order
        var_order: VarOrder,
        /// Whether the formula contains exclusive disjunctions
        xor: bool,
        /// Whether the formula contains equivalences
        eq: bool,
        /// The formula
        ast: Prop,
    },
}

/// Nodes in a pre-linearized binary clause order tree
///
/// Since conjunction is a commutative and associative operator, a list of
/// clauses may be processed in different ways. In some cases, `A ∧ (B ∧ C)` may
/// be much more efficient to process `(A ∧ B) ∧ C`. This is why
/// [`Problem::CNF`] allows to specify a clause order.
///
/// We represent clause orders as a pre-linearized binary tree:
/// `[Conj, Conj, Clause(1), Clause(2), Clause(3)]` refers to `(1 ∧ 2) ∧ 3`,
/// `[Conj, Clause(1), Conj, Clause(2), Clause(3)]` refers to `1 ∧ (2 ∧ 3)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ClauseOrderNode {
    /// A clause
    Clause(NonZeroUsize),
    /// Conjunction of the two following subtrees
    Conj,
}

/// Propositional formula
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Prop {
    /// A literal. The boolean field is true iff this is a negated literal.
    Lit(Var, bool),
    /// Negation of the inner propositional formula
    Neg(Box<Prop>),
    /// Conjunction of the inner propositional formulas
    And(Vec<Prop>),
    /// Disjunction of the inner propositional formulas
    Or(Vec<Prop>),
    /// Exclusive disjunction of the inner propositional formulas
    Xor(Vec<Prop>),
    /// Equivalence of the inner propositional formulas
    Eq(Vec<Prop>),
}

impl fmt::Display for Prop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let l = match self {
            Prop::Lit(n, false) => return write!(f, "{n}"),
            Prop::Lit(n, true) => return write!(f, "-{n}"),
            Prop::Neg(e) => return write!(f, "(- {e})"),
            Prop::And(l) => {
                write!(f, "(*")?;
                l
            }
            Prop::Or(l) => {
                write!(f, "(+")?;
                l
            }
            Prop::Xor(l) => {
                write!(f, "(^")?;
                l
            }
            Prop::Eq(l) => {
                write!(f, "(=")?;
                l
            }
        };
        for e in l {
            write!(f, " {e}")?;
        }
        write!(f, ")")
    }
}

/// Options for the parsers
#[non_exhaustive]
#[derive(Clone, Builder, Debug)]
pub struct ParseOptions {
    /// Whether to parse orders (e.g. variable or clause order)
    ///
    /// The [DIMACS satisfiability formats][dimacs], for instance, do not
    /// natively support specifying orders, however it is not uncommon to use
    /// the comment lines for them. But while some files may contain orders in
    /// the comment lines, others may use them for arbitrary comments. Hence, it
    /// may be desired to turn on parsing orders for some files and turn it off
    /// for other files.
    #[builder(default = "false")]
    pub orders: bool,
}
