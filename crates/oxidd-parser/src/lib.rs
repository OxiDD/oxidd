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
//!     Problem::CNF(cnf) => println!("{:?}", cnf.clauses()),
//!     Problem::Prop(prop) => println!("{:?}", prop.formula()),
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
use std::num::NonZeroUsize;

use derive_builder::Builder;

pub mod aiger;
pub mod dimacs;
mod tv_bitvec;
mod util;
mod vec2d;

use tv_bitvec::TVBitVec;
pub use vec2d::{Vec2d, Vec2dIter};

#[cfg(feature = "load-file")]
pub mod load_file;

/// Variable type
///
/// The most significant bit is never set.
pub type Var = usize;

/// A possibly negated variable
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Literal(usize);

impl Literal {
    /// Create a new literal
    pub fn new(negative: bool, var: usize) -> Self {
        debug_assert_eq!(
            var & 1 << (usize::BITS - 1),
            0,
            "Most significant bit of `var` must not be set"
        );
        Self(var << 1 | negative as usize)
    }

    /// Is the literal positive?
    ///
    /// Same as [`!self.negative()`][Self::negative]
    #[inline(always)]
    pub fn positive(self) -> bool {
        self.0 & 1 == 0
    }
    /// Is the literal negative?
    ///
    /// Same as [`!self.positive()`][Self::positive]
    #[inline(always)]
    pub fn negative(self) -> bool {
        !self.positive()
    }

    /// Get the variable number
    #[inline(always)]
    pub fn variable(self) -> Var {
        self.0 >> 1
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sign = if self.positive() { '+' } else { '-' };
        write!(f, "{sign}{}", self.variable())
    }
}
impl fmt::Debug for Literal {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

/// Different problem kinds that may be returned by the problem parsers
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Problem {
    /// Conjunctive normal form
    CNF(Box<CNFProblem>),
    /// Propositional formula
    Prop(Box<PropProblem>),
    /// And-inverter graph
    AIG(Box<AIG>),
}

/// CNF problem instance
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CNFProblem {
    num_vars: usize,
    var_order: Vec<(Var, String)>,
    clauses: Vec2d<Literal>,
    clause_order: Vec<ClauseOrderNode>,
}

impl CNFProblem {
    /// Number of variables
    #[inline(always)]
    pub fn vars(&self) -> usize {
        self.num_vars
    }

    /// Get the clauses
    #[inline(always)]
    pub fn clauses(&self) -> &Vec2d<Literal> {
        &self.clauses
    }
    /// Get the clauses
    pub fn clauses_mut(&mut self) -> &mut Vec2d<Literal> {
        &mut self.clauses
    }

    /// Variable order including variable names
    ///
    /// An entry `(1337, "Foo")` at index 42 means that the variable with number
    /// 1337 in the problem should be placed at index 42.
    #[inline(always)]
    pub fn var_order(&self) -> Option<&[(Var, String)]> {
        if self.var_order.is_empty() {
            None
        } else {
            Some(&self.var_order)
        }
    }

    /// Pre-linearized binary tree representing the clause order
    #[inline(always)]
    pub fn clause_order(&self) -> Option<&[ClauseOrderNode]> {
        if self.clause_order.is_empty() {
            None
        } else {
            Some(&self.clause_order)
        }
    }
}

/// Propositional formula problem
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PropProblem {
    num_vars: usize,
    var_order: Vec<(Var, String)>,
    xor: bool,
    eq: bool,
    ast: Prop,
}

impl PropProblem {
    /// Number of variables
    #[inline(always)]
    pub fn vars(&self) -> usize {
        self.num_vars
    }

    /// Get the formula
    #[inline(always)]
    pub fn formula(&self) -> &Prop {
        &self.ast
    }

    /// Variable order including variable names
    ///
    /// An entry `(1337, "Foo")` at index 42 means that the variable with number
    /// 1337 in the problem should be placed at index 42.
    #[inline(always)]
    pub fn var_order(&self) -> Option<&[(Var, String)]> {
        if self.var_order.is_empty() {
            None
        } else {
            Some(&self.var_order)
        }
    }

    /// Whether the formula may contain exclusive disjunctions
    #[inline(always)]
    pub fn xor_allowed(&self) -> bool {
        self.xor
    }
    /// Whether the formula may contain equivalences
    #[inline(always)]
    pub fn eq_allowed(&self) -> bool {
        self.eq
    }
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
    /// A literal
    Lit(Literal),
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let list = match self {
            Prop::Lit(l) => return write!(f, "{l}"),
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
        for e in list {
            write!(f, " {e}")?;
        }
        write!(f, ")")
    }
}

/// And-inverter graph
///
/// A variable `i` (of a [`Literal`]) has the following meaning:
/// - `0`: `⊥`
/// - `1..(1 + inputs)`: input variable `i - 1`
/// - `(1 + inputs)..(1 + inputs + latches.len())`: output of `latches[i - 1
///   - inputs]`
/// - `(1 + inputs + latches.len())..(1 + inputs + latches.len() +
///   and_gates.len())`: output of `and_gates[i - 1 - inputs - latches.len()]`
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AIG {
    /// Number of input variables
    inputs: usize,
    /// Latch inputs
    latches: Vec<Literal>,
    /// Latch initial values
    latch_init_values: TVBitVec,
    /// And gate inputs
    and_gates: Vec<(Literal, Literal)>,
    /// Outputs
    outputs: Vec<Literal>,
    /// Bad state literals
    bad: Vec<Literal>,
    /// Invariant constraints
    invariants: Vec<Literal>,
    /// Justice properties
    justice: Vec2d<Literal>,
    /// Fairness constraints
    fairness: Vec<Literal>,

    /// Input names
    ///
    /// Either empty or has length `inputs`
    input_names: Vec<Option<String>>,
    /// Latch names
    ///
    /// Either empty or has the same length as `latches`
    latch_names: Vec<Option<String>>,
    /// Output names
    ///
    /// Either empty or has the same length as `outputs`
    output_names: Vec<Option<String>>,
    /// Bad state literal names
    ///
    /// Either empty or has the same length as `bad`
    bad_names: Vec<Option<String>>,
    /// Invariant state names
    ///
    /// Either empty or has the same length as `invariants`
    invariant_names: Vec<Option<String>>,
    /// Justice property names
    ///
    /// Either empty or has the same length as `justice`
    justice_names: Vec<Option<String>>,
    /// Fairness constraint names
    ///
    /// Either empty or has the same length as `fairness`
    fairness_names: Vec<Option<String>>,
}

/// Kinds of AIG vars
///
/// See also [`AIG::decode_var()`]
#[non_exhaustive]
pub enum AIGVar {
    /// false
    False,
    /// i-th input
    Input(usize),
    /// i-th latch
    Latch(usize),
    /// i-th and gate
    AndGate(usize),
}

impl AIG {
    /// Decode `var` into the respective variable kinds
    ///
    /// The indices of [`AIGVar::Latch`] and [`AIGVar::AndGate`] are valid for
    /// the slices returned by [`Self::latches()`] or [`Self::and_gates()`],
    /// respectively.
    #[inline]
    pub fn decode_var(&self, var: Var) -> Option<AIGVar> {
        let first_latch = 1 + self.inputs;
        if var < first_latch {
            Some(if var == 0 {
                AIGVar::False
            } else {
                AIGVar::Input(var - 1)
            })
        } else {
            let first_and_gate = first_latch + self.latches.len();
            if var < first_and_gate {
                Some(AIGVar::Latch(var - first_latch))
            } else if var < first_and_gate + self.and_gates.len() {
                Some(AIGVar::AndGate(var - first_and_gate))
            } else {
                None
            }
        }
    }

    /// Get the number of input variables
    #[inline(always)]
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Get the input literals of latch
    #[inline(always)]
    pub fn latches(&self) -> &[Literal] {
        &self.latches
    }

    /// Get the initial value of latch `i`
    #[inline(always)]
    pub fn latch_init_value(&self, i: usize) -> Option<bool> {
        self.latch_init_values[i]
    }

    /// Get the and gate definitions
    ///
    /// A pair of [`Literal`]s represents the two inputs of an and gate.
    #[inline(always)]
    pub fn and_gates(&self) -> &[(Literal, Literal)] {
        &self.and_gates
    }

    /// Get the name for input `i`
    #[inline(always)]
    pub fn input_name(&self, i: usize) -> Option<&str> {
        self.input_names.get(i)?.as_deref()
    }
    /// Get the name for latch `i`
    #[inline(always)]
    pub fn latch_name(&self, i: usize) -> Option<&str> {
        self.latch_names.get(i)?.as_deref()
    }
    /// Get the name for output `i`
    #[inline(always)]
    pub fn output_name(&self, i: usize) -> Option<&str> {
        self.output_names.get(i)?.as_deref()
    }
    /// Get the name for bad state literal `i`
    #[inline(always)]
    pub fn bad_name(&self, i: usize) -> Option<&str> {
        self.bad_names.get(i)?.as_deref()
    }
    /// Get the name for invariant constraint `i`
    #[inline(always)]
    pub fn invariant_name(&self, i: usize) -> Option<&str> {
        self.invariant_names.get(i)?.as_deref()
    }
    /// Get the name for justice constraint `i`
    #[inline(always)]
    pub fn justice_name(&self, i: usize) -> Option<&str> {
        self.justice_names.get(i)?.as_deref()
    }
}

/// Options for the parsers
#[non_exhaustive]
#[derive(Clone, Builder, Default, Debug)]
pub struct ParseOptions {
    /// Whether to parse orders (e.g., variable or clause order)
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
