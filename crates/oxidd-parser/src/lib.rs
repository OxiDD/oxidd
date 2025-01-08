//! Collection of parsers for various logical problem formats
//!
//! Every format is parsed into a [`Circuit`], i.e., a propositional formula
//! with structure sharing, to enable (mostly) uniform handling of different
//! formats in an application.
//!
//! ## Example
//!
//! ```no_run
//! # use oxidd_parser::*;
//! let parse_options = ParseOptionsBuilder::default().build().unwrap();
//! let Some(problem) = load_file("foo.dimacs", &parse_options) else {
//!     return; // an error message has been printed to stderr
//! };
//! println!("circuit: {:?}", problem.circuit);
//! println!("additional details: {:?}", problem.details);
//! ```
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::type_complexity)]

use std::collections::hash_map::Entry;
use std::fmt::{self, Write};

use bitvec::{slice::BitSlice, vec::BitVec};
use bumpalo::boxed::Box as BumpBox;
use bumpalo::collections::Vec as BumpVec;
use derive_builder::Builder;
use rustc_hash::FxHashMap;

pub mod aiger;
pub mod dimacs;
pub mod nnf;
mod tv_bitvec;
mod util;
mod vec2d;

use tv_bitvec::TVBitVec;
pub use vec2d::{Vec2d, Vec2dIter};

#[cfg(feature = "load-file")]
mod load_file;
#[cfg(feature = "load-file")]
pub use load_file::*;

/// Variable type
///
/// The two most significant bits are never set.
pub type Var = usize;

/// A possibly negated variable
///
/// There are three kinds of variables: constants ([`Literal::FALSE`] and
/// [`Literal::TRUE`]), circuit inputs, and gates.
///
/// Concerning ordering, the following properties hold: Given two [`Literal`]s
/// `a, b`, `a <= b` is equivalent to the lexicographic comparison
/// `(a.positive(), a.is_negative()) <= (b.positive(), b.is_negative())`.
/// Furthermore, `Literal::FALSE <= a` holds (and thus `Literal::TRUE <= a` if
/// `a != Literal::FALSE`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Literal(usize);

impl Literal {
    // We rely on POLARITY_BIT being the least significant bit for the ordering
    const POLARITY_BIT: u32 = 0;
    const GATE_BIT: u32 = 1;
    const VAR_LSB: u32 = 2;

    /// Maximum input number representable as literal
    pub const MAX_INPUT: usize = (usize::MAX >> Self::VAR_LSB) - 2;
    /// Maximum gate number representable as literal
    pub const MAX_GATE: usize = usize::MAX >> Self::VAR_LSB;

    /// Literal representing the constant `⊥`
    ///
    /// This is literal is considered to be positive (i.e., not negated):
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::FALSE.is_positive());
    /// assert!(Literal::FALSE < Literal::TRUE);
    /// ```
    pub const FALSE: Self = Self(0);
    /// Literal representing the constant `⊤`
    ///
    /// This is literal is considered to be negated:
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::TRUE.is_negative());
    /// assert!(Literal::TRUE < Literal::from_input(false, 0));
    /// assert!(Literal::TRUE < Literal::from_gate(false, 0));
    /// ```
    pub const TRUE: Self = Self(1 << Self::POLARITY_BIT);

    /// Undefined literal
    ///
    /// This is considered to be a positive (i.e., not negated) input, but its
    /// variable number is larger than `Self::MAX_INPUT`. Attempting to create
    /// this literal using [`Self::from_input()`] will trigger a debug
    /// assertion.
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::UNDEF.is_positive());
    /// assert!(Literal::UNDEF.is_input());
    /// assert_eq!(Literal::UNDEF.get_input(), Some(Literal::MAX_INPUT + 1));
    /// ```
    pub const UNDEF: Self = Self((Self::MAX_INPUT + 2) << Self::VAR_LSB);

    /// Create a new literal from an input variable
    #[track_caller]
    #[inline]
    pub const fn from_input(negative: bool, input: Var) -> Self {
        debug_assert!(input <= Self::MAX_INPUT, "input too large");
        Self(((input + 1) << Self::VAR_LSB) | ((negative as usize) << Self::POLARITY_BIT))
    }

    /// Create a new literal from an input variable (value range `1..`) or false
    /// (`0`)
    #[track_caller]
    #[inline]
    const fn from_input_or_false(negative: bool, input: Var) -> Self {
        debug_assert!(input <= Self::MAX_INPUT + 1, "input too large");
        Self((input << Self::VAR_LSB) | ((negative as usize) << Self::POLARITY_BIT))
    }

    /// Create a new literal from a gate number
    #[track_caller]
    #[inline]
    pub const fn from_gate(negative: bool, gate: Var) -> Self {
        debug_assert!(gate <= Self::MAX_GATE, "gate number too large");
        Self(
            (gate << Self::VAR_LSB)
                | (1 << Self::GATE_BIT)
                | ((negative as usize) << Self::POLARITY_BIT),
        )
    }

    /// Is the literal positive?
    ///
    /// Same as [`!self.is_negative()`][Self::is_negative]
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::from_input(false, 42).is_positive());
    /// assert!(!Literal::from_input(true, 1337).is_positive());
    /// ```
    #[inline(always)]
    pub const fn is_positive(self) -> bool {
        self.0 & (1 << Self::POLARITY_BIT) == 0
    }
    /// Is the literal negative?
    ///
    /// Same as [`!self.is_positive()`][Self::is_positive]
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::from_gate(true, 42).is_negative());
    /// assert!(!Literal::from_gate(false, 1337).is_negative());
    /// ```
    #[inline(always)]
    pub const fn is_negative(self) -> bool {
        !self.is_positive()
    }

    /// Get the positive variant of this literal
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::from_input(true, 42).positive().is_positive());
    /// ```
    #[inline(always)]
    pub const fn positive(self) -> Self {
        Self(self.0 & !(1 << Self::POLARITY_BIT))
    }

    /// Get the negative variant of this literal
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert_eq!(Literal::from_input(true, 42), Literal::from_input(false, 42).negative());
    /// assert!(Literal::from_input(false, 42).negative().is_negative());
    /// ```
    #[inline(always)]
    pub const fn negative(self) -> Self {
        Self(self.0 | (1 << Self::POLARITY_BIT))
    }

    /// Does this literal refer to an input?
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::from_input(false, 42).is_input());
    /// assert!(!Literal::FALSE.is_input());
    /// assert!(!Literal::TRUE.is_input());
    /// assert!(!Literal::from_gate(false, 1337).is_input());
    /// ```
    #[inline(always)]
    pub const fn is_input(self) -> bool {
        self.get_input().is_some()
    }

    /// Check if this literal refers to a gate
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert!(Literal::from_gate(false, 42).is_gate());
    /// assert!(!Literal::FALSE.is_gate());
    /// assert!(!Literal::TRUE.is_gate());
    /// assert!(!Literal::from_input(false, 1337).is_gate());
    /// ```
    #[inline(always)]
    pub const fn is_gate(self) -> bool {
        self.0 & (1 << Self::GATE_BIT) != 0
    }

    /// Get the input number (if this literal refers to an input)
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert_eq!(Literal::from_input(false, 42).get_input(), Some(42));
    /// assert_eq!(Literal::FALSE.get_input(), None);
    /// assert_eq!(Literal::TRUE.get_input(), None);
    /// assert_eq!(Literal::from_gate(false, 42).get_input(), None);
    /// ```
    #[inline]
    pub const fn get_input(self) -> Option<Var> {
        if self.is_gate() {
            return None;
        }
        (self.0 >> Self::VAR_LSB).checked_sub(1)
    }

    /// Get the gate number (if this literal refers to an input)
    ///
    /// ```
    /// # use oxidd_parser::Literal;
    /// assert_eq!(Literal::from_gate(false, 42).get_gate_no(), Some(42));
    /// assert_eq!(Literal::FALSE.get_gate_no(), None);
    /// assert_eq!(Literal::TRUE.get_gate_no(), None);
    /// assert_eq!(Literal::from_input(false, 42).get_gate_no(), None);
    /// ```
    ///
    /// See also: [`Circuit::gate()`]
    #[inline]
    pub const fn get_gate_no(self) -> Option<Var> {
        if self.is_gate() {
            return Some(self.0 >> Self::VAR_LSB);
        }
        None
    }

    /// Map this literal based on `gate_map`
    ///
    /// If this literal refers to a gate, the method performs a lookup for the
    /// gate number in `gate_map`. If the gate number is in bounds, the return
    /// value is the mapped literal with its sign adjusted, otherwise the return
    /// value is [`Literal::UNDEF`]. Non-gate literals are returned as they are.
    #[inline]
    pub fn apply_gate_map(self, gate_map: &[Literal]) -> Self {
        if let Some(gate) = self.get_gate_no() {
            if let Some(mapped) = gate_map.get(gate) {
                *mapped ^ self.is_negative()
            } else {
                Self::UNDEF
            }
        } else {
            self
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let i = self.0 >> Literal::VAR_LSB;
        let (kind, i) = if self.0 & (1 << Self::GATE_BIT) != 0 {
            ('g', i)
        } else {
            if i == 0 {
                return f.write_char(if self.is_positive() { '⊤' } else { '⊥' });
            }
            if i == Literal::MAX_INPUT + 2 {
                return f.write_str(if self.is_positive() { "+U" } else { "-U" });
            }
            ('i', i - 1)
        };

        let sign = if self.is_positive() { '+' } else { '-' };
        write!(f, "{sign}{kind}{i}")
    }
}
impl fmt::Debug for Literal {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl std::ops::Not for Literal {
    type Output = Self;

    fn not(self) -> Self {
        Self(self.0 ^ (1 << Self::POLARITY_BIT))
    }
}

impl std::ops::BitXor<bool> for Literal {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: bool) -> Self::Output {
        Self(self.0 ^ ((rhs as usize) << Self::POLARITY_BIT))
    }
}

/// Rooted tree with values of type `T` at the leaves
#[allow(missing_docs)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Tree<T> {
    Inner(Box<[Tree<T>]>),
    Leaf(T),
}

impl<T: Clone> Tree<T> {
    fn flatten_into(&self, into: &mut Vec<T>) {
        match self {
            Tree::Inner(sub) => sub.iter().for_each(|t| t.flatten_into(into)),
            Tree::Leaf(v) => into.push(v.clone()),
        }
    }
}

/// Problem that may be returned by the problem parsers
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Problem {
    /// Combinational circuit as the core of the problem
    pub circuit: Circuit,

    /// Additional details on the problem
    pub details: ProblemDetails,
}

impl Problem {
    /// Simplify the circuit (see [`Circuit::simplify()`]), and map the gate
    /// literals in the additional problem details accordingly
    ///
    /// If the reachable circuit fragment contains a cycle, this method returns
    /// `Err(l)`, where `l` represents a gate that is part of the cycle.
    ///
    /// On success, `simplify()` returns the simplified problem along with a
    /// `Vec<Literal>` which maps the gates of `self` to literals valid in the
    /// simplified circuit. The simplified circuit only contains gates reachable
    /// from the problem details (except for the
    /// [AIGER literal map][AIGERDetails::map_aiger_literal]).
    pub fn simplify(&self) -> Result<(Self, Vec<Literal>), Literal> {
        let (circuit, map) = match &self.details {
            ProblemDetails::Root(l) => self.circuit.simplify([*l]),
            ProblemDetails::AIGER(aig) => {
                let aig = &**aig;
                self.circuit.simplify(
                    aig.latches
                        .iter()
                        .chain(aig.outputs.iter())
                        .chain(aig.bad.iter())
                        .chain(aig.invariants.iter())
                        .chain(aig.justice.all_elements().iter())
                        .chain(aig.fairness.iter())
                        .copied(),
                )
            }
        }?;
        let new_problem = Self {
            circuit,
            details: self.details.apply_gate_map(&map),
        };
        Ok((new_problem, map))
    }
}

/// Details of a [`Problem`] in addition to the circuit structure
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ProblemDetails {
    /// Simple circuit with a single root
    Root(Literal),
    /// Details from the AIGER format
    AIGER(Box<AIGERDetails>),
}

impl ProblemDetails {
    /// Map the literals in `self` based on `gate_map`
    ///
    /// See [`Literal::apply_gate_map()`] for more details.
    pub fn apply_gate_map(&self, map: &[Literal]) -> Self {
        match self {
            ProblemDetails::Root(l) => ProblemDetails::Root(l.apply_gate_map(map)),
            ProblemDetails::AIGER(aig) => ProblemDetails::AIGER(Box::new(aig.apply_gate_map(map))),
        }
    }

    /// In-place version of [`Self::apply_gate_map()`]
    pub fn apply_gate_map_in_place(&mut self, map: &[Literal]) {
        match self {
            ProblemDetails::Root(l) => *l = l.apply_gate_map(map),
            ProblemDetails::AIGER(aig) => aig.apply_gate_map_in_place(map),
        }
    }
}

/// Variable set, potentially along with a variable order and variable names
///
/// The variable numbers are in range `0..self.len()`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VarSet {
    /// Number of variables
    len: usize,

    /// Permutation of the variables. `order[0]` is supposed to be the number of
    /// the top-most variable.
    order: Vec<Var>,
    /// If present, `order` is just the flattened tree
    order_tree: Option<Tree<Var>>,

    /// Mapping from variable numbers to optional names. Has minimal length,
    /// i.e., `names.last() != Some(&None)`.
    names: Vec<Option<String>>,
}

impl VarSet {
    /// Create a variable set without a variable order and names
    #[inline(always)]
    pub const fn new(n: usize) -> Self {
        Self {
            len: n,
            order: Vec::new(),
            order_tree: None,
            names: Vec::new(),
        }
    }

    /// Create a variable set with the given names
    ///
    /// The number of variables is given by `names.len()`.
    pub fn with_names(mut names: Vec<Option<String>>) -> Self {
        let len = names.len();
        while let Some(None) = names.last() {
            names.pop();
        }
        Self {
            len,
            order: Vec::new(),
            order_tree: None,
            names,
        }
    }

    /// Number of variables
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true iff the number of variables is 0
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the linear variable order, if present
    ///
    /// If [`self.order_tree()`][Self::order_tree] is not `None`, then it is the
    /// flattened tree.
    #[inline]
    pub fn order(&self) -> Option<&[Var]> {
        if self.len != self.order.len() {
            None
        } else {
            Some(&self.order)
        }
    }

    /// Get the tree of variable groups, if present
    ///
    /// This may be useful for, e.g., group sifting.
    #[inline]
    pub fn order_tree(&self) -> Option<&Tree<Var>> {
        self.order_tree.as_ref()
    }

    /// Get the name for variable `var`
    #[inline]
    pub fn name(&self, var: Var) -> Option<&str> {
        self.names.get(var)?.as_deref()
    }

    #[allow(unused)]
    fn check_valid(&self) {
        assert!(self.order.is_empty() || self.order.len() == self.len);
        assert!(!self.order.is_empty() || self.order_tree.is_none());
        assert_ne!(self.names.last(), Some(&None));
    }
}

type GateVec2d = vec2d::with_metadata::Vec2d<Literal, { Circuit::GATE_METADATA_BITS }>;

/// Combinational circuit with negation as well as n-ary AND, OR, and XOR gates
#[derive(Clone, PartialEq, Eq)]
pub struct Circuit {
    inputs: VarSet,
    gates: GateVec2d,
}

/// Logical gate including the literals representing its inputs
///
/// This enum type includes the phony gate variants `False` (representing the
/// constant false) and `Input` (for a circuit input). This is mainly to make
/// using [`Circuit::gate()`] more ergonomic.
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Gate<'a> {
    pub kind: GateKind,
    pub inputs: &'a [Literal],
}

impl fmt::Debug for Gate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = f.debug_tuple(match self.kind {
            GateKind::And => "and",
            GateKind::Or => "or",
            GateKind::Xor => "xor",
        });
        for literal in self.inputs {
            builder.field(literal);
        }
        builder.finish()
    }
}

#[allow(unused)] // used in tests
impl<'a> Gate<'a> {
    const fn and(inputs: &'a [Literal]) -> Self {
        Self {
            kind: GateKind::And,
            inputs,
        }
    }
    const fn or(inputs: &'a [Literal]) -> Self {
        Self {
            kind: GateKind::Or,
            inputs,
        }
    }
    const fn xor(inputs: &'a [Literal]) -> Self {
        Self {
            kind: GateKind::Xor,
            inputs,
        }
    }
}

/// Kind of a logical gate in a [`Circuit`]
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum GateKind {
    And,
    Or,
    Xor,
}

impl GateKind {
    /// Get the literal representing a gate of this kind without any inputs
    const fn empty_gate(self) -> Literal {
        match self {
            GateKind::And => Literal::TRUE,
            GateKind::Or | GateKind::Xor => Literal::FALSE,
        }
    }
}

impl From<usize> for GateKind {
    fn from(value: usize) -> Self {
        match value {
            _ if value == GateKind::And as usize => GateKind::And,
            _ if value == GateKind::Or as usize => GateKind::Or,
            _ => GateKind::Xor,
        }
    }
}

impl Circuit {
    const GATE_METADATA_BITS: u32 = 2;

    /// Create an empty circuit with the given circuit inputs
    #[inline(always)]
    pub fn new(inputs: VarSet) -> Self {
        Self {
            inputs,
            gates: Default::default(),
        }
    }

    /// Get the circuit inputs
    #[inline(always)]
    pub fn inputs(&self) -> &VarSet {
        &self.inputs
    }

    /// Reserve space for at least `additional` more gates. Note that this will
    /// not reserve space for gate inputs. To that end, use
    /// [`Self::reserve_gate_inputs()`].
    ///
    /// This is essentially a wrapper around [`Vec::reserve()`], so the
    /// documentation there provides more details.
    #[inline(always)]
    pub fn reserve_gates(&mut self, additional: usize) {
        self.gates.reserve_vectors(additional);
    }

    /// Reserve space for at least `additional` more gate inputs (the space is
    /// shared between all gates). Note that this will not reserve space for
    /// the gate metadata. To that end, use [`Self::reserve_gates()`].
    ///
    /// This is essentially a wrapper around [`Vec::reserve()`], so the
    /// documentation there provides more details.
    #[inline(always)]
    pub fn reserve_gate_inputs(&mut self, additional: usize) {
        self.gates.reserve_elements(additional);
    }

    /// Get the number of gates in this circuit
    #[inline(always)]
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get the [`Gate`] for `literal`
    ///
    /// Returns [`None`] iff `literal` refers to a constant, a circuit input, or
    /// a gate that is not present in this circuit.
    ///
    /// See also: [`Self::gate_for_no()`]
    #[inline]
    pub fn gate(&self, literal: Literal) -> Option<Gate> {
        self.gate_for_no(literal.get_gate_no()?)
    }

    /// Get the [`Gate`] for `gate_no`
    ///
    /// Returns [`None`] iff `literal` refers to a gate that is not present in
    /// this circuit.
    ///
    /// See also: [`Self::gate()`]
    pub fn gate_for_no(&self, gate_no: Var) -> Option<Gate> {
        if let Some((kind, inputs)) = self.gates.get(gate_no) {
            Some(Gate {
                kind: kind.into(),
                inputs,
            })
        } else {
            None
        }
    }

    /// Get a mutable reference to the gate inputs
    ///
    /// Returns [`None`] iff `literal` refers to a constant, a circuit input, or
    /// a gate that is not present in this circuit.
    #[inline]
    pub fn gate_inputs_mut(&mut self, literal: Literal) -> Option<&mut [Literal]> {
        self.gates.get_mut(literal.get_gate_no()?)
    }

    /// Get a mutable reference to the gate inputs
    ///
    /// Returns [`None`] iff `literal` refers to a constant, a circuit input, or
    /// a gate that is not present in this circuit.
    #[inline(always)]
    pub fn gate_inputs_mut_for_no(&mut self, gate_no: Var) -> Option<&mut [Literal]> {
        self.gates.get_mut(gate_no)
    }

    /// Set the kind of the gate referenced by `literal`
    ///
    /// Panics if `literal` does not refer to a gate in this circuit.
    #[track_caller]
    #[inline]
    pub fn set_gate_kind(&mut self, literal: Literal, kind: GateKind) {
        self.gates.set_metadata(
            literal
                .get_gate_no()
                .expect("`literal` must refer to a gate"),
            kind as usize,
        );
    }

    /// Set the kind of the gate referenced by `gate_no`
    ///
    /// Panics if `literal` does not refer to a gate in this circuit.
    #[track_caller]
    #[inline(always)]
    pub fn set_gate_kind_for_no(&mut self, gate_no: Var, kind: GateKind) {
        self.gates.set_metadata(gate_no, kind as usize);
    }

    /// Set the kind of the gate referenced by `literal`
    ///
    /// Panics if the circuit is empty.
    #[track_caller]
    #[inline]
    pub fn set_last_gate_kind(&mut self, kind: GateKind) {
        let n = self.gates.len();
        if n == 0 {
            panic!("there are no gates in the circuit");
        }
        self.gates.set_metadata(n - 1, kind as usize);
    }

    /// Get the first gate
    ///
    /// Returns [`None`] iff `literal` refers to a constant, a circuit input, or
    /// a gate that is not present in this circuit.
    pub fn first_gate(&self) -> Option<Gate> {
        let (kind, inputs) = self.gates.first()?;
        Some(Gate {
            kind: kind.into(),
            inputs,
        })
    }

    /// Get the last gate
    ///
    /// Returns [`None`] iff `literal` refers to a constant, a circuit input, or
    /// a gate that is not present in this circuit.
    pub fn last_gate(&self) -> Option<Gate> {
        let (kind, inputs) = self.gates.last()?;
        Some(Gate {
            kind: kind.into(),
            inputs,
        })
    }

    /// Create a new empty gate at the end of the sequence
    pub fn push_gate(&mut self, kind: GateKind) -> Literal {
        let l = Literal::from_gate(false, self.gates.len());
        self.gates.push_vec(kind as _);
        l
    }

    /// Remove the last gate (if there is one)
    ///
    /// Returns true iff the number of gates was positive before removal.
    #[inline(always)]
    pub fn pop_gate(&mut self) -> bool {
        self.gates.pop_vec()
    }

    /// Add an input to the last gate
    ///
    /// There must be already one gate in the circuit, otherwise this method
    /// triggers a debug assertion or does not do anything, respectively.
    #[inline(always)]
    #[track_caller]
    pub fn push_gate_input(&mut self, literal: Literal) {
        self.gates.push_element(literal);
    }

    /// Add inputs to the last gate
    ///
    /// There must be already one gate in the circuit, otherwise this method
    /// triggers a debug assertion or does not do anything, respectively.
    #[inline(always)]
    #[track_caller]
    pub fn push_gate_inputs(&mut self, literals: impl IntoIterator<Item = Literal>) {
        self.gates.push_elements(literals);
    }

    /// Iterate over all gates in the circuit
    ///
    /// The iteration order is from first to last in the gate sequence.
    #[inline(always)]
    pub fn iter_gates(&self) -> CircuitGateIter {
        CircuitGateIter(self.gates.iter())
    }

    /// Retain the gates for which `predicate` returns `true`
    ///
    /// Note that removing one gate changes the gate numbers of all subsequent
    /// gates, but this method does not automatically update the gate inputs.
    #[inline]
    pub fn retain_gates(&mut self, mut predicate: impl FnMut(&mut [Literal]) -> bool) {
        self.gates.retain(move |_, inputs| predicate(inputs))
    }

    /// Remove all gate definitions from the circuit
    #[inline(always)]
    pub fn clear_gates(&mut self) {
        self.gates.clear();
    }

    /// Check if this circuit is acyclic
    ///
    /// Returns [`None`] if acyclic, or [`Some(literal)`][Some] if `literal`
    /// depends on itself.
    pub fn find_cycle(&self) -> Option<Literal> {
        let mut visited = BitVec::repeat(false, self.gates.len() * 2);

        fn inner(gates: &GateVec2d, visited: &mut BitSlice, index: usize) -> bool {
            if visited[index * 2 + 1] {
                return false; // finished
            }
            if visited[index * 2] {
                return true; // discovered -> cycle
            }
            visited.set(index * 2, true); // discovered

            for &l in gates.get(index).unwrap().1 {
                if l.is_gate() && inner(gates, visited, l.0 >> Literal::VAR_LSB) {
                    return true;
                }
            }

            visited.set(index * 2 + 1, true); // finished
            false
        }

        for index in 0..self.gates.len() {
            if inner(&self.gates, visited.as_mut_bitslice(), index) {
                return Some(Literal::from_gate(false, index));
            }
        }
        None
    }

    /// Simplify the circuit such that
    ///
    /// 1. no gate has constant inputs
    /// 2. no inputs of XOR gates are negated (only the output)
    /// 3. for every gate, all its inputs are distinct (disregarding polarities)
    /// 4. all gates have at least two inputs
    /// 5. there are no two structurally equivalent gates (same kind and inputs,
    ///    disregarding the input order)
    ///
    /// Additionally, the new circuit will only contain gates reachable from
    /// `roots`. The order of gate inputs in the resulting circuit follows the
    /// order of the first discovered equivalent gate (with duplicates etc.
    /// removed).
    ///
    /// If the reachable circuit fragment contains a cycle, this method returns
    /// `Err(l)`, where `l` represents a gate that is part of the cycle.
    /// Likewise, if the reachable fragment depends on unknown inputs
    /// (`input_number >= self.inputs().len()`, including [`Literal::UNDEF`]),
    /// the return value is `Err(l)`, where `l` is the unknown literal.
    ///
    /// On success, the gates in the returned circuit are topologically sorted.
    /// The additional `Vec<Literal>` maps the gates of `self` to literals valid
    /// in the simplified circuit.
    ///
    /// Hint: [`Problem::simplify`] uses this method to simplify the circuit and
    /// also maps the
    pub fn simplify(
        &self,
        roots: impl IntoIterator<Item = Literal>,
    ) -> Result<(Self, Vec<Literal>), Literal> {
        const DISCOVERED: Literal = Literal::UNDEF.negative();

        let bump = bumpalo::Bump::new();
        let mut gate_map = Vec::new();
        gate_map.resize(self.gates.len(), Literal::UNDEF);
        let mut input_set = BitVec::repeat(false, 2 * (self.gates.len() + self.inputs.len()));
        let mut new_gates: GateVec2d =
            GateVec2d::with_capacity(self.gates.len(), self.gates.all_elements().len());
        let mut unique_map = FxHashMap::default();
        unique_map.reserve(self.gates.len());

        fn inner<'a>(
            bump: &'a bumpalo::Bump,
            gates: &GateVec2d,
            index: usize,
            input_set: &mut BitSlice,
            unique_map: &mut FxHashMap<(GateKind, BumpBox<'a, [Literal]>), Literal>,
            new_gates: &mut GateVec2d,
            gate_map: &mut [Literal],
        ) -> Result<(), Literal> {
            if gate_map[index] == DISCOVERED {
                // discovered -> cycle
                return Err(Literal::from_gate(false, index));
            }
            if gate_map[index] != Literal::UNDEF {
                return Ok(()); // finished
            }
            gate_map[index] = DISCOVERED;

            let (meta, inputs) = gates.get(index).unwrap();
            let kind = GateKind::from(meta);

            for &l in inputs {
                if let Some(gate) = l.get_gate_no() {
                    inner(
                        bump, gates, gate, input_set, unique_map, new_gates, gate_map,
                    )?;
                }
            }

            // apply `gate_map` to the inputs and establish conditions 1+2
            let mut neg_out = false;
            let mut mapped = BumpVec::with_capacity_in(inputs.len(), bump);
            let known_inputs = input_set.len() - gates.len();
            match kind {
                GateKind::And | GateKind::Or => {
                    let (identity, dominator) = match kind {
                        GateKind::And => (Literal::TRUE, Literal::FALSE),
                        GateKind::Or => (Literal::FALSE, Literal::TRUE),
                        _ => unreachable!(),
                    };
                    for &l in inputs {
                        let l = l.get_gate_no().map_or(l, |i| gate_map[i] ^ l.is_negative());
                        if l.is_input() && l.get_input().unwrap() > known_inputs {
                            return Err(l);
                        }
                        if l == dominator {
                            gate_map[index] = dominator;
                            return Ok(());
                        }
                        if l != identity {
                            mapped.push(l);
                        }
                    }
                }
                GateKind::Xor => {
                    for &l in inputs {
                        neg_out ^= l.is_negative(); // negate for every flip
                        let l = if let Some(i) = l.get_gate_no() {
                            let l = gate_map[i];
                            neg_out ^= l.is_negative();
                            l.positive()
                        } else {
                            let l = l.positive();
                            debug_assert!(l != Literal::TRUE);
                            if l == Literal::FALSE {
                                continue; // x ⊕ ⊥ ≡ x
                            }
                            l
                        };
                        if l.is_input() && l.get_input().unwrap() > known_inputs {
                            return Err(l);
                        }
                        mapped.push(l);
                    }
                }
            };
            let mut inputs = mapped;

            // first part of condition 4
            if inputs.is_empty() {
                gate_map[index] = match kind {
                    GateKind::And => Literal::TRUE,
                    GateKind::Or => Literal::FALSE,
                    GateKind::Xor => Literal::TRUE ^ neg_out,
                };
                return Ok(());
            }

            // condition 3 with fast-path for exactly two different inputs
            if inputs.len() >= 3
                || (inputs.len() == 2 && inputs[0].negative() == inputs[1].negative())
            {
                let no_gate_add = 2 * gates.len() - 2; // -2 for constants
                let map = move |l: Literal| {
                    const { assert!(Literal::POLARITY_BIT == 0) };
                    let i = ((l.0 >> Literal::VAR_LSB) << 1) | (l.0 & (1 << Literal::POLARITY_BIT));
                    if l.is_gate() {
                        i
                    } else {
                        i + no_gate_add
                    }
                };

                match kind {
                    GateKind::And | GateKind::Or => {
                        for (i, &l) in inputs.iter().enumerate() {
                            if input_set[map(!l)] {
                                // found complement literal -> clear `input_set` and return ⊥/⊤
                                for &l in &inputs[..i] {
                                    input_set.set(map(l), false);
                                }
                                gate_map[index] = match kind {
                                    GateKind::And => Literal::FALSE, // x ∧ ¬x ≡ ⊥
                                    GateKind::Or => Literal::TRUE,   // x ∨ ¬x ≡ ⊤
                                    _ => unreachable!(),
                                };
                                return Ok(());
                            }
                            input_set.set(map(l), true);
                        }
                    }
                    GateKind::Xor => {
                        for &l in &inputs {
                            let i = map(l);
                            input_set.set(i, !input_set[i]);
                        }
                    }
                }

                inputs.retain(|&l| {
                    let i = map(l);
                    if input_set[i] {
                        input_set.set(i, false);
                        true
                    } else {
                        false
                    }
                });
            }
            // second part of condition 4
            if let [l] = &inputs[..] {
                gate_map[index] = *l ^ neg_out;
                return Ok(());
            }

            // save the children in the current order and sort
            let new_gate_no = new_gates.len();
            new_gates.push_vec(kind as usize);
            new_gates.push_elements(inputs.iter().copied());
            inputs.sort_unstable();

            let l = match unique_map.entry((kind, inputs.into_boxed_slice())) {
                Entry::Occupied(e) => {
                    new_gates.pop_vec();
                    *e.get()
                }
                Entry::Vacant(e) => *e.insert(Literal::from_gate(false, new_gate_no)),
            };
            gate_map[index] = l ^ neg_out;

            Ok(())
        }

        for root in roots {
            if let Some(i) = root.get_gate_no() {
                inner(
                    &bump,
                    &self.gates,
                    i,
                    &mut input_set,
                    &mut unique_map,
                    &mut new_gates,
                    &mut gate_map,
                )?;
            }
        }

        let new_circuit = Self {
            inputs: self.inputs.clone(),
            gates: new_gates,
        };
        Ok((new_circuit, gate_map))
    }
}

/// Iterator returned by [`Circuit::iter_gates()`]
#[derive(Clone)]
pub struct CircuitGateIter<'a>(
    vec2d::with_metadata::Vec2dIter<'a, Literal, { Circuit::GATE_METADATA_BITS }>,
);

impl<'a> Iterator for CircuitGateIter<'a> {
    type Item = Gate<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (kind, inputs) = self.0.next()?;
        Some(Gate {
            kind: kind.into(),
            inputs,
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for CircuitGateIter<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Debug for CircuitGateIter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl fmt::Debug for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circuit")
            .field("inputs", &self.inputs)
            .field("gates", &self.iter_gates())
            .finish()
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
pub struct AIGERDetails {
    /// Number of input variables
    inputs: usize,
    /// Latch inputs
    latches: Vec<Literal>,
    /// Latch initial values
    latch_init_values: TVBitVec,
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

    /// Mapping from AIGER variable numbers (literals divided by 2) to literals
    /// in the circuit
    map: Vec<Literal>,

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

impl AIGERDetails {
    /// Get the latch number of `literal`, if it refers to a latch
    ///
    /// If the return value is `Some(index)`, `index` is valid for the slice
    /// returned by [`Self::latches()`].
    #[inline]
    pub fn get_latch_no(&self, literal: Literal) -> Option<usize> {
        if literal.is_gate() {
            return None;
        }
        let first_latch = 1 + self.inputs;
        let result = (literal.0 >> Literal::VAR_LSB).checked_sub(first_latch)?;
        if result >= self.latches.len() {
            return None;
        }
        Some(result)
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

    /// Get the output definitions
    #[inline(always)]
    pub fn outputs(&self) -> &[Literal] {
        &self.outputs
    }

    /// Map the given AIGER literal to a [`Literal`] for use with the
    /// [`Circuit`]`
    ///
    /// Note that the literals in [`AIGERDetails`] are already mapped
    /// accordingly. This method is useful if you want to refer to specific
    /// gates or literals using the values
    #[inline(always)]
    pub fn map_aiger_literal(&self, literal: usize) -> Option<Literal> {
        let l = *self.map.get(literal >> 1)?;
        Some(if literal & 1 != 0 { !l } else { l })
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

    /// Map the literals in `self` based on `gate_map`
    ///
    /// See [`Literal::apply_gate_map()`] for more details.
    pub fn apply_gate_map(&self, gate_map: &[Literal]) -> Self {
        let map_slice = move |slice: &[Literal]| {
            slice
                .iter()
                .map(move |l| l.apply_gate_map(gate_map))
                .collect::<Vec<_>>()
        };

        let mut justice =
            Vec2d::with_capacity(self.justice.len(), self.justice.all_elements().len());
        for j in self.justice.iter() {
            justice.push_vec();
            justice.push_elements(j.iter().map(move |l| l.apply_gate_map(gate_map)));
        }

        Self {
            inputs: self.inputs,
            latches: map_slice(&self.latches),
            latch_init_values: self.latch_init_values.clone(),
            outputs: map_slice(&self.outputs),
            bad: map_slice(&self.bad),
            invariants: map_slice(&self.invariants),
            justice,
            fairness: map_slice(&self.fairness),
            map: map_slice(&self.map),
            output_names: self.output_names.clone(),
            bad_names: self.bad_names.clone(),
            invariant_names: self.invariant_names.clone(),
            justice_names: self.justice_names.clone(),
            fairness_names: self.fairness_names.clone(),
        }
    }

    /// In-place version of [`Self::apply_gate_map()`]
    pub fn apply_gate_map_in_place(&mut self, map: &[Literal]) {
        let map_slice = move |slice: &mut [Literal]| {
            for l in slice.iter_mut() {
                *l = l.apply_gate_map(map);
            }
        };

        map_slice(&mut self.latches);
        map_slice(&mut self.outputs);
        map_slice(&mut self.bad);
        map_slice(&mut self.invariants);
        map_slice(self.justice.all_elements_mut());
        map_slice(&mut self.fairness);
        map_slice(&mut self.map);
    }
}

/// Options for the parsers
#[non_exhaustive]
#[derive(Clone, Builder, Default, Debug)]
pub struct ParseOptions {
    /// Whether to parse a variable orders
    ///
    /// The [DIMACS satisfiability formats][dimacs], for instance, do not
    /// natively support specifying a variable order, however it is not uncommon
    /// to use the comment lines for this purpose. But while some files may
    /// contain a variable order in the comment lines, others may use them for
    /// arbitrary comments. Hence, it may be desired to turn on parsing orders
    /// for some files and turn it off for other files.
    #[builder(default = "false")]
    pub var_order: bool,

    /// Whether to parse a clause tree (for [DIMACS CNF][dimacs])
    ///
    /// If the CNF contains, e.g., five clauses, and one of the comment lines
    /// contains `co [[0, 1], [2, 3, 4]]`, then the parsed circuit will have
    /// three AND gates (one with clauses 0 and 1 as input, one with clauses
    /// 2, 3, and 4, as well as one with those two conjuncts).
    #[builder(default = "false")]
    pub clause_tree: bool,

    /// Whether to check that the circuit is acyclic
    ///
    /// The time complexity of this check is linear in the circuit's size, and
    /// therefore rather inexpensive. Therefore, it is enabled by default.
    #[builder(default = "true")]
    pub check_acyclic: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::*;

    #[test]
    fn simplify() {
        let mut circuit = Circuit::new(VarSet::new(3));
        circuit.push_gate(GateKind::Xor);
        circuit.push_gate_inputs([!v(0), v(1), v(2)]);
        circuit.push_gate(GateKind::And);
        circuit.push_gate_input(g(0));

        let (simplified, map) = circuit.simplify([Literal::from_gate(false, 1)]).unwrap();
        assert_eq!(simplified.num_gates(), 1);
        assert_eq!(
            simplified.gate_for_no(0),
            Some(Gate::xor(&[v(0), v(1), v(2)]))
        );
        assert_eq!(&map, &[!g(0), !g(0)]);
    }
}
