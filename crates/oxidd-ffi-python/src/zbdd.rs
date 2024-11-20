//! Zero-suppressed binary decision diagrams (ZBDDs)

use std::hash::BuildHasherDefault;
use std::path::PathBuf;

use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyNone, PyTuple};
use rustc_hash::FxHasher;

use oxidd::util::{num::F64, AllocResult, OptBool};
use oxidd::{BooleanFunction, BooleanVecSet, Function, LevelNo, Manager, ManagerRef};

use crate::util::DDMemoryError;

/// Manager for zero-suppressed binary decision diagrams.
///
/// Implements: :class:`~oxidd.protocols.BooleanFunctionManager`\
/// [:class`ZBDDFunction`]
#[pyclass(frozen, eq, hash, module = "oxidd.zbdd")]
#[derive(PartialEq, Eq, Hash)]
pub struct ZBDDManager(oxidd::zbdd::ZBDDManagerRef);

#[pymethods]
impl ZBDDManager {
    /// Create a new manager.
    ///
    /// Args:
    ///     inner_node_capacity (int): Maximum count of inner nodes
    ///     apply_cache_capacity (int): Maximum count of apply cache entries
    ///     threads (int): Worker thread count for the internal thread pool
    ///
    /// Returns:
    ///     ZBDDManager: The new manager
    #[new]
    fn new(inner_node_capacity: usize, apply_cache_capacity: usize, threads: u32) -> Self {
        Self(oxidd::zbdd::new_manager(
            inner_node_capacity,
            apply_cache_capacity,
            threads,
        ))
    }

    /// Get a fresh variable in the form of a singleton set.
    ///
    /// This adds a new level to a decision diagram. Note that if you interpret
    /// Boolean functions with respect to all variables, then the semantics
    /// change from f to f'(x₁, …, xₙ, xₙ₊₁) = f(x₁, …, xₙ) ∧ ¬xₙ₊₁. This is
    /// different compared to B(C)DDs where we have
    /// f'(x₁, …, xₙ, xₙ₊₁) = f(x₁, …, xₙ).
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Returns:
    ///     ZBDDFunction: The singleton set
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn new_singleton(&self) -> PyResult<ZBDDFunction> {
        self.0
            .with_manager_exclusive(|manager| oxidd::zbdd::ZBDDFunction::new_singleton(manager))
            .try_into()
    }

    /// Get a fresh variable, adding a new level to a decision diagram.
    ///
    /// Note that if you interpret Boolean functions with respect to all
    /// variables, then adding a level changes the semantics change from
    /// f to f'(x₁, …, xₙ, xₙ₊₁) = f(x₁, …, xₙ) ∧ ¬xₙ₊₁. This is different
    /// compared to B(C)DDs where we have f'(x₁, …, xₙ, xₙ₊₁) = f(x₁, …, xₙ).
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Returns:
    ///     ZBDDFunction: A Boolean function that is true if and only if the
    ///         variable is true
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn new_var(&self) -> PyResult<ZBDDFunction> {
        self.0
            .with_manager_exclusive(|manager| oxidd::zbdd::ZBDDFunction::new_var(manager))
            .try_into()
    }

    /// Get the ZBDD set ∅.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     ZBDDFunction: The set `∅` (or equivalently `⊥`)
    fn empty(&self) -> ZBDDFunction {
        self.0
            .with_manager_shared(|manager| ZBDDFunction(oxidd::zbdd::ZBDDFunction::empty(manager)))
    }

    /// Get the ZBDD set {∅}.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     ZBDDFunction: The set `{∅}`
    fn base(&self) -> ZBDDFunction {
        self.0
            .with_manager_shared(|manager| ZBDDFunction(oxidd::zbdd::ZBDDFunction::base(manager)))
    }

    /// Get the constant true Boolean function ``⊤``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     ZBDDFunction: The constant true Boolean function ``⊤``
    fn r#true(&self) -> ZBDDFunction {
        self.0
            .with_manager_shared(|manager| ZBDDFunction(oxidd::zbdd::ZBDDFunction::t(manager)))
    }

    /// Get the constant false Boolean function ``⊥``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     ZBDDFunction: The constant false Boolean function ``⊥``
    fn r#false(&self) -> ZBDDFunction {
        self.empty()
    }

    /// Get the number of inner nodes.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The number of inner nodes stored in this manager
    fn num_inner_nodes(&self) -> usize {
        self.0
            .with_manager_shared(|manager| manager.num_inner_nodes())
    }

    /// Dump the entire decision diagram in this manager as Graphviz DOT code.
    ///
    /// The output may also include nodes that are not reachable from
    /// ``functions``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     path (str | PathLike[str]): Path of the output file. If a file at
    ///         ``path`` exists, it will be truncated, otherwise a new one will
    ///         be created.
    ///     functions (Iterable[tuple[ZBDDFunction, str]]): Optional names for
    ///         ZBDD functions
    ///     variables (Iterable[tuple[ZBDDFunction, str]]): Optional names for
    ///         variables. The variables must be handles for the respective
    ///         decision diagram levels, i.e., singleton sets.
    ///
    /// Returns:
    ///     None
    #[pyo3(
        signature = (/, path, functions=None, variables=None),
        text_signature = "($self, /, path, functions=[], variables=[])"
    )]
    fn dump_all_dot_file<'py>(
        &self,
        path: PathBuf,
        functions: Option<&Bound<'py, PyAny>>,
        variables: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let collect =
            crate::util::collect_func_str_pairs::<oxidd::zbdd::ZBDDFunction, ZBDDFunction>;
        let functions = collect(functions)?;
        let variables = collect(variables)?;

        let file = std::fs::File::create(path)?;

        self.0.with_manager_shared(|manager| {
            oxidd_dump::dot::dump_all(
                file,
                manager,
                variables.iter().map(|(v, s)| (v, s.to_string_lossy())),
                functions.iter().map(|(f, s)| (f, s.to_string_lossy())),
            )
        })?;

        Ok(())
    }
}

/// Boolean function as zero-suppressed binary decision diagram (ZBDD).
///
/// Implements:
///     :class:`~oxidd.protocols.BooleanFunction`,
///     :class:`~oxidd.protocols.HasLevel`
///
/// All operations constructing ZBDDs may throw a
/// :exc:`~oxidd.util.DDMemoryError` in case they run out of memory.
///
/// Note that comparisons like ``f <= g`` are based on an arbitrary total order
/// and not related to logical implications. See the
/// :meth:`Function <oxidd.protocols.Function.__lt__>` protocol for more
/// details.
#[pyclass(frozen, eq, ord, hash, module = "oxidd.zbdd")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZBDDFunction(oxidd::zbdd::ZBDDFunction);

impl TryFrom<AllocResult<oxidd::zbdd::ZBDDFunction>> for ZBDDFunction {
    type Error = PyErr;

    fn try_from(value: AllocResult<oxidd::zbdd::ZBDDFunction>) -> Result<Self, Self::Error> {
        match value {
            Ok(f) => Ok(Self(f)),
            Err(_) => Err(DDMemoryError::new_err(
                "OxiDD ZBDD operation ran out of memory",
            )),
        }
    }
}

impl AsRef<oxidd::zbdd::ZBDDFunction> for ZBDDFunction {
    fn as_ref(&self) -> &oxidd::zbdd::ZBDDFunction {
        &self.0
    }
}

#[pymethods]
impl ZBDDFunction {
    /// ZBDDManager: The associated manager.
    #[getter]
    fn manager(&self) -> ZBDDManager {
        ZBDDManager(self.0.manager_ref())
    }

    /// Get the cofactors ``(f_true, f_false)`` of ``self``.
    ///
    /// Let f(x₀, …, xₙ) be represented by ``self``, where x₀ is (currently) the
    /// top-most variable. Then f\ :sub:`true`\ (x₁, …, xₙ) = f(⊤, x₁, …, xₙ)
    /// and f\ :sub:`false`\ (x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
    ///
    /// Note that the domain of f is 𝔹\ :sup:`n+1` while the domain of
    /// f\ :sub:`true` and f\ :sub:`false` is 𝔹\ :sup:`n`. This is irrelevant in
    /// case of BDDs and BCDDs, but not for ZBDDs: For instance, g(x₀) = x₀ and
    /// g'(x₀, x₁) = x₀ have the same representation as BDDs or BCDDs, but
    /// different representations as ZBDDs.
    ///
    /// Structurally, the cofactors are simply the children in case with edge
    /// tags adjusted accordingly.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     tuple[Self, Self] | None: The cofactors ``(f_true, f_false)``, or
    ///         ``None`` if ``self`` references a terminal node.
    ///
    /// See Also:
    ///     :meth:`cofactor_true`, :meth:`cofactor_false` if you only need one
    ///     of the cofactors.
    fn cofactors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.0.cofactors() {
            Some((ct, cf)) => {
                let ct = Self(ct).into_pyobject(py)?;
                let cf = Self(cf).into_pyobject(py)?;
                PyTuple::new(py, [ct, cf])?.into_any()
            }
            None => PyNone::get(py).to_owned().into_any(),
        })
    }

    /// Get the cofactor ``f_true`` of ``self``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     Self | None: The cofactor ``f_true``, or ``None`` if ``self``
    ///         references a terminal node.
    ///
    /// See Also:
    ///     :meth:`cofactors`, also for a more detailed description
    fn cofactor_true(&self) -> Option<Self> {
        Some(Self(self.0.cofactor_true()?))
    }
    /// Get the cofactor ``f_false`` of ``self``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     Self | None: The cofactor ``f_false``, or ``None`` if ``self``
    ///         references a terminal node.
    ///
    /// See Also:
    ///     :meth:`cofactors`, also for a more detailed description
    fn cofactor_false(&self) -> Option<Self> {
        Some(Self(self.0.cofactor_false()?))
    }

    /// Get the level of the underlying node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     int | None: The level, or ``None`` if the node is a terminal
    fn level(&self) -> Option<LevelNo> {
        match self
            .0
            .with_manager_shared(|manager, edge| manager.get_node(edge).level())
        {
            LevelNo::MAX => None,
            l => Some(l),
        }
    }

    /// Get the Boolean function v for the singleton set {v}.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     Self: The Boolean function `v` as ZBDD
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn var_boolean_function(&self, py: Python) -> PyResult<Self> {
        py.allow_threads(move || {
            self.0.with_manager_shared(move |manager, singleton| {
                let res = oxidd::zbdd::var_boolean_function(manager, singleton)?;
                Ok(oxidd::zbdd::ZBDDFunction::from_edge(manager, res))
            })
        })
        .try_into()
    }

    /// Compute the negation ``¬self``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     Self: ``¬self``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn __invert__(&self, py: Python) -> PyResult<Self> {
        py.allow_threads(move || self.0.not()).try_into()
    }
    /// Compute the conjunction ``self ∧ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ∧ rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn __and__(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.and(&rhs.0)).try_into()
    }
    /// Compute the disjunction ``self ∨ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ∨ rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn __or__(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.or(&rhs.0)).try_into()
    }
    /// Compute the exclusive disjunction ``self ⊕ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ⊕ rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn __xor__(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.xor(&rhs.0)).try_into()
    }
    /// Compute the set difference ``self ∖ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ∖ rhs``, or equivalently ``rhs.strict_imp(self)``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn __sub__(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        rhs.imp_strict(py, self)
    }
    /// Compute the negated conjunction ``self ⊼ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ⊼ rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    #[pyo3(signature = (rhs, /))]
    fn nand(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.nand(&rhs.0)).try_into()
    }
    /// Compute the negated disjunction ``self ⊽ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ⊽ rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    #[pyo3(signature = (rhs, /))]
    fn nor(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.nor(&rhs.0)).try_into()
    }
    /// Compute the equivalence ``self ↔ rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self ↔ rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    #[pyo3(signature = (rhs, /))]
    fn equiv(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.equiv(&rhs.0)).try_into()
    }
    /// Compute the implication ``self → rhs`` (or ``f ≤ g``).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self → rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    #[pyo3(signature = (rhs, /))]
    fn imp(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.imp(&rhs.0)).try_into()
    }
    /// Compute the strict implication ``self < rhs``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     rhs (Self): Right-hand side operand. Must belong to the same manager
    ///         as ``self``
    ///
    /// Returns:
    ///     Self: ``self < rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    #[pyo3(signature = (rhs, /))]
    fn imp_strict(&self, py: Python, rhs: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.imp_strict(&rhs.0))
            .try_into()
    }

    /// Compute the ZBDD for the conditional ``t if self else e``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     t (Self): Then-case; must belong to the same manager as ``self``
    ///     e (Self): Else-case; must belong to the same manager as ``self``
    ///
    /// Returns:
    ///     Self: The Boolean function ``f(v: 𝔹ⁿ) = t(v) if self(v) else e(v)``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn ite(&self, py: Python, t: &Self, e: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.ite(&t.0, &e.0)).try_into()
    }

    /// Create a node at ``self``'s level with edges ``hi`` and ``lo``.
    ///
    /// ``self`` must be a singleton set at a level above the top level of
    /// ``hi`` and ``lo``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     hi (Self): Edge for the case where the variable is true; must belong
    ///         to the same manager as ``self``
    ///     lo (Self): Edge for the case where the variable is false; must
    ///         belong to the same manager as ``self``
    ///
    /// Returns:
    ///     Self: The new ZBDD node
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn make_node(&self, hi: &Self, lo: &Self) -> PyResult<Self> {
        self.0
            .with_manager_shared(move |manager, var| {
                let hi = hi.0.as_edge(manager);
                let lo = lo.0.as_edge(manager);
                let hi = manager.clone_edge(hi);
                let lo = manager.clone_edge(lo);
                let res = oxidd::zbdd::make_node(manager, var, hi, lo)?;
                Ok(oxidd::zbdd::ZBDDFunction::from_edge(manager, res))
            })
            .try_into()
    }

    /// Get the number of descendant nodes.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The count of descendant nodes including the node referenced by
    ///     ``self`` and terminal nodes.
    fn node_count(&self, py: Python) -> usize {
        py.allow_threads(move || self.0.node_count())
    }

    /// Check for satisfiability.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     bool: Whether the Boolean function has at least one satisfying
    ///         assignment
    fn satisfiable(&self) -> bool {
        self.0.satisfiable()
    }
    /// Check for validity.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     bool: Whether all assignments satisfy the Boolean function
    fn valid(&self) -> bool {
        self.0.valid()
    }

    /// Count the number of satisfying assignments.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     vars (int): Assume that the function's domain has this many
    ///         variables.
    ///
    /// Returns:
    ///     int: The exact number of satisfying assignments
    fn sat_count(&self, py: Python, vars: LevelNo) -> BigUint {
        py.allow_threads(move || {
            self.0
                .sat_count::<BigUint, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        })
    }

    /// Count the number of satisfying assignments.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     vars (int): Assume that the function's domain has this many
    ///         variables.
    ///
    /// Returns:
    ///     float: (An approximation of) the number of satisfying assignments
    fn sat_count_float(&self, py: Python, vars: LevelNo) -> f64 {
        py.allow_threads(move || {
            self.0
                .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
                .0
        })
    }

    /// Pick a satisfying assignment.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     list[bool | None] | None: The satisfying assignment where the i-th
    ///     value means that the i-th variable is false, true, or "don't care,"
    ///     respectively, or ``None`` if ``self`` is unsatisfiable
    fn pick_cube<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match py.allow_threads(move || self.0.pick_cube([], move |_, _, _| false)) {
            Some(r) => {
                let iter = r.into_iter().map(|v| match v {
                    OptBool::None => PyNone::get(py).to_owned().into_any(),
                    _ => PyBool::new(py, v != OptBool::False).to_owned().into_any(),
                });
                Ok(PyList::new(py, iter)?.into_any())
            }
            None => Ok(PyNone::get(py).to_owned().into_any()),
        }
    }
    /// Pick a satisfying assignment, represented as decision diagram.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     Self: The satisfying assignment as decision diagram, or ``⊥`` if
    ///     ``self`` is unsatisfiable
    fn pick_cube_dd(&self, py: Python) -> PyResult<Self> {
        py.allow_threads(move || self.0.pick_cube_dd(move |_, _, _| false))
            .try_into()
    }
    /// Pick a satisfying assignment as DD, with choices as of ``literal_set``.
    ///
    /// ``literal_set`` is a conjunction of literals. Whenever there is a choice
    /// for a variable, it will be set to true if the variable has a positive
    /// occurrence in ``literal_set``, and set to false if it occurs negated in
    /// ``literal_set``. If the variable does not occur in ``literal_set``, then
    /// it will be left as don't care if possible, otherwise an arbitrary (not
    /// necessarily random) choice will be performed.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     literal_set (Self): Conjunction of literals to determine the choice
    ///         for variables
    ///
    /// Returns:
    ///     Self: The satisfying assignment as decision diagram, or ``⊥`` if
    ///     ``self`` is unsatisfiable
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn pick_cube_dd_set(&self, py: Python, literal_set: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.pick_cube_dd_set(&literal_set.0))
            .try_into()
    }

    /// Evaluate this Boolean function with arguments ``args``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     args (Iterable[tuple[Self, bool]]): ``(variable, value)`` pairs
    ///         where variables must be handles for the respective decision
    ///         diagram levels, i.e., singleton sets.
    ///         Missing variables are assumed to be false. However, note that
    ///         the arguments may also determine the domain, e.g., in case of
    ///         ZBDDs. If variables are given multiple times, the last value
    ///         counts. Besides that, the order is irrelevant.
    ///         All variable handles must belong to the same manager as ``self``
    ///         and must reference inner nodes.
    ///
    /// Returns:
    ///     bool: The result of applying the function ``self`` to ``args``
    fn eval(&self, args: &Bound<PyAny>) -> PyResult<bool> {
        let mut fs = Vec::with_capacity(args.len().unwrap_or(0));
        for pair in args.try_iter()? {
            let pair: Bound<PyTuple> = pair?.downcast_into()?;
            let f = pair.get_borrowed_item(0)?;
            let f = f.downcast::<Self>()?.get().0.clone();
            let b = pair.get_borrowed_item(1)?.is_truthy()?;
            fs.push((f, b));
        }

        Ok(self.0.eval(fs.iter().map(|(f, b)| (f, *b))))
    }
}
