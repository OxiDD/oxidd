//! Binary decision diagrams (BDDs, without complement edges)

use std::hash::BuildHasherDefault;
use std::path::PathBuf;

use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyNone, PyRange, PyString, PyTuple, PyType};
use rustc_hash::FxHasher;

use oxidd::util::{num::F64, AllocResult, OptBool};
use oxidd::{
    BooleanFunction, BooleanFunctionQuant, Function, FunctionSubst, HasLevel, LevelNo, Manager,
    ManagerRef, Subst, VarNo,
};

use crate::util::DDMemoryError;

/// Manager for binary decision diagrams (without complement edges).
///
/// Implements: :class:`~oxidd.protocols.BooleanFunctionManager`\
/// [:class:`BDDFunction`]
#[pyclass(frozen, eq, hash, module = "oxidd.bdd")]
#[derive(PartialEq, Eq, Hash)]
pub struct BDDManager(oxidd::bdd::BDDManagerRef);

#[pymethods]
impl BDDManager {
    /// Create a new manager.
    ///
    /// Args:
    ///     inner_node_capacity (int): Maximum count of inner nodes
    ///     apply_cache_capacity (int): Maximum count of apply cache entries
    ///     threads (int): Worker thread count for the internal thread pool
    ///
    /// Returns:
    ///     BDDManager: The new manager
    #[new]
    fn new(inner_node_capacity: usize, apply_cache_capacity: usize, threads: u32) -> Self {
        Self(oxidd::bdd::new_manager(
            inner_node_capacity,
            apply_cache_capacity,
            threads,
        ))
    }

    /// Get the count of inner nodes.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The number of inner nodes stored in this manager
    fn num_inner_nodes(&self) -> usize {
        self.0
            .with_manager_shared(|manager| manager.num_inner_nodes())
    }

    /// Get an approximate count of inner nodes.
    ///
    /// For concurrent implementations, it may be much less costly to determine
    /// an approximation of the inner node count than an accurate count
    /// (:meth:`num_inner_nodes`).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: An approximate count of inner nodes stored in this manager
    fn approx_num_inner_nodes(&self) -> usize {
        self.0
            .with_manager_shared(|manager| manager.approx_num_inner_nodes())
    }

    /// Get the number of variables in this manager.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The number of variables
    fn num_vars(&self) -> VarNo {
        self.0.with_manager_shared(|manager| manager.num_vars())
    }

    /// Get the number of named variables in this manager.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The number of named variables
    fn num_named_vars(&self) -> VarNo {
        self.0
            .with_manager_shared(|manager| manager.num_named_vars())
    }

    /// Add ``additional`` unnamed variables to the decision diagram.
    ///
    /// The new variables are added at the bottom of the variable order. More
    /// precisely, the level number equals the variable number for each new
    /// variable.
    ///
    /// Note that some algorithms may assume that the domain of a function
    /// represented by a decision diagram is just the set of all variables. In
    /// this regard, adding variables can change the semantics of decision
    /// diagram nodes.
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Args:
    ///     additional (int): Count of variables to add
    ///
    /// Returns:
    ///     range: The new variable numbers
    fn add_vars<'py>(&self, py: Python<'py>, additional: VarNo) -> PyResult<Bound<'py, PyRange>> {
        let vars = self.0.with_manager_exclusive(|manager| {
            manager.num_vars().checked_add(additional)?;
            Some(manager.add_vars(additional))
        });
        if let Some(vars) = vars {
            Ok(PyRange::new(py, vars.start as _, vars.end as _)?)
        } else {
            Err(pyo3::exceptions::PyOverflowError::new_err(
                "too many variables",
            ))
        }
    }

    /// Add named variables to the decision diagram.
    ///
    /// This is a shorthand for :meth:`add_vars` and respective
    /// :meth:`set_var_name` calls. More details can be found there.
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Args:
    ///     names (Iterable[str]): Names of the new variables
    ///
    /// Returns:
    ///     range: The new variable numbers
    ///
    /// Raises:
    ///     ValueError: If a variable name occurs twice in ``names``. The
    ///         exception's argument is a :class:`DuplicateVarName`.
    fn add_named_vars<'py>(
        &self,
        py: Python<'py>,
        names: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyRange>> {
        crate::util::add_named_vars(&self.0, py, names)
    }

    /// Get ``var``'s name.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int): The variable number
    ///
    /// Returns:
    ///     str: The name, or an empty string for unnamed variables
    ///
    /// Raises:
    ///     IndexError: If ``var >= self.num_vars()``
    fn var_name<'py>(&self, py: Python<'py>, var: VarNo) -> PyResult<Bound<'py, PyString>> {
        self.0.with_manager_shared(|manager| {
            crate::util::var_no_bounds_check(manager, var)?;
            Ok(PyString::new(py, manager.var_name(var)))
        })
    }

    /// Label ``var`` as ``name``.
    ///
    /// An empty name means that the variable will become unnamed, and cannot be
    /// retrieved via :meth:`name_to_var` anymore.
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Args:
    ///     var (int): The variable number
    ///     name (str): The new variable name
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     ValueError: If ``name`` is not unique (and not ``""``). The
    ///         exception's argument is a :class:`DuplicateVarName`.
    ///     IndexError: If ``var >= self.num_vars()``
    fn set_var_name(&self, var: VarNo, name: &str) -> PyResult<()> {
        self.0.with_manager_exclusive(|manager| {
            crate::util::var_no_bounds_check(manager, var)?;
            if let Err(err) = manager.set_var_name(var, name) {
                let err: crate::util::DuplicateVarName = err.into();
                Err(pyo3::exceptions::PyValueError::new_err(err))
            } else {
                Ok(())
            }
        })
    }

    /// Get the variable number for the given variable name, if present.
    ///
    /// Note that you cannot retrieve unnamed variables.
    /// ``manager.name_to_var("")`` always returns ``None``.
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Args:
    ///     name (str): The variable name
    ///
    /// Returns:
    ///     int | None: The variable number if found, or ``None``
    fn name_to_var(&self, name: &str) -> Option<VarNo> {
        self.0
            .with_manager_shared(|manager| manager.name_to_var(name))
    }

    /// Get the level for the given variable.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int): The variable number
    ///
    /// Returns:
    ///     int: The level number
    ///
    /// Raises:
    ///     IndexError: If ``var >= self.num_vars()``
    fn var_to_level(&self, var: VarNo) -> PyResult<LevelNo> {
        self.0.with_manager_shared(|manager| {
            crate::util::var_no_bounds_check(manager, var)?;
            Ok(manager.var_to_level(var))
        })
    }

    /// Get the variable for the given level.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     level (int): The level number
    ///
    /// Returns:
    ///     int: The variable number
    ///
    /// Raises:
    ///     IndexError: If ``var >= self.num_vars()``
    fn level_to_var(&self, level: LevelNo) -> PyResult<LevelNo> {
        self.0.with_manager_shared(|manager| {
            crate::util::level_no_bounds_check(manager, level)?;
            Ok(manager.level_to_var(level))
        })
    }

    /// Get the Boolean function that is true if and only if `var` is true.
    ///
    /// Acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int): The variable number.
    ///
    /// Returns:
    ///     BDDFunction: A Boolean function that is true if and only if the
    ///     variable is true
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///     IndexError: If ``var >= self.num_vars()``
    fn var(&self, var: VarNo) -> PyResult<BDDFunction> {
        self.0.with_manager_shared(|manager| {
            crate::util::var_no_bounds_check(manager, var)?;
            oxidd::bdd::BDDFunction::var(manager, var).try_into()
        })
    }

    /// Get the Boolean function that is true if and only if `var` is false.
    ///
    /// Acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int): The variable number.
    ///
    /// Returns:
    ///     BDDFunction: A Boolean function that is true if and only if the
    ///     variable is false
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///     IndexError: If ``var >= self.num_vars()``
    fn not_var(&self, var: VarNo) -> PyResult<BDDFunction> {
        self.0.with_manager_shared(|manager| {
            crate::util::var_no_bounds_check(manager, var)?;
            oxidd::bdd::BDDFunction::not_var(manager, var).try_into()
        })
    }

    /// Get the constant true Boolean function ``⊤``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     BDDFunction: The constant true Boolean function ``⊤``
    fn r#true(&self) -> BDDFunction {
        self.0
            .with_manager_shared(|manager| BDDFunction(oxidd::bdd::BDDFunction::t(manager)))
    }

    /// Get the constant false Boolean function ``⊥``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     BDDFunction: The constant false Boolean function ``⊥``
    fn r#false(&self) -> BDDFunction {
        self.0
            .with_manager_shared(|manager| BDDFunction(oxidd::bdd::BDDFunction::f(manager)))
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
    ///     functions (Iterable[tuple[BDDFunction, str]]): Optional names for
    ///         BDD functions
    ///
    /// Returns:
    ///     None
    #[pyo3(
        signature = (/, path, functions=None),
        text_signature = "($self, /, path, functions=[])"
    )]
    fn dump_all_dot_file<'py>(
        &self,
        path: PathBuf,
        functions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let collect = crate::util::collect_func_str_pairs::<oxidd::bdd::BDDFunction, BDDFunction>;
        let functions = collect(functions)?;

        let file = std::fs::File::create(path)?;

        self.0.with_manager_shared(|manager| {
            oxidd_dump::dot::dump_all(
                file,
                manager,
                functions.iter().map(|(f, s)| (f, s.to_string_lossy())),
            )
        })?;

        Ok(())
    }
}

/// Substitution mapping variables to replacement functions.
///
/// Implements: :class:`~oxidd.protocols.FunctionSubst`
#[pyclass(frozen, module = "oxidd.bdd")]
pub struct BDDSubstitution(Subst<oxidd::bdd::BDDFunction>);

#[pymethods]
impl BDDSubstitution {
    /// Create a new substitution object for BDDs.
    ///
    /// See :meth:`BDDFunction.make_substitution` for more details.
    ///
    /// Args:
    ///     pairs (Iterable[tuple[int, BDDFunction]]):
    ///         ``(variable, replacement)`` pairs, where all variables are
    ///         distinct. The order of the pairs is irrelevant.
    ///
    /// Returns:
    ///     BDDSubstitution: The new substitution
    #[new]
    fn new(pairs: &Bound<PyAny>) -> PyResult<Self> {
        let len = pairs.len().unwrap_or(0);
        let mut vars = Vec::with_capacity(len);
        let mut replacements = Vec::with_capacity(len);
        for pair in pairs.try_iter()? {
            let pair: Bound<PyTuple> = pair?.downcast_into()?;
            let v = pair.get_borrowed_item(0)?;
            let r = pair.get_borrowed_item(1)?;
            vars.push(v.extract()?);
            replacements.push(r.downcast::<BDDFunction>()?.get().0.clone());
        }

        Ok(Self(Subst::new(vars, replacements)))
    }
}

/// Boolean function represented as a simple binary decision diagram (BDD).
///
/// Implements:
///     :class:`~oxidd.protocols.BooleanFunctionQuant`,
///     :class:`~oxidd.protocols.FunctionSubst`\ [:class:`BDDSubstitution`],
///     :class:`~oxidd.protocols.HasLevel`
///
/// All operations constructing BDDs may throw a
/// :exc:`~oxidd.util.DDMemoryError` in case they run out of memory.
///
/// Note that comparisons like ``f <= g`` are based on an arbitrary total order
/// and not related to logical implications. See the
/// :meth:`Function <oxidd.protocols.Function.__lt__>` protocol for more
/// details.
#[pyclass(frozen, eq, ord, hash, module = "oxidd.bdd")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BDDFunction(oxidd::bdd::BDDFunction);

impl TryFrom<AllocResult<oxidd::bdd::BDDFunction>> for BDDFunction {
    type Error = PyErr;

    fn try_from(value: AllocResult<oxidd::bdd::BDDFunction>) -> Result<Self, Self::Error> {
        match value {
            Ok(f) => Ok(Self(f)),
            Err(_) => Err(DDMemoryError::new_err(
                "OxiDD BDD operation ran out of memory",
            )),
        }
    }
}

impl AsRef<oxidd::bdd::BDDFunction> for BDDFunction {
    fn as_ref(&self) -> &oxidd::bdd::BDDFunction {
        &self.0
    }
}

#[pymethods]
impl BDDFunction {
    /// BDDManager: The associated manager.
    #[getter]
    fn manager(&self) -> BDDManager {
        BDDManager(self.0.manager_ref())
    }

    /// Get the cofactors ``(f_true, f_false)`` of ``self``.
    ///
    /// Let f(x₀, …, xₙ) be represented by ``self``, where x₀ is (currently) the
    /// top-most variable. Then f\ :sub:`true`\ (x₁, …, xₙ) = f(⊤, x₁, …, xₙ)
    /// and f\ :sub:`false`\ (x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
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
    ///     ``None`` if ``self`` references a terminal node.
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
    ///     references a terminal node.
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
    ///     references a terminal node.
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
    fn node_level(&self) -> Option<LevelNo> {
        self.0
            .with_manager_shared(|manager, edge| match manager.get_node(edge) {
                oxidd::Node::Inner(n) => Some(n.level()),
                oxidd::Node::Terminal(_) => None,
            })
    }
    /// Deprecated alias for :meth:`node_level`.
    ///
    /// Returns:
    ///     int | None: The level, or ``None`` if the node is a terminal
    ///
    /// .. deprecated:: 0.11
    ///    Use :meth:`node_level` instead
    fn level(&self) -> Option<LevelNo> {
        self.node_level()
    }
    /// Get the variable number for the underlying node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Time complexity: O(1)
    ///
    /// Returns:
    ///     int | None: The variable number, or ``None`` if the node is a
    ///     terminal
    fn node_var(&self) -> Option<VarNo> {
        self.0
            .with_manager_shared(|manager, edge| match manager.get_node(edge) {
                oxidd::Node::Inner(n) => Some(manager.level_to_var(n.level())),
                oxidd::Node::Terminal(_) => None,
            })
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

    /// Compute the BDD for the conditional ``t if self else e``.
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

    /// Create a new substitution object from pairs ``(var, replacement)``.
    ///
    /// The intent behind substitution objects is to optimize the case where the
    /// same substitution is applied multiple times. We would like to re-use
    /// apply cache entries across these operations, and therefore, we need a
    /// compact identifier for the substitution. This identifier is provided by
    /// the returned substitution object.
    ///
    /// Args:
    ///     pairs (Iterable[tuple[int, Self]]): ``(variable, replacement)``
    ///         pairs, where all variables are distinct. The order of the pairs
    ///         is irrelevant.
    ///
    /// Returns:
    ///     BDDSubstitution: The substitution to be used with :meth:`substitute`
    #[classmethod]
    #[pyo3(signature = (pairs, /))]
    fn make_substitution(_cls: &Bound<PyType>, pairs: &Bound<PyAny>) -> PyResult<BDDSubstitution> {
        BDDSubstitution::new(pairs)
    }

    /// Substitute variables in ``self`` according to ``substitution``.
    ///
    /// The substitution is performed in a parallel fashion, e.g.:
    /// ``(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y``
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     substitution (BDDSubstitution): A substitution object created using
    ///         :meth:`make_substitution`. All contained DD functions must
    ///         belong to the same manager as ``self``.
    ///
    /// Returns:
    ///     Self: ``self`` with variables substituted
    #[pyo3(signature = (substitution, /))]
    fn substitute(&self, py: Python, substitution: &BDDSubstitution) -> PyResult<Self> {
        py.allow_threads(move || self.0.substitute(&substitution.0))
            .try_into()
    }

    /// Compute the universal quantification over ``vars``.
    ///
    /// This operation removes all occurrences of variables in ``vars`` by
    /// universal quantification. Universal quantification ∀x. f(…, x, …) of a
    /// Boolean function f(…, x, …) over a single variable x is
    /// f(…, 0, …) ∧ f(…, 1, …).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     vars (Self): Set of variables represented as conjunction thereof.
    ///         Must belong to the same manager as ``self``.
    ///
    /// Returns:
    ///     Self: ∀ vars: self
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn forall(&self, py: Python, vars: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.forall(&vars.0)).try_into()
    }
    /// Compute the existential quantification over ``vars``.
    ///
    /// This operation removes all occurrences of variables in ``vars`` by
    /// existential quantification. Existential quantification ∃x. f(…, x, …) of
    /// a Boolean function f(…, x, …) over a single variable x is
    /// f(…, 0, …) ∨ f(…, 1, …).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     vars (Self): Set of variables represented as conjunction thereof.
    ///         Must belong to the same manager as ``self``.
    ///
    /// Returns:
    ///     Self: ∃ vars: self
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn exists(&self, py: Python, vars: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.exists(&vars.0)).try_into()
    }
    /// Deprecated alias for :meth:`exists`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     vars (Self): Set of variables represented as conjunction thereof.
    ///         Must belong to the same manager as ``self``.
    ///
    /// Returns:
    ///     Self: ∃ vars: self
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///
    /// .. deprecated:: 0.10
    ///    Use :meth:`exists` instead
    fn exist(&self, py: Python, vars: &Self) -> PyResult<Self> {
        self.exists(py, vars)
    }
    /// Compute the unique quantification over ``vars``.
    ///
    /// This operation removes all occurrences of variables in ``vars`` by
    /// unique quantification. Unique quantification ∃!x. f(…, x, …) of a
    /// Boolean function f(…, x, …) over a single variable x is
    /// f(…, 0, …) ⊕ f(…, 1, …). Unique quantification is also known as the
    /// `Boolean difference <https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors>`_ or
    /// `Boolean derivative <https://en.wikipedia.org/wiki/Boolean_differential_calculus>`_.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     vars (Self): Set of variables represented as conjunction thereof.
    ///         Must belong to the same manager as ``self``.
    ///
    /// Returns:
    ///     Self: ∃! vars: self
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn unique(&self, py: Python, vars: &Self) -> PyResult<Self> {
        py.allow_threads(move || self.0.unique(&vars.0)).try_into()
    }

    /// Combined application of ``op`` and :meth:`forall`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     op (BooleanOperator): Binary Boolean operator to apply to ``self``
    ///         and ``rhs``
    ///     rhs (Self): Right-hand side of the operator. Must belong to the same
    ///         manager as ``self``.
    ///     vars (Self): Set of variables to quantify over. Represented as
    ///         conjunction of variables. Must belong to the same manager as
    ///         ``self``.
    ///
    /// Returns:
    ///     Self: ``∀ vars. self <op> rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn apply_forall(
        &self,
        py: Python,
        op: &Bound<PyAny>,
        rhs: &Self,
        vars: &Self,
    ) -> PyResult<Self> {
        let op = crate::util::boolean_operator(op)?;
        py.allow_threads(move || self.0.apply_forall(op, &rhs.0, &vars.0))
            .try_into()
    }
    /// Combined application of ``op`` and :meth:`exists`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     op (BooleanOperator): Binary Boolean operator to apply to ``self``
    ///         and ``rhs``
    ///     rhs (Self): Right-hand side of the operator. Must belong to the same
    ///         manager as ``self``.
    ///     vars (Self): Set of variables to quantify over. Represented as
    ///         conjunction of variables. Must belong to the same manager as
    ///         ``self``.
    ///
    /// Returns:
    ///     Self: ``∃ vars. self <op> rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn apply_exists(
        &self,
        py: Python,
        op: &Bound<PyAny>,
        rhs: &Self,
        vars: &Self,
    ) -> PyResult<Self> {
        let op = crate::util::boolean_operator(op)?;
        py.allow_threads(move || self.0.apply_exists(op, &rhs.0, &vars.0))
            .try_into()
    }
    /// Deprecated alias for ``apply_exists()``.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     op (BooleanOperator): Binary Boolean operator to apply to ``self``
    ///         and ``rhs``
    ///     rhs (Self): Right-hand side of the operator. Must belong to the same
    ///         manager as ``self``.
    ///     vars (Self): Set of variables to quantify over. Represented as
    ///         conjunction of variables. Must belong to the same manager as
    ///         ``self``.
    ///
    /// Returns:
    ///     Self: ``∃ vars. self <op> rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///
    /// .. deprecated:: 0.10
    ///    Use :meth:`apply_exists` instead
    fn apply_exist(
        &self,
        py: Python,
        op: &Bound<PyAny>,
        rhs: &Self,
        vars: &Self,
    ) -> PyResult<Self> {
        self.apply_exists(py, op, rhs, vars)
    }
    /// Combined application of ``op`` and :meth:`unique`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     op (BooleanOperator): Binary Boolean operator to apply to ``self``
    ///         and ``rhs``
    ///     rhs (Self): Right-hand side of the operator. Must belong to the same
    ///         manager as ``self``.
    ///     vars (Self): Set of variables to quantify over. Represented as
    ///         conjunction of variables. Must belong to the same manager as
    ///         ``self``.
    ///
    /// Returns:
    ///     Self: ``∃! vars. self <op> rhs``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn apply_unique(
        &self,
        py: Python,
        op: &Bound<PyAny>,
        rhs: &Self,
        vars: &Self,
    ) -> PyResult<Self> {
        let op = crate::util::boolean_operator(op)?;
        py.allow_threads(move || self.0.apply_unique(op, &rhs.0, &vars.0))
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
    ///     assignment
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
        match py.allow_threads(move || self.0.pick_cube(move |_, _, _| false)) {
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
    ///     args (Iterable[tuple[int, bool]]): ``(variable, value)`` pairs that
    ///         determine the valuation for all variables in the function's
    ///         domain. The order is irrelevant (except that if the valuation
    ///         for a variable is given multiple times, the last value counts).
    ///         Should there be a decision node for a variable not part of the
    ///         domain, then ``False`` is used as the decision value.
    ///
    /// Returns:
    ///     bool: The result of applying the function ``self`` to ``args``
    ///
    /// Raises:
    ///     IndexError: If any variable number in ``args`` is greater or equal
    ///         to ``self.manager.num_vars()``
    fn eval(&self, args: &Bound<PyAny>) -> PyResult<bool> {
        crate::util::eval(&self.0, args)
    }
}
