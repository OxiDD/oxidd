//! Zero-suppressed binary decision diagrams (ZBDDs)

use std::hash::BuildHasherDefault;
use std::path::PathBuf;

use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyNone, PyRange, PyString, PyTuple};
use rustc_hash::FxHasher;

use oxidd::util::{num::F64, AllocResult, OptBool};
use oxidd::{
    BooleanFunction, BooleanVecSet, Function, HasLevel, LevelNo, Manager, ManagerRef, VarNo,
};

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
            crate::util::var_no_bounds_check(manager, level)?;
            Ok(manager.level_to_var(level))
        })
    }

    /// Perform garbage collection.
    ///
    /// This method looks for nodes that are neither referenced by a
    /// :class:`ZBDDFunction` nor another node and removes them. The method
    /// works from top to bottom, so if a node is only referenced by nodes
    /// that can be removed, this node will be removed as well.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The count of nodes removed
    fn gc(&self, py: Python) -> usize {
        py.detach(|| self.0.with_manager_shared(|manager| manager.gc()))
    }

    /// Get the count of garbage collections.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     int: The garbage collection count
    fn gc_count(&self) -> u64 {
        self.0.with_manager_shared(|manager| manager.gc_count())
    }

    /// Get the singleton set {var}.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int | str): The variable number or name
    ///
    /// Returns:
    ///     ZBDDFunction: The singleton set {var}
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///     IndexError: If ``var`` is an ``int`` and ``var >= self.num_vars()``
    ///     KeyError: If ``var`` is a string and
    ///         ``self.name_to_var(var) is None``
    fn singleton(&self, var: crate::util::VarId) -> PyResult<ZBDDFunction> {
        crate::util::with_var_no(&self.0, var, |manager, var| {
            oxidd::zbdd::ZBDDFunction::singleton(manager, var).try_into()
        })
    }

    /// Get the Boolean function that is true if and only if ``var`` is true.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int | str): The variable number or name
    ///
    /// Returns:
    ///     ZBDDFunction: A Boolean function that is true if and only if the
    ///     variable is true
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///     IndexError: If ``var`` is an ``int`` and ``var >= self.num_vars()``
    ///     KeyError: If ``var`` is a string and
    ///         ``self.name_to_var(var) is None``
    fn var(&self, var: crate::util::VarId) -> PyResult<ZBDDFunction> {
        crate::util::with_var_no(&self.0, var, |manager, var| {
            oxidd::zbdd::ZBDDFunction::var(manager, var).try_into()
        })
    }

    /// Get the Boolean function that is true if and only if ``var`` is false.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     var (int | str): The variable number or name
    ///
    /// Returns:
    ///     ZBDDFunction: A Boolean function that is true if and only if the
    ///     variable is false
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///     IndexError: If ``var`` is an ``int`` and ``var >= self.num_vars()``
    ///     KeyError: If ``var`` is a string and
    ///         ``self.name_to_var(var) is None``
    fn not_var(&self, var: crate::util::VarId) -> PyResult<ZBDDFunction> {
        crate::util::with_var_no(&self.0, var, |manager, var| {
            oxidd::zbdd::ZBDDFunction::not_var(manager, var).try_into()
        })
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

    /// Reorder the variables according to ``order``.
    ///
    /// If a variable ``x`` occurs before variable ``y`` in ``order``, then
    /// ``x`` will be above ``y`` in the decision diagram when this function
    /// returns. Variables not mentioned in ``order`` will be placed in a
    /// position such that the least number of level swaps need to be
    /// performed.
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// Args:
    ///     order (Iterable[int]): The variable order to establish
    ///
    /// Returns:
    ///     None
    fn set_var_order(&self, py: Python, order: &Bound<PyAny>) -> PyResult<()> {
        let order: Vec<VarNo> = crate::util::collect_vec(order)?;
        py.detach(|| {
            self.0
                .with_manager_exclusive(|manager| oxidd_reorder::set_var_order(manager, &order))
        });
        Ok(())
    }

    /// Import the decision diagram from the DDDMP ``file``.
    ///
    /// Note that the support variables must also be ordered by their current
    /// level (lower level numbers first). To this end, you can use
    /// :meth:`set_var_order` with ``support_vars`` (or
    /// :attr:`file.support_var_order
    /// <oxidd.util.DDDMPFile.support_var_order>`).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     file (DDDMPFile): The DDDMP file handle
    ///     support_vars (Iterable[int] | None): Optional mapping from support
    ///         variables of the DDDMP file to variable numbers in this manager.
    ///         By default, :attr:`file.support_var_order
    ///         <oxidd.util.DDDMPFile.support_var_order>` will be used.
    ///
    /// Returns:
    ///     list[ZBDDFunction]: The imported ZBDD functions
    #[pyo3(signature = (/, file, support_vars = None))]
    fn import_dddmp<'py>(
        &self,
        py: Python<'py>,
        file: &mut crate::util::DDDMPFile,
        support_vars: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let imported = crate::util::import_dddmp(&self.0, file, support_vars)?;
        PyList::new(py, imported.into_iter().map(ZBDDFunction))
    }

    /// Export the given decision diagram functions as DDDMP file.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     path (str | PathLike[str]): Path of the output file. If a file at
    ///         ``path`` exists, it will be overwritten, otherwise a new one
    ///         will be created.
    ///     functions (Iterable[ZBDDFunction]): Functions to export (must be
    ///         stored in this manager).
    ///     version (DDDMPVersion): DDDMP format version to use
    ///     ascii (bool): If ``True``, ASCII mode will be enforced for the
    ///         export. By default (and if ``False``), binary mode will be used
    ///         if supported for the decision diagram kind.
    ///         Binary mode is currently supported for BCDDs only.
    ///     strict (bool): If ``True`` (the default), enable `strict mode`_
    ///     diagram_name (str): Name of the decision diagram
    ///
    /// Returns:
    ///     None
    ///
    /// .. _`strict mode`: https://docs.rs/oxidd-dump/latest/oxidd_dump/dddmp/struct.ExportSettings.html#method.strict
    #[pyo3(
        signature = (/, path, functions, *, version=None, ascii=false, strict=true, diagram_name=""),
        text_signature = "($self, /, path, functions, *, version=DDDMPVersion.V2_0, ascii=False, strict=True, diagram_name=\"\")"
    )]
    fn export_dddmp<'py>(
        &self,
        path: PathBuf,
        functions: &Bound<'py, PyAny>,
        version: Option<&Bound<'py, PyAny>>,
        ascii: bool,
        strict: bool,
        diagram_name: &str,
    ) -> PyResult<()> {
        crate::util::export_dddmp::<ZBDDFunction>(
            &self.0,
            &path,
            functions,
            version,
            ascii,
            strict,
            diagram_name,
        )
    }

    /// Export the given decision diagram functions as DDDMP file.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     path (str | PathLike[str]): Path of the output file. If a file at
    ///         ``path`` exists, it will be overwritten, otherwise a new one
    ///         will be created.
    ///     functions (Iterable[tuple[ZBDDFunction, str]]): Pairs of function
    ///         and name. All functions must be stored in this manager.
    ///     version (DDDMPVersion): DDDMP format version to use
    ///     ascii (bool): If ``True``, ASCII mode will be enforced for the
    ///         export. By default (and if ``False``), binary mode will be used
    ///         if supported for the decision diagram kind.
    ///         Binary mode is currently supported for BCDDs only.
    ///     strict (bool): If ``True`` (the default), enable `strict mode`_
    ///     diagram_name (str): Name of the decision diagram
    ///
    /// Returns:
    ///     None
    ///
    /// .. _`strict mode`: https://docs.rs/oxidd-dump/latest/oxidd_dump/dddmp/struct.ExportSettings.html#method.strict
    #[pyo3(
        signature = (/, path, functions, *, version=None, ascii=false, strict=true, diagram_name=""),
        text_signature = "($self, /, path, functions, *, version=DDDMPVersion.V2_0, ascii=False, strict=True, diagram_name=\"\")"
    )]
    fn export_dddmp_with_names<'py>(
        &self,
        path: PathBuf,
        functions: &Bound<'py, PyAny>,
        version: Option<&Bound<'py, PyAny>>,
        ascii: bool,
        strict: bool,
        diagram_name: &str,
    ) -> PyResult<()> {
        crate::util::export_dddmp_with_names::<ZBDDFunction>(
            &self.0,
            &path,
            functions,
            version,
            ascii,
            strict,
            diagram_name,
        )
    }

    /// Serve the given decision diagram functions for visualization.
    ///
    /// Blocks until the visualization has been fetched by `OxiDD-vis`_ (or
    /// another compatible tool).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     diagram_name (str): Name of the decision diagram
    ///     functions (Iterable[ZBDDFunction]): Functions to visualize (must be
    ///         stored in this manager)
    ///     port (int): The port to provide the data on, defaults to 4000.
    ///
    /// Returns:
    ///     None
    ///
    /// .. _OxiDD-vis: https://oxidd.net/vis
    #[pyo3(signature = (/, diagram_name, functions, *, port=4000))]
    fn visualize<'py>(
        &self,
        py: Python<'py>,
        diagram_name: &str,
        functions: &Bound<'py, PyAny>,
        port: u16,
    ) -> PyResult<()> {
        let mut visualizer = oxidd_dump::Visualizer::new().port(port);
        let mut iter = crate::util::TryIter::<ZBDDFunction>::try_from(functions)?;
        visualizer = self
            .0
            .with_manager_shared(|manager| visualizer.add(diagram_name, manager, &mut iter));
        crate::util::visualize_serve(py, &mut visualizer)?;
        iter.err
    }

    /// Serve the given decision diagram functions for visualization.
    ///
    /// Blocks until the visualization has been fetched by `OxiDD-vis`_ (or
    /// another compatible tool).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Args:
    ///     diagram_name (str): Name of the decision diagram
    ///     functions (Iterable[tuple[ZBDDFunction, str]]): Pairs of function
    ///         and name. All functions must be stored in this manager.
    ///     port (int): The port to provide the data on, defaults to 4000.
    ///
    /// Returns:
    ///     None
    ///
    /// .. _OxiDD-vis: https://oxidd.net/vis
    #[pyo3(signature = (/, diagram_name, functions, *, port=4000))]
    fn visualize_with_names<'py>(
        &self,
        py: Python<'py>,
        diagram_name: &str,
        functions: &Bound<'py, PyAny>,
        port: u16,
    ) -> PyResult<()> {
        let mut visualizer = oxidd_dump::Visualizer::new().port(port);
        let mut iter = crate::util::FuncStrPairIter::<ZBDDFunction>::try_from(functions)?;
        visualizer = self.0.with_manager_shared(|manager| {
            visualizer.add_with_names(diagram_name, manager, &mut iter)
        });
        py.detach(|| visualizer.serve())?;
        iter.err
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
    ///         ``path`` exists, it will be overwritten, otherwise a new one
    ///         will be created.
    ///     functions (Iterable[tuple[ZBDDFunction, str]]): Optional names for
    ///         ZBDD functions
    ///
    /// Returns:
    ///     None
    #[pyo3(
        signature = (/, path, functions=None),
        text_signature = "($self, /, path, functions=[])"
    )]
    fn dump_all_dot<'py>(
        &self,
        path: PathBuf,
        functions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        crate::util::dump_all_dot::<ZBDDFunction>(&self.0, &path, functions)
    }
    /// Deprecated alias for :meth:`dump_all_dot`.
    ///
    /// Args:
    ///     path (str | PathLike[str]): Path of the output file. If a file at
    ///         ``path`` exists, it will be overwritten, otherwise a new one
    ///         will be created.
    ///     functions (Iterable[tuple[ZBDDFunction, str]]): Optional names for
    ///         BCDD functions
    ///
    /// Returns:
    ///     None
    ///
    /// .. deprecated:: 0.11
    ///    Use :meth:`dump_all_dot` instead
    #[pyo3(
        signature = (/, path, functions=None),
        text_signature = "($self, /, path, functions=[])"
    )]
    fn dump_all_dot_file<'py>(
        &self,
        path: PathBuf,
        functions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        crate::util::dump_all_dot::<ZBDDFunction>(&self.0, &path, functions)
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
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl std::ops::Deref for ZBDDFunction {
    type Target = oxidd::zbdd::ZBDDFunction;

    fn deref(&self) -> &Self::Target {
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

    /// Get the Boolean function v for the singleton set {v}.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Returns:
    ///     Self: The Boolean function ``v`` as ZBDD
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    ///
    /// .. deprecated:: 0.11
    ///    Use :meth:`ZBDDManager.var` instead
    fn var_boolean_function(&self, py: Python) -> PyResult<Self> {
        py.detach(move || {
            self.0.with_manager_shared(move |manager, singleton| {
                #[allow(deprecated)]
                let res = oxidd::zbdd::var_boolean_function(manager, singleton)?;
                Ok(oxidd::zbdd::ZBDDFunction::from_edge(manager, res))
            })
        })
        .try_into()
    }

    /// Get the subset of ``self`` not containing ``var``.
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Args:
    ///     var (int):  The variable
    ///
    /// Returns:
    ///     Self: ``{s ∈ self | {var} ∉ s}``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn subset0(&self, py: Python, var: VarNo) -> PyResult<Self> {
        py.detach(move || self.0.subset0(var)).try_into()
    }
    /// Get the subset of ``self`` containing ``var``, with ``var`` removed.
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Args:
    ///     var (int):  The variable
    ///
    /// Returns:
    ///     Self: ``{s ∖ {{var}} | s ∈ self ∧ {var} ∈ s}``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn subset1(&self, py: Python, var: VarNo) -> PyResult<Self> {
        py.detach(move || self.0.subset1(var)).try_into()
    }
    /// Swap :meth:`subset0` and :meth:`subset1` with respect to ``var``.
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Args:
    ///     var (int): The variable
    ///
    /// Returns:
    ///     Self: ``{s ∪ {{var}} | s ∈ self ∧ {var} ∉ s}
    ///     ∪ {s ∖ {{var}} | s ∈ self ∧ {var} ∈ s}``
    ///
    /// Raises:
    ///     DDMemoryError: If the operation runs out of memory
    fn change(&self, py: Python, var: VarNo) -> PyResult<Self> {
        py.detach(move || self.0.change(var)).try_into()
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
        py.detach(move || self.0.not()).try_into()
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
        py.detach(move || self.0.and(&rhs.0)).try_into()
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
        py.detach(move || self.0.or(&rhs.0)).try_into()
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
        py.detach(move || self.0.xor(&rhs.0)).try_into()
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
        py.detach(move || self.0.nand(&rhs.0)).try_into()
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
        py.detach(move || self.0.nor(&rhs.0)).try_into()
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
        py.detach(move || self.0.equiv(&rhs.0)).try_into()
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
        py.detach(move || self.0.imp(&rhs.0)).try_into()
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
        py.detach(move || self.0.imp_strict(&rhs.0)).try_into()
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
        py.detach(move || self.0.ite(&t.0, &e.0)).try_into()
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
        py.detach(move || self.0.node_count())
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
        py.detach(move || {
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
        py.detach(move || {
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
        match py.detach(move || self.0.pick_cube(move |_, _, _| false)) {
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
        py.detach(move || self.0.pick_cube_dd(move |_, _, _| false))
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
        py.detach(move || self.0.pick_cube_dd_set(&literal_set.0))
            .try_into()
    }

    /// Evaluate this Boolean function with arguments ``args``.
    ///
    /// Note that the domain of the Boolean function represented by ``self`` is
    /// implicit and may comprise a strict subset of the variables in the
    /// manager only. This method assumes that the function's domain
    /// corresponds the set of variables in ``args``. Remember that for ZBDDs,
    /// the domain plays a crucial role for the interpretation of decision
    /// diagram nodes as a Boolean function. This is in contrast to, e.g.,
    /// ordinary BDDs, where extending the domain does not affect the
    /// evaluation result.
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
