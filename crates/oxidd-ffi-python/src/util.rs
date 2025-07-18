//! Primitives and utilities

use std::fmt;
use std::ops::Range;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::True;
use pyo3::types::{PyRange, PyString, PyTuple};
use pyo3::PyClass;

use oxidd::{BooleanFunction, BooleanOperator, LevelNo, Manager, ManagerRef, VarNo};
use oxidd_core::Countable;

// pyi: class DDMemoryError(MemoryError):
//d Exception that is raised in case a DD operation runs out of memory
pyo3::create_exception!(
    oxidd.util,
    DDMemoryError,
    pyo3::exceptions::PyMemoryError,
    "Exception that is raised in case a DD operation runs out of memory."
);

#[inline]
pub(crate) fn var_no_bounds_check<M: Manager>(manager: &M, var: VarNo) -> PyResult<()> {
    if var < manager.num_vars() {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "variable number out of bounds",
        ))
    }
}

#[inline]
pub(crate) fn level_no_bounds_check<M: Manager>(manager: &M, level: LevelNo) -> PyResult<()> {
    if level < manager.num_vars() {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "level number out of bounds",
        ))
    }
}

/// Error details for labelling a variable with a name that is already in use.
#[pyclass(frozen, eq, hash, str, module = "oxidd.util")]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct DuplicateVarName {
    /// str: The variable name.
    #[pyo3(get)]
    pub name: String,
    /// int: Variable number already using the name.
    #[pyo3(get)]
    pub present_var: VarNo,
    pub added_vars: Range<VarNo>,
}

impl fmt::Display for DuplicateVarName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the variable name '{}' is already in use for variable number {}",
            self.name, self.present_var
        )
    }
}

impl From<oxidd::error::DuplicateVarName> for DuplicateVarName {
    fn from(err: oxidd::error::DuplicateVarName) -> Self {
        Self {
            name: err.name,
            present_var: err.present_var,
            added_vars: err.added_vars,
        }
    }
}

#[pymethods]
impl DuplicateVarName {
    /// range: Range of variables successfully added before the error occurred.
    #[getter]
    fn added_vars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyRange>> {
        PyRange::new(py, self.added_vars.start as _, self.added_vars.end as _)
    }

    /// Get a string representation.
    ///
    /// Returns:
    ///     str: The string representation.
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

pub(crate) fn add_named_vars<'py, M: ManagerRef>(
    manager_ref: &M,
    py: Python<'py>,
    names: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyRange>> {
    struct NameIter<'py> {
        size_hint: Option<usize>,
        /// Remaining variables before [`Manager::add_named_vars()`] would
        /// panic because of too many variables
        remaining_vars: VarNo,
        iter: Bound<'py, pyo3::types::PyIterator>,
        exc: PyResult<()>,
    }

    impl<'a> Iterator for NameIter<'a> {
        type Item = String;

        fn next(&mut self) -> Option<String> {
            self.exc.as_ref().ok()?;
            if self.remaining_vars == 0 {
                self.exc = Err(pyo3::exceptions::PyOverflowError::new_err(
                    "too many variables",
                ));
                return None;
            }
            match self
                .iter
                .next()?
                .and_then(|v| Ok(v.downcast_into::<PyString>()?))
            {
                Err(err) => {
                    self.exc = Err(err);
                    None
                }
                Ok(val) => {
                    self.remaining_vars -= 1;
                    Some(val.to_string_lossy().into_owned())
                }
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, self.size_hint)
        }
    }

    manager_ref.with_manager_exclusive(|manager| {
        let mut name_iter = NameIter {
            size_hint: names.len().ok(),
            remaining_vars: VarNo::MAX - manager.num_vars(),
            iter: names.try_iter()?,
            exc: Ok(()),
        };

        let res = manager.add_named_vars(name_iter.by_ref());
        name_iter.exc?;

        match res {
            Ok(vars) => Ok(PyRange::new(py, vars.start as _, vars.end as _)?),
            Err(err) => Err(pyo3::exceptions::PyValueError::new_err(DuplicateVarName {
                name: err.name,
                present_var: err.present_var,
                added_vars: err.added_vars,
            })),
        }
    })
}

pub(crate) fn collect_func_str_pairs<'py, F, PYF>(
    pairs: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<(F, Bound<'py, PyString>)>>
where
    F: Clone,
    PYF: Sync + PyClass<Frozen = True> + AsRef<F>,
{
    let mut ps = Vec::new();
    if let Some(pairs) = pairs {
        if let Ok(len) = pairs.len() {
            ps.reserve(len);
        }
        for pair in pairs.try_iter()? {
            let pair: Bound<PyTuple> = pair?.downcast_into()?;
            let f = pair.get_borrowed_item(0)?;
            let f = f.downcast::<PYF>()?.get().as_ref().clone();
            let s: Bound<PyString> = pair.get_item(1)?.downcast_into()?;
            ps.push((f, s));
        }
    }
    Ok(ps)
}

pub(crate) fn boolean_operator(obj: &Bound<PyAny>) -> PyResult<BooleanOperator> {
    if let Ok(val) = obj.getattr("value") {
        if let Ok(val) = val.extract() {
            if val <= BooleanOperator::MAX_VALUE {
                return Ok(BooleanOperator::from_usize(val));
            }
        }
    }
    Err(match obj.get_type().fully_qualified_name() {
        Ok(name) => PyTypeError::new_err(format!(
            "Expected an instance of oxidd.util.BooleanOperator, got {}",
            name.to_string_lossy()
        )),
        Err(_) => PyTypeError::new_err("Expected an instance of oxidd.util.BooleanOperator"),
    })
}

pub(crate) fn eval<B: BooleanFunction>(f: &B, args: &Bound<PyAny>) -> PyResult<bool> {
    let mut vals = Vec::with_capacity(args.len().unwrap_or(0));
    f.with_manager_shared(|manager, edge| {
        let num_vars = manager.num_vars();
        for pair in args.try_iter()? {
            let pair: Bound<PyTuple> = pair?.downcast_into()?;
            let var = pair.get_borrowed_item(0)?.extract()?;
            if var >= num_vars {
                var_no_bounds_check(manager, var)?;
            }
            let b = pair.get_borrowed_item(1)?.is_truthy()?;
            vals.push((var, b));
        }
        Ok(B::eval_edge(manager, edge, vals))
    })
}
