//! Primitives and utilities

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::True;
use pyo3::types::{PyString, PyTuple};
use pyo3::PyClass;

use oxidd::BooleanOperator;
use oxidd_core::Countable;

// pyi: class DDMemoryError(MemoryError):
//d Exception that is raised in case a DD operation runs out of memory
pyo3::create_exception!(
    oxidd.util,
    DDMemoryError,
    pyo3::exceptions::PyMemoryError,
    "Exception that is raised in case a DD operation runs out of memory."
);

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
