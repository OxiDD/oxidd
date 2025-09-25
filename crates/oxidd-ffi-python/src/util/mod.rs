//! Primitives and utilities

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyRange, PyString, PyTuple};

use oxidd::{
    BooleanFunction, BooleanOperator, Function, HasLevel, LevelNo, Manager, ManagerRef, VarNo,
};
use oxidd_core::function::{ETagOfFunc, INodeOfFunc, TermOfFunc};
use oxidd_dump::dot::DotStyle;

mod dddmp;
pub use dddmp::*;

// pyi: class DDMemoryError(MemoryError):
pyo3::create_exception!(
    oxidd.util,
    DDMemoryError,
    pyo3::exceptions::PyMemoryError,
    "Exception that is raised in case a DD operation runs out of memory."
);

macro_rules! enum_conversion_fn {
    ($f:ident, $pyt:literal -> $t:ty) => {
        pub fn $f(obj: &::pyo3::Bound<::pyo3::types::PyAny>) -> ::pyo3::PyResult<$t> {
            if let Ok(val) = obj.getattr("value") {
                if let Ok(val) = val.extract() {
                    if val <= <$t as ::oxidd_core::Countable>::MAX_VALUE {
                        return Ok(<$t as ::oxidd_core::Countable>::from_usize(val));
                    }
                }
            }
            Err(match obj.get_type().fully_qualified_name() {
                Ok(name) => ::pyo3::exceptions::PyTypeError::new_err(format!(
                    concat!("Expected an instance of ", $pyt, " got {}"),
                    name.to_string_lossy()
                )),
                Err(_) => ::pyo3::exceptions::PyTypeError::new_err(concat!(
                    "Expected an instance of ",
                    $pyt
                )),
            })
        }
    };
}
use enum_conversion_fn;

enum_conversion_fn!(boolean_operator, "oxidd.util.BooleanOperator" -> BooleanOperator);

#[inline]
pub fn var_no_bounds_check<M: Manager>(manager: &M, var: VarNo) -> PyResult<()> {
    if var < manager.num_vars() {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyIndexError::new_err(
            "variable number out of bounds",
        ))
    }
}

#[inline]
pub fn level_no_bounds_check<M: Manager>(manager: &M, level: LevelNo) -> PyResult<()> {
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

pub fn add_named_vars<'py, M: ManagerRef>(
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

#[derive(FromPyObject)]
pub enum VarId {
    #[pyo3(transparent, annotation = "int")]
    Number(VarNo),
    #[pyo3(transparent, annotation = "str")]
    Name(String),
}

pub fn with_var_no<MR: ManagerRef, R>(
    manager_ref: &MR,
    var: VarId,
    f: impl for<'id> FnOnce(&MR::Manager<'id>, VarNo) -> PyResult<R>,
) -> PyResult<R> {
    manager_ref.with_manager_shared(|manager| {
        let var = match var {
            VarId::Number(var) => {
                var_no_bounds_check(manager, var)?;
                var
            }
            VarId::Name(name) => match manager.name_to_var(&name) {
                Some(var) => var,
                None => return Err(pyo3::exceptions::PyKeyError::new_err(name)),
            },
        };
        f(manager, var)
    })
}

pub struct TryIter<'py, T> {
    iter: Bound<'py, PyIterator>,
    i: usize,
    len: Option<usize>,
    pub err: PyResult<()>,
    _phantom: PhantomData<T>,
}

impl<'py, T> TryFrom<&Bound<'py, PyAny>> for TryIter<'py, T> {
    type Error = PyErr;

    fn try_from(iterable: &Bound<'py, PyAny>) -> PyResult<Self> {
        let len = iterable.len().ok();
        Ok(Self {
            iter: iterable.try_iter()?,
            i: 0,
            len,
            err: Ok(()),
            _phantom: PhantomData,
        })
    }
}

impl<'py, T: FromPyObject<'py>> Iterator for TryIter<'py, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.iter.next()?.and_then(|v| v.extract::<T>());
        if let Err(err) = res {
            self.err = Err(err);
            return None;
        }
        self.i += 1;
        res.ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.len {
            Some(len) => {
                let n = len - self.i;
                (n, Some(n))
            }
            None => (0, None),
        }
    }
}

pub fn collect_vec<'py, T: FromPyObject<'py>>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<T>> {
    obj.try_iter()?
        .map(|v| v.and_then(|v| v.extract()))
        .collect()
}

#[derive(Clone, FromPyObject)]
pub struct PyStringDisplay<'py>(Bound<'py, PyString>);

impl<'py> fmt::Display for PyStringDisplay<'py> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.to_string_lossy())
    }
}

pub type FuncStrPairIter<'py, PYF> = TryIter<'py, (PYF, PyStringDisplay<'py>)>;

struct OptIter<I>(Option<I>);

impl<I: Iterator> Iterator for OptIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut()?.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            Some(it) => it.size_hint(),
            None => (0, Some(0)),
        }
    }
}

pub fn dump_all_dot<'py, PYF>(
    manager_ref: &<PYF::Target as Function>::ManagerRef,
    path: &Path,
    functions: Option<&Bound<'py, PyAny>>,
) -> PyResult<()>
where
    PYF: FromPyObject<'py> + Deref,
    PYF::Target: Function + for<'id> DotStyle<ETagOfFunc<'id, PYF::Target>>,
    for<'id> INodeOfFunc<'id, PYF::Target>: HasLevel,
    for<'id> ETagOfFunc<'id, PYF::Target>: fmt::Debug,
    for<'id> TermOfFunc<'id, PYF::Target>: fmt::Display,
{
    let file = std::fs::File::create(path)?;

    let mut iter = OptIter(match functions {
        Some(iterable) => Some(FuncStrPairIter::<PYF>::try_from(iterable)?),
        None => None,
    });
    manager_ref.with_manager_shared(|manager| {
        oxidd_dump::dot::dump_all(std::io::BufWriter::new(file), manager, &mut iter)
    })?;

    if let Some(it) = iter.0 {
        it.err?;
    }

    Ok(())
}

pub fn eval<B: BooleanFunction>(f: &B, args: &Bound<PyAny>) -> PyResult<bool> {
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

pub fn visualize_serve(py: Python<'_>, visualizer: &mut oxidd_dump::Visualizer) -> PyResult<()> {
    // Polling repeatedly is not as elegant as just waiting (via
    // `visualizer.serve()`), but this appears to be the only way we can handle
    // Python signals and, e.g., return in case of a keyboard interrupt.
    let mut listener = visualizer.serve_nonblocking()?;
    while !py.detach(|| listener.poll())? {
        py.check_signals()?;
        py.detach(|| std::thread::sleep(std::time::Duration::from_millis(200)));
    }
    Ok(())
}
