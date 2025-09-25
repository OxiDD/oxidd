use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::{fs, io};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use oxidd::{BooleanFunction, Function, HasLevel, ManagerRef};
use oxidd_core::function::{ETagOfFunc, INodeOfFunc, TermOfFunc};
use oxidd_dump::dddmp;

// spell-checker:dictionaries dddmp

super::enum_conversion_fn!(dddmp_version, "oxidd.util.DDDMPVersion" -> dddmp::DDDMPVersion);

pub fn dddmp_version_or_default(version: Option<&Bound<PyAny>>) -> PyResult<dddmp::DDDMPVersion> {
    match version {
        Some(obj) => dddmp_version(obj),
        None => Ok(dddmp::DDDMPVersion::default()),
    }
}

/// DDDMP header loaded as the first step of an import process.
#[pyclass]
pub struct DDDMPFile {
    reader: Option<io::BufReader<fs::File>>,
    header: dddmp::DumpHeader,
}

#[pymethods]
impl DDDMPFile {
    /// Load a DDDMP header from file.
    ///
    /// Args:
    ///     path (str | PathLike[str]): Path to the DDDMP file
    ///
    /// Returns:
    ///     DDDMPFile: The loaded DDDMP header
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let mut reader = io::BufReader::new(fs::File::open(path)?);
        let header = dddmp::DumpHeader::load(&mut reader)?;
        Ok(DDDMPFile {
            reader: Some(reader),
            header,
        })
    }

    /// Close the file handle.
    ///
    /// If the file is already close, this is a no-op.
    ///
    /// Returns:
    ///     None
    fn close(&mut self) {
        drop(self.reader.take())
    }

    /// Enter the runtime context related to this object.
    ///
    /// This is a no-op.
    ///
    /// Returns:
    ///     Self: This object
    fn __enter__<'a>(slf: PyRef<'a, Self>) -> PyRef<'a, Self> {
        slf
    }

    /// Exit the runtime context related to this object.
    ///
    /// Closes the associated file handle (via :meth:`close`).
    ///
    /// Args:
    ///     exc_type (type[BaseException] | None): The exception type if an
    ///         exception was raised within the context, or ``None``
    ///     exc_value (BaseException | None): The exception that was raised
    ///         within the context, or ``None``
    ///     traceback (TracebackType | None): The traceback related to the
    ///         exception raised within the context, or ``None``
    ///
    /// Returns:
    ///     bool: ``True`` iff the exception should be suppressed
    fn __exit__(
        &mut self,
        exc_type: &Bound<PyAny>,
        exc_value: &Bound<PyAny>,
        traceback: &Bound<PyAny>,
    ) -> bool {
        let _ = (exc_type, exc_value, traceback);
        self.close();
        false
    }

    /// str | None: Name of the decision diagram.
    ///
    /// Corresponds to the DDDMP ``.dd`` field.
    #[getter]
    fn diagram_name(&self) -> Option<&str> {
        self.header.diagram_name()
    }

    /// int: Number of nodes in the dumped decision diagram.
    ///
    /// Corresponds to the DDDMP ``.nnodes`` field.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.header.num_nodes()
    }

    /// int: Number of all variables in the exported decision.
    ///
    /// Corresponds to the DDDMP ``.nvars`` field.
    #[getter]
    fn num_vars(&self) -> u32 {
        self.header.num_vars()
    }

    /// int: Number of variables in the true support of the decision diagram.
    ///
    /// Corresponds to the DDDMP ``.nsuppvars`` field.
    #[getter]
    fn num_support_vars(&self) -> u32 {
        self.header.num_support_vars()
    }

    /// list[int]: Variables in the true support of the decision diagram.
    ///
    /// Concretely, these are indices of the original variable numbering. Hence,
    /// the list contains :attr:`num_support_vars` integers in strictly
    /// ascending order.
    ///
    /// .. admonition:: Example
    ///
    ///    Consider a decision diagram that was created with the variables
    ///    ``x``, ``y``, and ``z``, in this order (``x`` is the top-most
    ///    variable). Suppose that only ``y`` and ``z`` are used by the dumped
    ///    functions. Then, this value is ``[1, 2]``, regardless of any
    ///    subsequent reordering.
    ///
    /// Corresponds to the DDDMP ``.ids`` field.
    #[getter]
    fn support_vars(&self) -> &[u32] {
        self.header.support_vars()
    }

    /// list[int]: Order of the support variables.
    ///
    /// This list is always :attr:`num_support_vars` elements long and
    /// represents a mapping from positions to variable numbers.
    ///
    /// .. admonition:: Example
    ///
    ///    Consider a decision diagram that was created with the variables
    ///    ``x``, ``y``, and ``z`` (``x`` is the top-most variable). The
    ///    variables were re-ordered to ``z``, ``x``, ``y``. Suppose that only
    ///    ``y`` and ``z`` are used by the dumped functions. Then, this value is
    ///    ``[2, 1]``.
    #[getter]
    fn support_var_order(&self) -> &[u32] {
        self.header.support_var_order()
    }

    /// list[int]: Mapping from the support variables to levels.
    ///
    /// This list is always :attr:`num_support_vars` elements long. If the value
    /// at index ``i`` is ``l``, then the ``i``\ th support variable is at level
    /// ``l`` in the dumped decision diagram. By the ``i``\ th support variable,
    /// we mean the variable ``header.support_vars[i]`` in the original
    /// numbering.
    ///
    /// .. admonition:: Example
    ///
    ///    Consider a decision diagram that was created with the variables
    ///    ``x``, ``y``, and ``z`` (``x`` is the top-most variable). The
    ///    variables were re-ordered to ``z``, ``x``, ``y``. Suppose that only
    ///    ``y`` and ``z`` are used by the dumped functions. Then, this value is
    ///    ``[2, 0]``.
    ///
    /// Corresponds to the DDDMP ``.permids`` field.
    #[getter]
    fn support_var_to_level(&self) -> &[u32] {
        self.header.support_var_to_level()
    }

    /// list[int]: Auxiliary variable IDs.
    ///
    /// This list contains :attr:`num_support_vars` elements.
    ///
    /// Corresponds to the DDDMP ``.auxids`` field.
    #[getter]
    fn auxiliary_var_ids(&self) -> &[u32] {
        self.header.auxiliary_var_ids()
    }

    /// list[str] | None: Names of all variables in the decision diagram.
    ///
    /// If present, this list contains :attr:`num_support_vars` many elements.
    /// The order is the "original" variable order.
    ///
    /// Corresponds to the DDDMP ``.varnames`` field, but ``.orderedvarnames``
    /// and ``.suppvarnames`` are also considered if one of the fields is
    /// missing. All variable names are non-empty unless only
    /// ``.suppvarnames`` is given in the input (in which case only the
    /// names of support variables are non-empty). The return value is only
    /// ``None`` if neither of ``.varnames``, ``.orderedvarnames``, and
    /// ``.suppvarnames`` is present in the input.
    #[getter]
    fn var_names(&self) -> Option<&[String]> {
        self.header.var_names()
    }

    /// int: Number of roots.
    ///
    /// :meth:`Manager.import()` returns this number of roots on success.
    /// Corresponds to the DDDMP ``.nroots`` field.
    #[getter]
    fn num_roots(&self) -> usize {
        self.header.num_roots()
    }

    /// list[int] | None: Names of roots, if present.
    ///
    /// The order matches the one of the result of :meth:`Manager.import()`.
    ///
    /// Corresponds to the DDDMP ``.rootnames`` field.
    #[getter]
    fn root_names(&self) -> Option<&[String]> {
        self.header.root_names()
    }
}

pub fn import_dddmp<'py, F>(
    manager_ref: &F::ManagerRef,
    file: &mut DDDMPFile,
    support_vars: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<F>>
where
    F: BooleanFunction,
    for<'id> INodeOfFunc<'id, F>: HasLevel,
    for<'id> TermOfFunc<'id, F>: oxidd_dump::ParseTagged<ETagOfFunc<'id, F>>,
{
    let Some(reader) = &mut file.reader else {
        return Err(PyRuntimeError::new_err(
            "DDDMP file is closed. Note that a DDDMPFile only supports a single import.",
        ));
    };
    Ok(match support_vars {
        Some(vars) => {
            let mut iter = super::TryIter::try_from(vars)?;
            let import_res = manager_ref.with_manager_shared(|manager| {
                oxidd_dump::dddmp::import(
                    reader,
                    &file.header,
                    manager,
                    &mut iter,
                    F::not_edge_owned,
                )
            });
            file.close();
            iter.err?;
            import_res
        }
        None => {
            let import_res = manager_ref.with_manager_shared(|manager| {
                oxidd_dump::dddmp::import(
                    reader,
                    &file.header,
                    manager,
                    file.header.support_var_order().iter().copied(),
                    F::not_edge_owned,
                )
            });
            file.close();
            import_res
        }
    }?)
}

pub fn export_dddmp<'py, PYF>(
    manager_ref: &<PYF::Target as Function>::ManagerRef,
    path: &Path,
    functions: &Bound<'py, PyAny>,
    version: Option<&Bound<'py, PyAny>>,
    ascii: bool,
    strict: bool,
    diagram_name: &str,
) -> PyResult<()>
where
    PYF: FromPyObject<'py> + Deref,
    PYF::Target: Function,
    for<'id> INodeOfFunc<'id, PYF::Target>: HasLevel,
    for<'id> TermOfFunc<'id, PYF::Target>: oxidd_dump::AsciiDisplay,
{
    let version = dddmp_version_or_default(version)?;
    let file = std::fs::File::create(path)?;

    let mut iter = super::TryIter::<PYF>::try_from(functions)?;
    manager_ref.with_manager_shared(|manager| {
        let settings = dddmp::ExportSettings::default();
        let settings = if ascii { settings.ascii() } else { settings };
        settings
            .diagram_name(diagram_name)
            .strict(strict)
            .version(version)
            .export(std::io::BufWriter::new(file), manager, &mut iter)?;
        Ok::<_, PyErr>(())
    })?;
    iter.err
}

pub fn export_dddmp_with_names<'py, PYF>(
    manager_ref: &<PYF::Target as Function>::ManagerRef,
    path: &Path,
    functions: &Bound<'py, PyAny>,
    version: Option<&Bound<'py, PyAny>>,
    ascii: bool,
    strict: bool,
    diagram_name: &str,
) -> PyResult<()>
where
    PYF: FromPyObject<'py> + Deref,
    PYF::Target: Function,
    for<'id> INodeOfFunc<'id, PYF::Target>: HasLevel,
    for<'id> TermOfFunc<'id, PYF::Target>: oxidd_dump::AsciiDisplay,
{
    let version = dddmp_version_or_default(version)?;
    let file = std::fs::File::create(path)?;

    let mut iter = super::FuncStrPairIter::<PYF>::try_from(functions)?;
    manager_ref.with_manager_shared(|manager| {
        let settings = dddmp::ExportSettings::default();
        let settings = if ascii { settings.ascii() } else { settings };
        settings
            .diagram_name(diagram_name)
            .strict(strict)
            .version(version)
            .export_with_names(std::io::BufWriter::new(file), manager, &mut iter)?;
        Ok::<_, PyErr>(())
    })?;
    iter.err
}
