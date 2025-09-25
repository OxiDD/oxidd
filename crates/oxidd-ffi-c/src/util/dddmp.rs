use std::borrow::Cow;
use std::ffi::{c_char, OsStr};
use std::{fs, io};

use oxidd::{BooleanFunction, HasLevel, LevelNo, ManagerRef, VarNo};
use oxidd_core::function::{ETagOfFunc, INodeOfFunc, TermOfFunc};
use oxidd_dump::{dddmp, AsciiDisplay, ParseTagged, Visualizer};

use super::{error_t, handle_err, handle_err_or_init, slice, str_t, CFunction, CManagerRef};

// spell-checker:dictionaries dddmp

/// DDDMP header loaded as the first step of an import process.
pub struct dddmp_file_t {
    reader: io::BufReader<fs::File>,
    header: dddmp::DumpHeader,
}

impl dddmp_file_t {
    pub unsafe fn import_into<CF: CFunction>(
        &mut self,
        manager: CF::CManagerRef,
        support_vars: *const VarNo,
        mut roots: *mut CF,
        error: *mut error_t,
    ) -> bool
    where
        CF::Function: BooleanFunction,
        for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
        for<'id> TermOfFunc<'id, CF::Function>: ParseTagged<ETagOfFunc<'id, CF::Function>>,
    {
        let support_vars = if support_vars.is_null() {
            self.header.support_var_order()
        } else {
            std::slice::from_raw_parts(support_vars, self.header.num_support_vars() as usize)
        };
        let res = manager.get().with_manager_shared(|manager| {
            oxidd_dump::dddmp::import::<CF::Function>(
                &mut self.reader,
                &self.header,
                manager,
                support_vars.iter().copied(),
                CF::Function::not_edge_owned,
            )
        });
        let Some(rs) = handle_err_or_init(res, error) else {
            return false;
        };
        for root in rs {
            unsafe { roots.write(root.into()) };
            roots = unsafe { roots.add(1) };
        }
        true
    }
}

/// Open the DDDMP file at `path` and load its header
///
/// @param  path      Path of the DDDMP file
/// @param  path_len  Length of `path` in bytes (excluding any trailing null
///                   byte)
/// @param  error     Output parameter for error details. May be `NULL` to
///                   ignore these details (the function will return `NULL` upon
///                   an error nonetheless).
///                   The error struct does not need to be initialized but is
///                   guaranteed to be initialized on return (if the pointer is
///                   non-null).
///
/// @returns  The DDDMP file handle or `NULL` on error
#[no_mangle]
pub unsafe extern "C" fn oxidd_dddmp_open(
    path: *const c_char,
    path_len: usize,
    error: *mut error_t,
) -> Option<Box<dddmp_file_t>> {
    let path = std::slice::from_raw_parts(path.cast(), path_len);
    let path = std::ffi::OsStr::from_encoded_bytes_unchecked(path);
    let mut reader = io::BufReader::new(handle_err(fs::File::open(path), error)?);

    let header = handle_err(dddmp::DumpHeader::load(&mut reader), error)?;
    Some(Box::new(dddmp_file_t { reader, header }))
}

/// Close the DDDMP file handle and deallocate the imported metadata
#[no_mangle]
pub extern "C" fn oxidd_dddmp_close(file: Option<Box<dddmp_file_t>>) {
    drop(file);
}

/// Get the name of the decision diagram
///
/// Corresponds to the DDDMP `.dd` field. If the header does not contain that
/// field, the returned string will be empty.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_diagram_name(file: &dddmp_file_t) -> str_t {
    file.header.diagram_name().unwrap_or_default().into()
}

/// Get the number of nodes in the dumped decision diagram
///
/// Corresponds to the DDDMP `.nnodes` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_num_nodes(file: &dddmp_file_t) -> usize {
    file.header.num_nodes()
}

/// Get the number of all variables in the exported decision diagram
///
/// Corresponds to the DDDMP `.nvars` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_num_vars(file: &dddmp_file_t) -> VarNo {
    file.header.num_vars()
}

/// Get the number of variables in the true support of the decision diagram
///
/// Corresponds to the DDDMP `.nsuppvars` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_num_support_vars(file: &dddmp_file_t) -> VarNo {
    file.header.num_support_vars()
}

/// Get the variables in the true support of the decision diagram
///
/// Concretely, these are indices of the original variable numbering. Hence, the
/// returned slice contains [`DumpHeader::num_support_vars()`] integers in
/// strictly ascending order.
///
/// Example: Consider a decision diagram that was created with the variables
/// `x`, `y`, and `z`, in this order (`x` is the top-most variable). Suppose
/// that only `y` and `z` are used by the dumped functions. Then, the
/// returned slice is `[1, 2]`, regardless of any subsequent reordering.
///
/// Corresponds to the DDDMP `.ids` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_support_vars(file: &dddmp_file_t) -> slice<VarNo> {
    file.header.support_vars().into()
}

/// Get the support variables' order
///
/// The returned slice is always [`DumpHeader::num_support_vars()`] elements
/// long and represents a mapping from positions to variable numbers.
///
/// Example: Consider a decision diagram that was created with the variables
/// `x`, `y`, and `z` (`x` is the top-most variable). The variables were
/// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
/// the dumped functions. Then, the returned slice is `[2, 1]`.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_support_var_order(file: &dddmp_file_t) -> slice<VarNo> {
    file.header.support_var_order().into()
}

/// Get the mapping from support variables to levels
///
/// The returned slice is always [`DumpHeader::num_support_vars()`]
/// elements long. If the value at the `i`th index is `l`, then the `i`th
/// support variable is at level `l` in the dumped decision diagram. By the
/// `i`th support variable, we mean the variable `header.support_vars()[i]`
/// in the original numbering.
///
/// Example: Consider a decision diagram that was created with the variables
/// `x`, `y`, and `z` (`x` is the top-most variable). The variables were
/// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
/// the dumped functions. Then, the returned slice is `[2, 0]`.
///
/// Corresponds to the DDDMP `.permids` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_support_var_to_level(file: &dddmp_file_t) -> slice<LevelNo> {
    file.header.support_var_to_level().into()
}

/// Get whether the header contains variable names
#[no_mangle]
pub extern "C" fn oxidd_dddmp_has_var_names(file: &dddmp_file_t) -> bool {
    file.header.var_names().is_some()
}

/// Get the name of variable `i`
///
/// `i` must be less than `oxidd_dddmp_num_vars(file)`.
///
/// The variable names are read from the DDDMP `.varnames` field, but
/// `.orderedvarnames` and `.suppvarnames` are also considered if one of the
/// fields is missing. All variable names are non-empty unless only
/// `.suppvarnames` is given in the input (in which case only the names of
/// support variables are non-empty). The return value is only `None` if neither
/// of `.varnames`, `.orderedvarnames`, and `.suppvarnames` is present in the
/// input.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_var_name(file: &dddmp_file_t, i: VarNo) -> str_t {
    match file.header.var_names() {
        Some(s) => s[i as usize].as_str().into(),
        None => "".into(),
    }
}

/// Get the number of roots
///
/// The `*_import_dddmp()` functions return this number of roots on success.
/// Corresponds to the DDDMP `.nroots` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_num_roots(file: &dddmp_file_t) -> usize {
    file.header.num_roots()
}

/// Get whether the header contains functions names
#[no_mangle]
pub extern "C" fn oxidd_dddmp_has_root_names(file: &dddmp_file_t) -> bool {
    file.header.root_names().is_some()
}

/// Get the name of root `i`
///
/// `i` must be less than `oxidd_dddmp_num_roots()`.
///
/// The names are read from the DDDMP `.rootnames` field.
#[no_mangle]
pub extern "C" fn oxidd_dddmp_root_name(file: &dddmp_file_t, i: usize) -> str_t {
    match file.header.root_names() {
        Some(s) => s[i].as_str().into(),
        None => "".into(),
    }
}

/// DDDMP format version version
#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(unused)] // variants may be constructed in the foreign language
pub enum dddmp_version {
    /// Version 2.0, bundled with [CUDD] 3.0
    ///
    /// [CUDD]: https://github.com/cuddorg/cudd
    _2_0, // cbindgen translates this to OXIDD_DDDMP_VERSION_2_0
    /// Version 3.0, used by [BDDSampler] and [Logic2BDD]
    ///
    /// [BDDSampler]: https://github.com/davidfa71/BDDSampler
    /// [Logic2BDD]: https://github.com/davidfa71/Extending-Logic
    _3_0,
}

/// Settings for exporting a decision diagram in the DDDMP format
#[derive(Clone, Copy)]
#[repr(C)]
pub struct dddmp_export_settings_t {
    /// DDDMP format version for the export
    pub version: dddmp_version,
    /// Whether to enforce the human-readable ASCII mode
    ///
    /// If `false`, the more compact binary format will be used, provided that
    /// it is supported for the respective decision diagram kind. Currently,
    /// binary mode is supported for BCDDs only.
    pub ascii: bool,
    /// Whether to enable strict mode
    ///
    /// The DDDMP format imposes some restrictions on diagram, variable, and
    /// function names:
    ///
    /// - None of them may contain ASCII control characters (e.g., line breaks).
    /// - Variable and function names must not contain spaces either.
    /// - Variable and function names must not be empty. (However, it is
    ///   possible to not export any variable or function names at all.)
    ///
    /// In the diagram name, control characters will be replaced by spaces. In
    /// variable and function names, an underscore (`_`) is used as the
    /// replacement character. When using any of the `*_dddmp_export_with_names`
    /// functions, empty function names are replaced by `_f{i}`, where `{i}`
    /// stands for the position in the iterator. Empty variable names are
    /// replaced by `_x{i}`. To retain uniqueness, as many underscores are
    /// added to the prefix as there are in the longest prefix over all
    /// present variable names.
    ///
    /// However, since such relabelling may lead to unexpected results, there is
    /// a strict mode. In strict mode, no variable names will be exported unless
    /// all variables are named. Additionally, an error will be generated upon
    /// any replacement. The error will not be propagated immediately but only
    /// after the file was written. This should simplify inspecting the error
    /// and also serve as a checkpoint for very long-running computations.
    pub strict: bool,
    /// Name of the decision diagram
    pub diagram_name: str_t,
}

pub fn with_export_settings<T>(
    settings: Option<&dddmp_export_settings_t>,
    f: impl FnOnce(&dddmp::ExportSettings) -> T,
) -> T {
    let set = dddmp::ExportSettings::default();
    if let Some(&dddmp_export_settings_t {
        version,
        ascii,
        strict,
        diagram_name,
    }) = settings
    {
        let export = if ascii { set.ascii() } else { set };
        let name = diagram_name.to_str_lossy();
        let version = match version {
            dddmp_version::_2_0 => dddmp::DDDMPVersion::V2_0,
            dddmp_version::_3_0 => dddmp::DDDMPVersion::V3_0,
        };
        f(&export.version(version).strict(strict).diagram_name(&name))
    } else {
        f(&set)
    }
}

pub unsafe fn export<CF: CFunction>(
    manager: CF::CManagerRef,
    path: &OsStr,
    functions: *const CF,
    num_functions: usize,
    function_names: *const *const c_char,
    settings: Option<&dddmp_export_settings_t>,
    error: *mut error_t,
) -> Option<()>
where
    for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
    for<'id> TermOfFunc<'id, CF::Function>: AsciiDisplay,
{
    let file = handle_err(std::fs::File::create(path), error)?;
    let functions = super::slice_from_raw_parts(functions, num_functions);
    let mut invalid_func = Ok(());
    let res = manager.get().with_manager_shared(|manager| {
        with_export_settings(settings, |settings| {
            if function_names.is_null() {
                let iter = functions.iter().enumerate().filter_map(|(i, f)| {
                    let res = f.get().ok();
                    if res.is_none() {
                        invalid_func = Err((i, Cow::Borrowed("")));
                    }
                    res
                });
                settings.export(file, manager, iter)
            } else {
                let function_names = std::slice::from_raw_parts(function_names, num_functions);
                let iter = functions.iter().zip(function_names).enumerate().filter_map(
                    |(i, (f, name))| {
                        let name = super::c_char_to_str(*name);
                        match f.get() {
                            Ok(f) => Some((f, name)),
                            Err(_) => {
                                invalid_func = Err((i, name));
                                None
                            }
                        }
                    },
                );
                settings.export_with_names(file, manager, iter)
            }
        })
    });
    handle_err(res, error)?;
    let invalid_func = invalid_func.map_err(|(i, name)| {
        if name.is_empty() {
            format!("function {i} is invalid")
        } else {
            format!("function {i} '{name}' is invalid")
        }
    });
    handle_err_or_init(invalid_func, error)
}

pub unsafe fn export_iter<CF: CFunction>(
    manager: CF::CManagerRef,
    path: &OsStr,
    functions: super::iter<CF>,
    settings: Option<&dddmp_export_settings_t>,
    error: *mut error_t,
) -> Option<()>
where
    for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
    for<'id> TermOfFunc<'id, CF::Function>: AsciiDisplay,
{
    let file = crate::util::handle_err(std::fs::File::create(path), error)?;
    let mut invalid_func = Ok(());
    let res = manager.get().with_manager_shared(|manager| {
        let iter = functions.into_iter().enumerate().filter_map(|(i, f)| {
            let res = f.get().ok();
            if res.is_none() {
                invalid_func = Err(i);
            }
            res
        });
        with_export_settings(settings, |settings| settings.export(file, manager, iter))
    });
    handle_err(res, error)?;
    let invalid_func = invalid_func.map_err(|i| format!("function {i} is invalid"));
    handle_err_or_init(invalid_func, error)
}

pub unsafe fn export_with_names_iter<CF: CFunction>(
    manager: CF::CManagerRef,
    path: &OsStr,
    functions: super::iter<super::named<CF>>,
    settings: Option<&dddmp_export_settings_t>,
    error: *mut error_t,
) -> Option<()>
where
    for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
    for<'id> TermOfFunc<'id, CF::Function>: AsciiDisplay,
{
    let file = handle_err(std::fs::File::create(path), error)?;
    let mut invalid_func = Ok(());
    let res = manager.get().with_manager_shared(|manager| {
        let iter = functions.into_iter().enumerate().filter_map(|(i, named)| {
            let name = named.name.to_str_lossy();
            match named.func.get() {
                Ok(f) => Some((f, name)),
                Err(_) => {
                    invalid_func = Err((i, name));
                    None
                }
            }
        });
        with_export_settings(settings, |settings| {
            settings.export_with_names(file, manager, iter)
        })
    });
    handle_err(res, error)?;
    let invalid_func =
        invalid_func.map_err(|(i, name)| format!("function {i} '{name}' is invalid"));
    handle_err_or_init(invalid_func, error)
}

pub unsafe fn visualize<CF: CFunction>(
    manager: CF::CManagerRef,
    diagram_name: &str,
    functions: *const CF,
    num_functions: usize,
    function_names: *const *const c_char,
    port: u16,
    error: *mut error_t,
) -> Option<()>
where
    for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
    for<'id> TermOfFunc<'id, CF::Function>: AsciiDisplay,
{
    let visualizer = manager.get().with_manager_shared(|manager| {
        if function_names.is_null() {
            Visualizer::new().add(
                diagram_name,
                manager,
                crate::util::slice_from_raw_parts(functions, num_functions)
                    .iter()
                    .filter_map(|f| f.get().ok()),
            )
        } else {
            Visualizer::new().add_with_names(
                diagram_name,
                manager,
                crate::util::slice_from_raw_parts(functions, num_functions)
                    .iter()
                    .zip(std::slice::from_raw_parts(function_names, num_functions))
                    .filter_map(|(f, name)| match f.get() {
                        Ok(f) => Some((f, super::c_char_to_str(*name))),
                        Err(_) => None,
                    }),
            )
        }
    });
    visualize_serve(visualizer, port, error)
}

pub unsafe fn visualize_iter<CF: CFunction>(
    manager: CF::CManagerRef,
    diagram_name: &str,
    functions: super::iter<CF>,
    port: u16,
    error: *mut error_t,
) -> Option<()>
where
    for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
    for<'id> TermOfFunc<'id, CF::Function>: AsciiDisplay,
{
    let visualizer = manager.get().with_manager_shared(|manager| {
        let iter = functions.into_iter().filter_map(|f| f.get().ok());
        Visualizer::new().add(diagram_name, manager, iter)
    });
    visualize_serve(visualizer, port, error)
}

pub unsafe fn visualize_with_names_iter<CF: CFunction>(
    manager: CF::CManagerRef,
    diagram_name: &str,
    functions: super::iter<super::named<CF>>,
    port: u16,
    error: *mut error_t,
) -> Option<()>
where
    for<'id> INodeOfFunc<'id, CF::Function>: HasLevel,
    for<'id> TermOfFunc<'id, CF::Function>: AsciiDisplay,
{
    let visualizer = manager.get().with_manager_shared(|manager| {
        let iter = functions
            .into_iter()
            .filter_map(|named| match named.func.get() {
                Ok(f) => Some((f, named.name.to_str_lossy())),
                Err(_) => None,
            });
        Visualizer::new().add_with_names(diagram_name, manager, iter)
    });
    visualize_serve(visualizer, port, error)
}

unsafe fn visualize_serve(
    mut visualizer: Visualizer,
    port: u16,
    error: *mut error_t,
) -> Option<()> {
    if port != 0 {
        visualizer = visualizer.port(port);
    }
    handle_err_or_init(visualizer.serve(), error)
}
