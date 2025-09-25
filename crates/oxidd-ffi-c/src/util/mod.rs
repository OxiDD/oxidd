use std::ffi::{c_char, OsStr};
use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::Range;

use oxidd::error::DuplicateVarName;
use oxidd::util::AllocResult;
use oxidd::{Manager, ManagerRef, VarNo};
use oxidd_core::function::{ETagOfFunc, INodeOfFunc, TermOfFunc};
use oxidd_core::util::Substitution;

pub mod dddmp;

mod interop;
pub use interop::*;

/// General-purpose error type with human-readable error messages
#[repr(C)]
pub struct error_t {
    /// A human-readable error message. This points to a null-terminated string.
    msg: *const c_char,
    /// Byte length of the message (excluding the trailing null byte)
    msg_len: usize,
    /// Capacity of the message character array. May be zero although `msg_len`
    /// is non-zero to mean that the string is borrowed. This field is for
    /// internal use only.
    _msg_cap: usize,
}

impl error_t {
    /// cbindgen:ignore
    pub const NONE: Self = error_t {
        msg: c"".as_ptr(),
        msg_len: 0,
        _msg_cap: 0,
    };
}

impl From<&'static std::ffi::CStr> for error_t {
    fn from(value: &'static std::ffi::CStr) -> Self {
        error_t {
            msg: value.as_ptr(),
            msg_len: value.count_bytes(),
            _msg_cap: 0,
        }
    }
}

impl From<String> for error_t {
    fn from(value: String) -> Self {
        let mut msg = ManuallyDrop::new(value.into_bytes());
        msg.push(0);
        error_t {
            msg: msg.as_ptr().cast(),
            msg_len: msg.len() - 1,
            _msg_cap: msg.capacity(),
        }
    }
}

impl From<std::io::Error> for error_t {
    fn from(value: std::io::Error) -> Self {
        value.to_string().into()
    }
}

impl Drop for error_t {
    fn drop(&mut self) {
        if self._msg_cap != 0 {
            debug_assert!(!self.msg.is_null());
            const { assert!(std::mem::size_of::<c_char>() == std::mem::size_of::<u8>()) };
            drop(unsafe { Vec::from_raw_parts(self.msg as *mut u8, 0, self._msg_cap) });
        }
    }
}

/// Deallocate the error
#[no_mangle]
pub extern "C" fn oxidd_error_free(error: error_t) {
    drop(error)
}

/// Handle a possible error by writing it to `target` (unless `target` is null)
///
/// # Safety
///
/// `target` must either be the null pointer or be valid for writes.
pub unsafe fn handle_err<T, E: Into<error_t>>(
    result: Result<T, E>,
    target: *mut error_t,
) -> Option<T> {
    match result {
        Ok(v) => Some(v),
        Err(e) => {
            if !target.is_null() {
                target.write(e.into());
            }
            None
        }
    }
}

pub unsafe fn handle_err_or_init<T, E: Into<error_t>>(
    result: Result<T, E>,
    target: *mut error_t,
) -> Option<T> {
    if target.is_null() {
        return result.ok();
    }
    match result {
        Ok(v) => {
            target.write(error_t::NONE);
            Some(v)
        }
        Err(e) => {
            target.write(e.into());
            None
        }
    }
}

/// cbindgen:ignore
pub const FUNC_UNWRAP_MSG: &str = "the given function is invalid";

pub trait CManagerRef: Copy {
    type ManagerRef: ManagerRef;

    unsafe fn get(self) -> ManuallyDrop<Self::ManagerRef>;

    unsafe fn set_var_name(self, var: VarNo, name: &str) -> VarNo {
        self.get()
            .with_manager_exclusive(|manager| match manager.set_var_name(var, name) {
                Ok(_) => VarNo::MAX,
                Err(err) => err.present_var,
            })
    }

    unsafe fn add_named_vars(
        self,
        names: *const *const c_char,
        count: VarNo,
    ) -> duplicate_var_name_result_t {
        self.get().with_manager_exclusive(|manager| {
            if names.is_null() {
                Ok(manager.add_vars(count)).into()
            } else {
                let names = std::slice::from_raw_parts(names, count as usize);
                manager
                    .add_named_vars(names.iter().map(|&ptr| c_char_to_str(ptr)))
                    .into()
            }
        })
    }
}

pub trait CFunction: Copy + From<Self::Function> + From<AllocResult<Self::Function>> {
    type CManagerRef: CManagerRef;
    type Function: for<'id> oxidd::Function<
        ManagerRef = <Self::CManagerRef as CManagerRef>::ManagerRef,
        Manager<'id> = <<Self::CManagerRef as CManagerRef>::ManagerRef as ManagerRef>::Manager<'id>,
    >;

    const INVALID: Self;

    unsafe fn get(self) -> AllocResult<ManuallyDrop<Self::Function>>;
}

/// Range of variables
#[derive(Clone, Copy)]
#[repr(C)]
pub struct var_no_range_t {
    /// The start
    pub start: VarNo,
    /// The end (exclusive)
    pub end: VarNo,
}

impl From<Range<VarNo>> for var_no_range_t {
    fn from(range: Range<VarNo>) -> Self {
        var_no_range_t {
            start: range.start,
            end: range.end,
        }
    }
}

/// Pair of a variable and a Boolean
#[derive(Clone, Copy)]
#[repr(C)]
pub struct var_no_bool_pair_t {
    /// The variable
    pub var: VarNo,
    /// The Boolean value
    pub val: bool,
}

/// Result of an operation adding variable names to the manager
///
/// Variable names in a manager are generally required to be unique. Upon an
/// attempt to add a name that is already in use, `present_var` refers to the
/// variable already using the name.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct duplicate_var_name_result_t {
    /// Range of variables that have successfully been added
    ///
    /// If no fresh variables were requested, this is simply the empty range
    /// starting and ending at the current variable count.
    pub added_vars: var_no_range_t,
    /// Number of the already present variable with `name` on error, or
    /// `(oxidd_var_no_t) -1` on success
    pub present_var: VarNo,
}

impl From<Result<Range<VarNo>, DuplicateVarName>> for duplicate_var_name_result_t {
    fn from(value: Result<Range<VarNo>, DuplicateVarName>) -> Self {
        match value {
            Ok(added_vars) => Self {
                added_vars: added_vars.into(),
                present_var: VarNo::MAX,
            },
            Err(err) => Self {
                added_vars: err.added_vars.into(),
                present_var: err.present_var,
            },
        }
    }
}

/// Boolean assignment
///
/// `data` is a pointer to `len` values. A value can be either 0 (false), 1
/// (true), or -1 (don't care).
#[repr(C)]
pub struct assignment_t {
    /// Pointer to the data array of length `len`
    ///
    /// Must never be modified
    pub data: *mut i8,
    /// Length of the assignment
    ///
    /// Must never be modified
    pub len: usize,
}

impl Drop for assignment_t {
    fn drop(&mut self) {
        const { assert!(!std::mem::needs_drop::<oxidd::util::OptBool>()) };
        if !self.data.is_null() {
            // Since `OptBool` values do not need to be dropped, we can just use
            // length 0. This way, we avoid UB in case integers without a
            // corresponding `OptBool` value are written to `data`.
            drop(unsafe {
                Vec::<oxidd::util::OptBool>::from_raw_parts(self.data.cast(), 0, self.len)
            })
        }
    }
}

impl assignment_t {
    /// cbindgen:ignore
    pub const EMPTY: Self = assignment_t {
        data: std::ptr::null_mut(),
        len: 0,
    };
}

impl From<Vec<oxidd::util::OptBool>> for assignment_t {
    fn from(mut vec: Vec<oxidd::util::OptBool>) -> Self {
        vec.shrink_to_fit();
        let len = vec.len();
        let data = vec.as_mut_ptr() as _;
        std::mem::forget(vec);
        assignment_t { data, len }
    }
}

/// Free the given assignment
///
/// `assignment.data` (i.e., the pointer value itself) and `assignment.length`
/// must be the values from the creation of the `assignment`.
///
/// In case `assignment.data` is `NULL`, this is a no-op.
#[no_mangle]
pub unsafe extern "C" fn oxidd_assignment_free(assignment: assignment_t) {
    drop(assignment);
}

pub struct Subst<'a, R> {
    pub id: u32,
    pub vars: &'a [VarNo],
    pub replacements: &'a [R],
}

impl<'a, R> Substitution for Subst<'a, R> {
    type Replacement = &'a R;

    #[inline]
    fn id(&self) -> u32 {
        self.id
    }
    #[inline]
    fn pairs(&self) -> impl ExactSizeIterator<Item = (VarNo, &'a R)> {
        self.vars.iter().copied().zip(self.replacements)
    }
}

/// A named decision diagram function
#[derive(Clone, Copy)]
#[repr(C)]
pub struct named<T> {
    /// The function
    pub func: T,
    /// The name
    pub name: str_t,
}

#[inline]
pub unsafe fn op1<CF: CFunction>(
    f: CF,
    op: impl FnOnce(&CF::Function) -> AllocResult<CF::Function>,
) -> CF {
    f.get().and_then(|f| op(&f)).into()
}

#[inline]
pub unsafe fn op2<CF: CFunction>(
    lhs: CF,
    rhs: CF,
    op: impl FnOnce(&CF::Function, &CF::Function) -> AllocResult<CF::Function>,
) -> CF {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

#[inline]
pub unsafe fn op2_var<CF: CFunction>(
    lhs: CF,
    rhs: VarNo,
    op: impl FnOnce(&CF::Function, VarNo) -> AllocResult<CF::Function>,
) -> CF {
    lhs.get().and_then(|lhs| op(&lhs, rhs)).into()
}

pub fn run_in_worker_pool<M: oxidd::HasWorkers>(
    manager: &M,
    callback: extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void,
    data: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    struct SendPtr(pub *mut std::ffi::c_void);
    // SAFETY: Sending a pointer to another thread is not unsafe, only using
    // that pointer for memory accesses may cause data races
    unsafe impl Send for SendPtr {}

    let data = SendPtr(data);
    oxidd::WorkerPool::install(oxidd::HasWorkers::workers(manager), move || {
        let data: SendPtr = data;
        SendPtr(callback(data.0))
    })
    .0
}

#[inline]
pub unsafe fn op3<CF: CFunction>(
    f1: CF,
    f2: CF,
    f3: CF,
    op: impl FnOnce(&CF::Function, &CF::Function, &CF::Function) -> AllocResult<CF::Function>,
) -> CF {
    f1.get()
        .and_then(|f1| op(&f1, &*f2.get()?, &*f3.get()?))
        .into()
}

pub unsafe fn dump_all_dot_path<CF: CFunction>(
    manager: CF::CManagerRef,
    path: &OsStr,
    functions: *const CF,
    function_names: *const *const c_char,
    num_function_names: usize,
    error: *mut error_t,
) -> Option<()>
where
    CF::Function: for<'id> oxidd_dump::dot::DotStyle<ETagOfFunc<'id, CF::Function>>,
    for<'id> INodeOfFunc<'id, CF::Function>: oxidd_core::HasLevel,
    for<'id> ETagOfFunc<'id, CF::Function>: fmt::Debug,
    for<'id> TermOfFunc<'id, CF::Function>: fmt::Display,
{
    let file = handle_err(std::fs::File::create(path), error)?;

    let (functions, function_names) =
        if !functions.is_null() && !function_names.is_null() && num_function_names != 0 {
            (
                std::slice::from_raw_parts(functions, num_function_names),
                std::slice::from_raw_parts(function_names, num_function_names),
            )
        } else {
            ([].as_slice(), [].as_slice())
        };

    let result = manager.get().with_manager_shared(|manager| {
        let functions = functions
            .iter()
            .zip(function_names)
            .filter_map(|(f, &name)| match f.get() {
                Ok(f) => Some((f, c_char_to_str(name))),
                Err(_) => None,
            });
        oxidd_dump::dot::dump_all(file, manager, functions)
    });
    handle_err_or_init(result, error)
}

pub unsafe fn dump_all_dot_path_iter<CF: CFunction>(
    manager: CF::CManagerRef,
    path: &OsStr,
    functions: iter<named<CF>>,
    error: *mut error_t,
) -> Option<()>
where
    CF::Function: for<'id> oxidd_dump::dot::DotStyle<ETagOfFunc<'id, CF::Function>>,
    for<'id> INodeOfFunc<'id, CF::Function>: oxidd_core::HasLevel,
    for<'id> ETagOfFunc<'id, CF::Function>: fmt::Debug,
    for<'id> TermOfFunc<'id, CF::Function>: fmt::Display,
{
    let file = handle_err(std::fs::File::create(path), error)?;

    let result = manager.get().with_manager_shared(|manager| {
        let functions = functions
            .into_iter()
            .filter_map(|named| match named.func.get() {
                Ok(f) => Some((f, named.name.to_str_lossy())),
                Err(_) => None,
            });
        oxidd_dump::dot::dump_all(file, manager, functions)
    });
    handle_err_or_init(result, error)
}
