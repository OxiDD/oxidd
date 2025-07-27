use std::borrow::Cow;
use std::ffi::{c_char, CStr};
use std::fmt;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::Range;

use oxidd::error::DuplicateVarName;
use oxidd::util::AllocResult;
use oxidd::{Manager, ManagerRef, VarNo};
use oxidd_core::function::{ETagOfFunc, INodeOfFunc, TermOfFunc};
use oxidd_core::util::Substitution;

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

/// A borrowed string
///
/// Borrowed means that the receiver must not deallocate the character array
/// after use.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct str_t {
    /// Pointer to the character array
    ptr: *const c_char,
    /// Byte length of the string
    len: usize,
}

impl str_t {
    unsafe fn to_str_lossy<'a>(self) -> Cow<'a, str> {
        const { assert!(std::mem::size_of::<c_char>() == 1) };
        if self.ptr.is_null() {
            std::borrow::Cow::Borrowed("")
        } else {
            String::from_utf8_lossy(std::slice::from_raw_parts(self.ptr.cast(), self.len))
        }
    }
}

/// Optional `str_t`
#[derive(Clone, Copy)]
#[repr(C)]
pub struct opt_str_t {
    /// Whether a string is present
    is_some: bool,
    /// The string. May be uninitialized iff `is_some` is false.
    str: MaybeUninit<str_t>,
}

impl From<opt_str_t> for Option<str_t> {
    fn from(value: opt_str_t) -> Self {
        if value.is_some {
            Some(unsafe { value.str.assume_init() })
        } else {
            None
        }
    }
}

/// Estimation on the amount of remaining elements in an iterator
#[repr(C)]
pub struct size_hint_t {
    /// Lower bound
    pub lower: usize,
    /// Upper bound. `SIZE_MAX` is interpreted as no bound.
    pub upper: usize,
}

impl From<size_hint_t> for (usize, Option<usize>) {
    fn from(hint: size_hint_t) -> Self {
        let upper = if hint.upper == usize::MAX {
            None
        } else {
            Some(hint.upper)
        };
        (hint.lower, upper)
    }
}

/// Iterator over strings (`str_t`)
#[repr(C)]
pub struct str_iter_t {
    /// Function to get the next string
    ///
    /// If the function returns an `opt_str_t` with `is_some` equal to `false`,
    /// this signals the end of the iteration.
    ///
    /// Must not be `NULL`
    next: extern "C" fn(*mut std::ffi::c_void) -> opt_str_t,
    /// Function to get an estimate on remaining element count
    ///
    /// @see  `size_hint_t`
    size_hint: Option<extern "C" fn(*mut std::ffi::c_void) -> size_hint_t>,
    /// Context passed as an argument when calling `next` and `size_hint`
    context: *mut std::ffi::c_void,
}

impl Iterator for str_iter_t {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        let str: str_t = Option::from((self.next)(self.context))?;
        Some(unsafe { str.to_str_lossy() }.into_owned())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(f) = self.size_hint {
            f(self.context).into()
        } else {
            (0, None)
        }
    }
}

/// Equivalent to [`std::ffi::CStr::from_ptr()`] followed by
/// [`std::ffi::CStr::to_string_lossy()`], except that it allows `ptr` to be
/// null
pub unsafe fn c_char_to_str<'a>(ptr: *const c_char) -> std::borrow::Cow<'a, str> {
    if ptr.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        std::ffi::CStr::from_ptr(ptr).to_string_lossy()
    }
}

/// Equivalent to [`std::slice::from_raw_parts()`] followed by
/// [`String::from_utf8_lossy()`], except that it allows `ptr` to be
/// null
pub unsafe fn c_char_array_to_str<'a>(ptr: *const c_char, len: usize) -> std::borrow::Cow<'a, str> {
    const { assert!(std::mem::size_of::<c_char>() == 1) };

    if ptr.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        String::from_utf8_lossy(std::slice::from_raw_parts(ptr.cast(), len))
    }
}

pub fn to_c_str(str: &str) -> *const c_char {
    const { assert!(std::mem::size_of::<c_char>() == 1) };

    let len = str.len();
    if len == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let copy: *mut c_char = libc::malloc(len + 1).cast();
        copy.copy_from(str.as_ptr().cast(), len);
        copy.add(len).write(0);
        copy
    }
}

#[cfg(feature = "cpp")]
extern "C" {
    /// C++ `std::string::assign(char *, size_t)`
    ///
    /// `std::ffi::c_void` should be `std::string`, but this is difficult to
    /// realize with cbindgen
    #[link_name = "oxidd$interop$std$string$assign"]
    pub fn cpp_std_string_assign(string: *mut std::ffi::c_void, data: *const u8, len: usize);
    // implementation in `interop.cpp`
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

/// Free the given assignment
///
/// `assignment.data` (i.e., the pointer value itself) and `assignment.length`
/// must be the values from the creation of the `assignment`.
///
/// In case `assignment.data` is `NULL`, this is a no-op.
#[no_mangle]
pub unsafe extern "C" fn oxidd_assignment_free(assignment: assignment_t) {
    if !assignment.data.is_null() {
        const { assert!(!std::mem::needs_drop::<oxidd::util::OptBool>()) };
        // Since `OptBool` values do not need to be dropped, we can just use
        // length 0. This way it does not matter if
        drop(Vec::<oxidd::util::OptBool>::from_raw_parts(
            assignment.data.cast(),
            0,
            assignment.len,
        ))
    }
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

pub unsafe fn dump_all_dot_file<CF: CFunction>(
    manager: CF::CManagerRef,
    path: *const c_char,
    functions: *const CF,
    function_names: *const *const c_char,
    num_function_names: usize,
) -> bool
where
    CF::Function: for<'id> oxidd_dump::dot::DotStyle<ETagOfFunc<'id, CF::Function>>,
    for<'id> INodeOfFunc<'id, CF::Function>: oxidd_core::HasLevel,
    for<'id> ETagOfFunc<'id, CF::Function>: fmt::Debug,
    for<'id> TermOfFunc<'id, CF::Function>: fmt::Display,
{
    let Ok(path) = CStr::from_ptr(path).to_str() else {
        return false;
    };
    let Ok(file) = std::fs::File::create(path) else {
        return false;
    };

    manager.get().with_manager_shared(|manager| {
        // collect the functions and their corresponding names
        let (functions, function_names) =
            if !functions.is_null() && !function_names.is_null() && num_function_names != 0 {
                (
                    std::slice::from_raw_parts(functions, num_function_names),
                    std::slice::from_raw_parts(function_names, num_function_names),
                )
            } else {
                ([].as_slice(), [].as_slice())
            };

        oxidd_dump::dot::dump_all(
            file,
            manager,
            functions.iter().zip(function_names).map(|(f, &name)| {
                let f = f.get().expect("invalid DD function");
                (f, c_char_to_str(name))
            }),
        )
        .is_ok()
    })
}
