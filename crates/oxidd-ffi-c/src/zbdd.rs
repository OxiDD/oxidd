use std::ffi::c_char;
use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use rustc_hash::FxHasher;

use oxidd::util::num::F64;
use oxidd::util::{AllocResult, OutOfMemory};
use oxidd::zbdd::{ZBDDFunction, ZBDDManagerRef};
use oxidd::{
    BooleanFunction, BooleanVecSet, Function, HasLevel, Manager, ManagerRef, RawFunction,
    RawManagerRef,
};

// We need to use the following items from `oxidd_core` since cbindgen only
// parses `oxidd_ffi_c` and `oxidd_core`:
use oxidd_core::{LevelNo, VarNo};

use crate::util::{
    assignment_t, op1, op2, op2_var, op3, var_no_bool_pair_t, CFunction, CManagerRef,
    FUNC_UNWRAP_MSG,
};

/// Reference to a manager of a zero-suppressed decision diagram (ZBDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking `oxidd_zbdd_manager_t`
/// instances as arguments do not take ownership of them (i.e., do not decrement
/// the reference counter). Returned `oxidd_zbdd_manager_t` instances must
/// typically be deallocated using `oxidd_zbdd_manager_unref()` to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct zbdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl CManagerRef for zbdd_manager_t {
    type ManagerRef = ZBDDManagerRef;

    #[inline]
    unsafe fn get(self) -> ManuallyDrop<ZBDDManagerRef> {
        assert!(!self._p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(ZBDDManagerRef::from_raw(self._p))
    }
}

/// Boolean function represented as a zero-suppressed decision diagram (ZBDD)
///
/// This is essentially an optional reference to a ZBDD node. In case an
/// operation runs out of memory, it returns an invalid ZBDD function. Unless
/// explicitly specified otherwise, an `oxidd_zbdd_t` parameter may be invalid
/// to permit "chaining" operations without explicit checks in between. In this
/// case, the returned ZBDD function is also invalid.
///
/// An instance of this type contributes to both the reference count of the
/// referenced node and the manager that stores this node. Unless explicitly
/// stated otherwise, functions taking `oxidd_zbdd_t` instances as arguments do
/// not take ownership of them (i.e., do not decrement the reference counters).
/// Returned `oxidd_zbdd_t` instances must typically be deallocated using
/// `oxidd_zbdd_unref()` to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct zbdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl CFunction for zbdd_t {
    type CManagerRef = zbdd_manager_t;
    type Function = ZBDDFunction;

    const INVALID: Self = Self {
        _p: std::ptr::null(),
        _i: 0,
    };

    #[inline]
    unsafe fn get(self) -> AllocResult<ManuallyDrop<ZBDDFunction>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(ZBDDFunction::from_raw(self._p, self._i)))
        }
    }
}

impl From<ZBDDFunction> for zbdd_t {
    #[inline]
    fn from(value: ZBDDFunction) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<ZBDDFunction>> for zbdd_t {
    #[inline]
    fn from(value: AllocResult<ZBDDFunction>) -> Self {
        match value {
            Ok(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            Err(_) => Self::INVALID,
        }
    }
}
impl From<Option<ZBDDFunction>> for zbdd_t {
    #[inline]
    fn from(value: Option<ZBDDFunction>) -> Self {
        match value {
            Some(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            None => Self::INVALID,
        }
    }
}

/// Pair of two `oxidd_zbdd_t` instances
#[repr(C)]
pub struct zbdd_pair_t {
    /// First component
    first: zbdd_t,
    /// Second component
    second: zbdd_t,
}

/// Create a new manager for a zero-suppressed decision diagram (ZBDD)
///
/// @param  inner_node_capacity   Maximum number of inner nodes. `0` means no
///                               limit.
/// @param  apply_cache_capacity  Maximum number of apply cache entries. The
///                               apply cache implementation may round this up
///                               to the next power of two.
/// @param  threads               Number of threads for concurrent operations.
///                               `0` means automatic selection.
///
/// @returns  The ZBDD manager with reference count 1
#[no_mangle]
pub extern "C" fn oxidd_zbdd_manager_new(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> zbdd_manager_t {
    zbdd_manager_t {
        _p: oxidd::zbdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
    }
}

/// Increment the manager reference counter
///
/// No-op if `manager` is invalid.
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_ref(manager: zbdd_manager_t) -> zbdd_manager_t {
    if !manager._p.is_null() {
        std::mem::forget(manager.get().clone());
    }
    manager
}

/// Decrement the manager reference counter
///
/// No-op if `manager` is invalid.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_unref(manager: zbdd_manager_t) {
    if !manager._p.is_null() {
        drop(ZBDDManagerRef::from_raw(manager._p));
    }
}

/// Increment the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// No-op if `f` is invalid.
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_ref(f: zbdd_t) -> zbdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// No-op if `f` is invalid.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_unref(f: zbdd_t) {
    if !f._p.is_null() {
        drop(ZBDDFunction::from_raw(f._p, f._i));
    }
}

/// Execute `callback(data)` in the worker thread pool of `manager`
///
/// Recursive calls in the multithreaded apply algorithms are always executed
/// within the manager's thread pool, requiring a rather expensive context
/// switch if the apply algorithm is not called from within the thread pool. If
/// the algorithm takes long to execute anyway, this may not be important, but
/// for many small operations, this may easily make a difference by factors.
///
/// This function blocks until `callback(data)` has finished.
///
/// @returns  The result of calling `callback(data)`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_run_in_worker_pool(
    manager: zbdd_manager_t,
    callback: extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void,
    data: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    crate::util::run_in_worker_pool(&*manager.get(), callback, data)
}

/// Get the manager that stores `f`
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  A manager reference with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_containing_manager(f: zbdd_t) -> zbdd_manager_t {
    zbdd_manager_t {
        _p: f.get().expect(FUNC_UNWRAP_MSG).manager_ref().into_raw(),
    }
}

/// Get the count of inner nodes stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_num_inner_nodes(manager: zbdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}
/// Deprecated alias for `oxidd_zbdd_manager_num_inner_nodes()`
///
/// @deprecated  Use `oxidd_zbdd_manager_num_inner_nodes()` instead
#[deprecated(
    since = "0.11.0",
    note = "use oxidd_zbdd_manager_num_inner_nodes instead"
)]
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_num_inner_nodes(manager: zbdd_manager_t) -> usize {
    oxidd_zbdd_manager_num_inner_nodes(manager)
}

/// Get an approximate count of inner nodes stored in `manager`
///
/// For concurrent implementations, it may be much less costly to determine an
/// approximation of the inner node count that an accurate count
/// (`oxidd_zbdd_manager_num_inner_nodes()`).
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  An approximate count of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_approx_num_inner_nodes(
    manager: zbdd_manager_t,
) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.approx_num_inner_nodes())
}

/// Get the number of variables stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of variables
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_num_vars(manager: zbdd_manager_t) -> VarNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_vars())
}

/// Get the number of named variables stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of named variables
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_num_named_vars(manager: zbdd_manager_t) -> VarNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_named_vars())
}

/// Add `additional` unnamed variables to the decision diagram in `manager`
///
/// The new variables are added at the bottom of the variable order. More
/// precisely, the level number equals the variable number for each new
/// variable.
///
/// Note that some algorithms may assume that the domain of a function
/// represented by a decision diagram is just the set of all variables. In this
/// regard, adding variables can change the semantics of decision diagram nodes.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager     The manager
/// @param  additional  Count of variables to add. Adding this to the current
///                     number of variables (`oxidd_zbdd_num_vars()`) must not
///                     overflow.
///
/// @returns  The range of new variable numbers.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_add_vars(
    manager: zbdd_manager_t,
    additional: VarNo,
) -> crate::util::var_no_range_t {
    manager
        .get()
        .with_manager_exclusive(|manager| manager.add_vars(additional))
        .into()
}

/// Add named variables to the decision diagram in `manager`
///
/// This is a shorthand for `oxidd_zbdd_add_vars()` and respective
/// `oxidd_zbdd_set_var_name()` calls. More details can be found there.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  names    Pointer to an array of (at least) `count` variable names.
///                  Each name must be a null-terminated UTF-8 string or `NULL`.
///                  Both an empty string and `NULL` mean that the variable is
///                  unnamed. Passing `NULL` as an argument is also allowed, in
///                  which case this function is equivalent to
///                  `oxidd_zbdd_manager_add_vars()`.
/// @param  count    Count of variables to add. Adding this to the current
///                  number of variables (`oxidd_zbdd_num_vars()`) must not
///                  overflow.
///
/// @returns  Result indicating whether renaming was successful or which name is
///           already in use. The `name` field is either `NULL` or one of the
///           pointers of the `names` argument (i.e., it must not be deallocated
///           separately).
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_add_named_vars(
    manager: zbdd_manager_t,
    names: *const *const c_char,
    count: VarNo,
) -> crate::util::duplicate_var_name_result_t {
    manager.add_named_vars(names, count)
}
/// Add named variables to the decision diagram in `manager`
///
/// This is a more flexible alternative to `oxidd_zbdd_manager_add_named_vars()`
/// allowing a custom iterator.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  iter     Iterator yielding strings. An empty string means that the
///                  variable is unnamed.
///                  The iterator must not yield so many strings that the
///                  variable count (`oxidd_zbdd_num_vars()`) overflows.
///
/// @returns  Result indicating whether renaming was successful or which name is
///           already in use. The `name` field is either `NULL` or one of the
///           pointers of the `names` argument (i.e., it must not be deallocated
///           separately).
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_add_named_vars_iter(
    manager: zbdd_manager_t,
    iter: crate::util::str_iter_t,
) -> crate::util::duplicate_var_name_result_t {
    manager
        .get()
        .with_manager_exclusive(|manager| manager.add_named_vars(iter).into())
}

/// Get `var`'s name
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_zbdd_manager_num_vars()`).
///
/// @returns  The name, or `NULL` for unnamed variables. The caller receives
///           ownership of the allocation and should eventually deallocate the
///           memory using `free()` (from libc).
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_var_name(
    manager: zbdd_manager_t,
    var: VarNo,
) -> *const c_char {
    manager
        .get()
        .with_manager_shared(|manager| crate::util::to_c_str(manager.var_name(var)))
}
#[cfg(feature = "cpp")]
/// Get `var`'s name
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_zbdd_manager_num_vars()`).
/// @param  string   Pointer to a C++ `std::string`. The name will be assigned
///                  to this string. For unnamed variables, the string will be
///                  empty.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_var_name_cpp(
    manager: zbdd_manager_t,
    var: VarNo,
    string: *mut std::ffi::c_void,
) {
    manager.get().with_manager_shared(|manager| {
        let name = manager.var_name(var);
        crate::util::cpp_std_string_assign(string, name.as_ptr(), name.len());
    })
}

/// Label `var` as `name`
///
/// An empty name means that the variable will become unnamed, and cannot be
/// retrieved via `oxidd_zbdd_manager_name_to_var()` anymore.
///
/// Note that variable names are required to be unique. If labelling `var` as
/// `name` would violate uniqueness, then `var`'s name is left unchanged.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_zbdd_manager_num_vars()`).
/// @param  name     A null-terminated UTF-8 string to be used as the variable
///                  name.
///                  May also be `NULL` to represent an empty string.
///
/// @returns  `(oxidd_var_no_t) -1` on success, otherwise the variable which
///           already uses `name`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_set_var_name(
    manager: zbdd_manager_t,
    var: VarNo,
    name: *const c_char,
) -> VarNo {
    manager.set_var_name(var, &crate::util::c_char_to_str(name))
}
/// Label `var` as `name`bc
///
/// An empty name means that the variable will become unnamed, and cannot be
/// retrieved via `oxidd_zbdd_manager_name_to_var()` anymore.
///
/// Note that variable names are required to be unique. If labelling `var` as
/// `name` would violate uniqueness, then `var`'s name is left unchanged.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_zbdd_manager_num_vars()`).
/// @param  name     A UTF-8 string to be used as the variable name.
///                  May also be `NULL` to represent an empty string.
/// @param  len      Byte length of `name`
///
/// @returns  `(oxidd_var_no_t) -1` on success, otherwise the variable which
///           already uses `name`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_set_var_name_with_len(
    manager: zbdd_manager_t,
    var: VarNo,
    name: *const c_char,
    len: usize,
) -> VarNo {
    manager.set_var_name(var, &crate::util::c_char_array_to_str(name, len))
}

/// Get the variable number for the given variable name, if present
///
/// Note that you cannot retrieve unnamed variables. Calling this function with
/// an empty name will always result in `(oxidd_var_no_t) -1`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  name     Null-terminated UTF-8 string to look up.
///                  May also be `NULL` to represent an empty string.
///
/// @returns  The variable number if found, otherwise `(oxidd_var_no_t) -1`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_name_to_var(
    manager: zbdd_manager_t,
    name: *const c_char,
) -> VarNo {
    let name = crate::util::c_char_to_str(name);
    if name.is_empty() {
        return VarNo::MAX;
    }
    manager
        .get()
        .with_manager_shared(|manager| manager.name_to_var(name).unwrap_or(VarNo::MAX))
}
/// Get the variable number for the given variable name, if present
///
/// Note that you cannot retrieve unnamed variables. Calling this function with
/// an empty name will always result in `(oxidd_var_no_t) -1`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  name     UTF-8 string to look up.
///                  May also be `NULL` to represent an empty string.
/// @param  len      Byte length of `name`
///
/// @returns  The variable number if found, otherwise `(oxidd_var_no_t) -1`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_name_to_var_with_len(
    manager: zbdd_manager_t,
    name: *const c_char,
    len: usize,
) -> VarNo {
    let name = crate::util::c_char_array_to_str(name, len);
    if name.is_empty() {
        return VarNo::MAX;
    }
    manager
        .get()
        .with_manager_shared(|manager| manager.name_to_var(name).unwrap_or(VarNo::MAX))
}

/// Get the level for the given variable
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_zbdd_manager_num_vars()`).
///
/// @returns  The corresponding level number
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_var_to_level(
    manager: zbdd_manager_t,
    var: VarNo,
) -> LevelNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.var_to_level(var))
}

/// Get the variable for the given level
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  level    The level number. Must be less than the level/variable
///                  count (`oxidd_zbdd_manager_num_vars()`).
///
/// @returns  The corresponding variable number
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_level_to_var(
    manager: zbdd_manager_t,
    level: LevelNo,
) -> VarNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.level_to_var(level))
}

/// Get the singleton set {var}
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_zbdd_manager_num_vars()`).
///
/// @returns  The ZBDD function representing the set
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_singleton(manager: zbdd_manager_t, var: VarNo) -> zbdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| ZBDDFunction::singleton(manager, var).into())
}

/// Get the ZBDD Boolean function v for the singleton set {v}
///
/// `var` must be a singleton set.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  singleton  The singleton set {v}
///
/// @returns  The ZBDD Boolean function v
///
/// @deprecated  Use `oxidd_zbdd_var()` instead
#[deprecated(since = "0.11.0", note = "use oxidd_zbdd_var instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_var_boolean_function(singleton: zbdd_t) -> zbdd_t {
    let res = singleton.get().and_then(|singleton| {
        singleton.with_manager_shared(|manager, edge| {
            Ok(ZBDDFunction::from_edge(
                manager,
                #[allow(deprecated)]
                oxidd::zbdd::var_boolean_function(manager, edge)?,
            ))
        })
    });
    res.into()
}

/// Create a new ZBDD node at the level of `var` with the given `hi` and `lo`
/// edges
///
/// `var` must be a singleton set.
///
/// This function takes ownership of `hi` and `lo` (but not `var`).
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set referencing the new node
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_make_node(var: zbdd_t, hi: zbdd_t, lo: zbdd_t) -> zbdd_t {
    let res = var.get().and_then(|var| {
        let hi = ManuallyDrop::into_inner(hi.get()?);
        let lo = ManuallyDrop::into_inner(lo.get()?);
        var.with_manager_shared(|manager, var| {
            oxidd::zbdd::make_node(manager, var, hi.into_edge(manager), lo.into_edge(manager))
                .map(|e| ZBDDFunction::from_edge(manager, e))
        })
    });
    res.into()
}

/// Get the ZBDD set ∅
///
/// This function is equivalent to `oxidd_zbdd_false()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_empty(manager: zbdd_manager_t) -> zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::empty(manager).into())
}

/// Get the ZBDD set {∅}
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_base(manager: zbdd_manager_t) -> zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::base(manager).into())
}

/// Get the cofactors `(f_true, f_false)` of `f`
///
/// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the top-most
/// variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
/// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
///
/// Structurally, the cofactors are the children. If you only need one of the
/// cofactors, then use `oxidd_zbdd_cofactor_true()` or
/// `oxidd_zbdd_cofactor_false()`. These functions are slightly more efficient
/// then.
///
/// Note that the domain of f is 𝔹ⁿ⁺¹ while the domain of f<sub>true</sub> and
/// f<sub>false</sub> is 𝔹ⁿ. (Remember that, e.g., g(x₀) = x₀ and
/// g'(x₀, x₁) = x₀ have different representations as ZBDDs.)
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The pair `f_true` and `f_false` if `f` is valid and references an
///           inner node, otherwise a pair of invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_cofactors(f: zbdd_t) -> zbdd_pair_t {
    if let Ok(f) = f.get() {
        if let Some((t, e)) = f.cofactors() {
            return zbdd_pair_t {
                first: t.into(),
                second: e.into(),
            };
        }
    }
    zbdd_pair_t {
        first: zbdd_t::INVALID,
        second: zbdd_t::INVALID,
    }
}

/// Get the cofactor `f_true` of `f`
///
/// This function is slightly more efficient than `oxidd_zbdd_cofactors()` in
/// case `f_false` is not needed. For a more detailed description, see
/// `oxidd_zbdd_cofactors()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  `f_true` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_cofactor_true(f: zbdd_t) -> zbdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_true().into()
    } else {
        zbdd_t::INVALID
    }
}

/// Get the cofactor `f_false` of `f`
///
/// This function is slightly more efficient than `oxidd_bdd_cofactors()` in
/// case `f_true` is not needed. For a more detailed description, see
/// `oxidd_bdd_cofactors()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  `f_false` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_cofactor_false(f: zbdd_t) -> zbdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_false().into()
    } else {
        zbdd_t::INVALID
    }
}

/// Get the subset of `self` not containing `var`, formally
/// `{s ∈ self | {var} ∉ s}`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_subset0(set: zbdd_t, var: VarNo) -> zbdd_t {
    op2_var(set, var, ZBDDFunction::subset0)
}

/// Get the subsets of `set` containing `var` with `var` removed afterwards,
/// formally `{s ∖ {{var}} | s ∈ set ∧ var ∈ s}`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_subset1(set: zbdd_t, var: VarNo) -> zbdd_t {
    op2_var(set, var, ZBDDFunction::subset1)
}

/// Swap `subset0` and `subset1` with respect to `var`, formally
/// `{s ∪ {{var}} | s ∈ set ∧ {var} ∉ s} ∪ {s ∖ {{var}} | s ∈ set ∧ {var} ∈ s}`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_change(set: zbdd_t, var: VarNo) -> zbdd_t {
    op2_var(set, var, ZBDDFunction::change)
}

/// Compute the ZBDD for the union `lhs ∪ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_union(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::union)
}

/// Compute the ZBDD for the intersection `lhs ∩ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_intsec(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::intsec)
}

/// Compute the ZBDD for the set difference `lhs ∖ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_diff(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::diff)
}

/// Get the Boolean function that is true if and only if `var` is true
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The ZBDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_var(manager: zbdd_manager_t, var: VarNo) -> zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::var(manager, var).into())
}

/// Get the Boolean function that is true if and only if `var` is false
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The ZBDD function representing the negated variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_not_var(manager: zbdd_manager_t, var: VarNo) -> zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::not_var(manager, var).into())
}

/// Get the constant false ZBDD Boolean function `⊥`
///
/// This is an alias for `oxidd_zbdd_empty()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_false(manager: zbdd_manager_t) -> zbdd_t {
    oxidd_zbdd_empty(manager)
}

/// Get the constant true ZBDD Boolean function `⊤`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_true(manager: zbdd_manager_t) -> zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::t(manager).into())
}

/// Get the level of `f`'s underlying node
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The level of the underlying inner node, or `(oxidd_level_no_t) -1`
///           for terminals and invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_node_level(f: zbdd_t) -> LevelNo {
    if let Ok(f) = f.get() {
        f.with_manager_shared(|manager, edge| manager.get_node(edge).level())
    } else {
        LevelNo::MAX
    }
}
/// Deprecated alias for `oxidd_zbdd_node_level()`
///
/// @deprecated  Use `oxidd_zbdd_node_level()` instead
#[deprecated(since = "0.11.0", note = "use oxidd_zbdd_node_level instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_level(f: zbdd_t) -> LevelNo {
    oxidd_zbdd_node_level(f)
}
/// Get the variable number for `f`'s underlying node
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The variable number of the underlying inner node, or
///           `(oxidd_var_no_t) -1` for terminals and invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_node_var(f: zbdd_t) -> VarNo {
    if let Ok(f) = f.get() {
        f.with_manager_shared(|manager, edge| match manager.get_node(edge) {
            oxidd::Node::Inner(n) => manager.level_to_var(n.level()),
            oxidd::Node::Terminal(_) => VarNo::MAX,
        })
    } else {
        VarNo::MAX
    }
}

/// Compute the ZBDD for the negation `¬f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_not(f: zbdd_t) -> zbdd_t {
    op1(f, ZBDDFunction::not)
}

/// Compute the ZBDD for the conjunction `lhs ∧ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_and(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::and)
}

/// Compute the ZBDD for the disjunction `lhs ∨ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_or(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::or)
}

/// Compute the ZBDD for the negated conjunction `lhs ⊼ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_nand(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::nand)
}

/// Compute the ZBDD for the negated disjunction `lhs ⊽ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_nor(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::nor)
}

/// Compute the ZBDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_xor(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::xor)
}

/// Compute the ZBDD for the equivalence `lhs ↔ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_equiv(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::equiv)
}

/// Compute the ZBDD for the implication `lhs → rhs` (or `lhs ≤ rhs`)
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_imp(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::imp)
}

/// Compute the ZBDD for the strict implication `lhs < rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_imp_strict(lhs: zbdd_t, rhs: zbdd_t) -> zbdd_t {
    op2(lhs, rhs, ZBDDFunction::imp_strict)
}

/// Compute the ZBDD for the conditional “if `cond` then `then_case` else
/// `else_case`”
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_ite(
    cond: zbdd_t,
    then_case: zbdd_t,
    else_case: zbdd_t,
) -> zbdd_t {
    op3(cond, then_case, else_case, ZBDDFunction::ite)
}

/// Count nodes in `f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f     A *valid* ZBDD function
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_node_count(f: zbdd_t) -> usize {
    f.get().expect(FUNC_UNWRAP_MSG).node_count()
}

/// Check if `f` is satisfiable
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  `true` iff there is a satisfying assignment for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_satisfiable(f: zbdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).satisfiable()
}

/// Check if `f` is valid
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  `true` iff there are only satisfying assignments for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_valid(f: zbdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).valid()
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f     A *valid* ZBDD function
/// @param  vars  Number of input variables
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_sat_count_double(f: zbdd_t, vars: LevelNo) -> f64 {
    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Pick a satisfying assignment
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  A satisfying assignment if there exists one. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           `oxidd_assignment_free()`.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_pick_cube(f: zbdd_t) -> assignment_t {
    let f = f.get().expect(FUNC_UNWRAP_MSG);
    let res = f.pick_cube(|_, _, _| false);
    match res {
        Some(mut v) => {
            v.shrink_to_fit();
            let len = v.len();
            let data = v.as_mut_ptr() as _;
            std::mem::forget(v);
            assignment_t { data, len }
        }
        None => assignment_t {
            data: std::ptr::null_mut(),
            len: 0,
        },
    }
}

/// Pick a satisfying assignment, represented as ZBDD
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
///           `f` is ⊥), ⊥ is returned.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_pick_cube_dd(f: zbdd_t) -> zbdd_t {
    f.get().and_then(|f| f.pick_cube_dd(|_, _, _| false)).into()
}

/// Pick a satisfying assignment, represented as ZBDD, using the literals in
/// `literal_set` if there is a choice
///
/// `literal_set` is represented as a conjunction of literals. Whenever there is
/// a choice for a variable, it will be set to true if the variable has a
/// positive occurrence in `literal_set`, and set to false if it occurs negated
/// in `literal_set`. If the variable does not occur in `literal_set`, then it
/// will be left as don't care if possible, otherwise an arbitrary (not
/// necessarily random) choice will be performed.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
///           `f` is ⊥), ⊥ is returned.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_pick_cube_dd_set(f: zbdd_t, literal_set: zbdd_t) -> zbdd_t {
    op2(f, literal_set, ZBDDFunction::pick_cube_dd_set)
}

/// Evaluate the Boolean function `f` with arguments `args`
///
/// `args` determines the valuation for all variables in the function's domain.
/// The order is irrelevant (except that if the valuation for a variable is
/// given multiple times, the last value counts).
///
/// Note that the domain of the Boolean function represented by `f` is implicit
/// and may comprise a strict subset of the variables in the manager only. This
/// method assumes that the function's domain corresponds the set of variables
/// in `args`. Remember that for ZBDDs, the domain plays a crucial role for the
/// interpretation of decision diagram nodes as a Boolean function. This is in
/// contrast to, e.g., ordinary BDDs, where extending the domain does not affect
/// the evaluation result.
///
/// Should there be a decision node for a variable not part of the domain, then
/// `false` is used as the decision value.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f         A *valid* ZBDD function
/// @param  args      Array of pairs `(variable, value)`, where each variable
///                   number must be less than the number of variables in the
///                   manager
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_eval(
    f: zbdd_t,
    args: *const var_no_bool_pair_t,
    num_args: usize,
) -> bool {
    let args = std::slice::from_raw_parts(args, num_args);

    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .with_manager_shared(|manager, edge| {
            ZBDDFunction::eval_edge(manager, edge, args.iter().map(|p| (p.var, p.val)))
        })
}

/// Dump the entire decision diagram represented by `manager` as Graphviz DOT
/// code to a file at `path`
///
/// If a file at `path` exists, it will be truncated, otherwise a new one will
/// be created.
///
/// This function optionally allows to name ZBDD functions. If `functions` and
/// `function_names` are non-null and `num_function_names` is non-zero, then
/// `functions` and `function_names` are assumed to point to an array of length
/// (at least) `num_functions`. In this case, the i-th function is labeled
/// with the i-th function name.
///
/// The output may also include nodes that are not reachable from `functions`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager             The manager
/// @param  path                Path at which the DOT file should be written
/// @param  functions           Array of `num_function_names` ZBDD functions in
///                             `manager` to be labeled.
///                             May be `NULL`, in which case there will be no
///                             labels.
/// @param  function_names      Array of `num_function_names` null-terminated
///                             UTF-8 strings, each labelling the respective
///                             ZBDD function.
///                             May be `NULL`, in which case there will be no
///                             labels.
/// @param  num_function_names  Count of functions to be labeled
///
/// @returns  `true` on success
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_dump_all_dot_file(
    manager: zbdd_manager_t,
    path: *const c_char,
    functions: *const zbdd_t,
    function_names: *const *const c_char,
    num_function_names: usize,
) -> bool {
    crate::util::dump_all_dot_file(manager, path, functions, function_names, num_function_names)
}

/// Print statistics to stderr
#[no_mangle]
pub extern "C" fn oxidd_zbdd_print_stats() {
    oxidd::zbdd::print_stats();
}
