use std::ffi::c_char;
use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use rustc_hash::FxHasher;

use oxidd::bdd::{BDDFunction, BDDManagerRef};
use oxidd::util::num::F64;
use oxidd::util::{AllocResult, OutOfMemory};
use oxidd::{
    BooleanFunction, BooleanFunctionQuant, Function, FunctionSubst, HasLevel, Manager, ManagerRef,
    RawFunction, RawManagerRef,
};

// We need to use the following items from `oxidd_core` since cbindgen only
// parses `oxidd_ffi_c` and `oxidd_core`:
use oxidd_core::function::BooleanOperator;
use oxidd_core::{LevelNo, VarNo};

use crate::util::{
    assignment_t, op1, op2, op3, var_no_bool_pair_t, CFunction, CManagerRef, FUNC_UNWRAP_MSG,
};

/// Reference to a manager of a simple binary decision diagram (BDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking `oxidd_bdd_manager_t`
/// instances as arguments do not take ownership of them (i.e., do not decrement
/// the reference counter). Returned `oxidd_bdd_manager_t` instances must
/// typically be deallocated using `oxidd_bdd_manager_unref()` to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct bdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl CManagerRef for bdd_manager_t {
    type ManagerRef = BDDManagerRef;

    #[inline]
    unsafe fn get(self) -> ManuallyDrop<BDDManagerRef> {
        assert!(!self._p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(BDDManagerRef::from_raw(self._p))
    }
}

/// Boolean function represented as a simple binary decision diagram (BDD)
///
/// This is essentially an optional reference to a BDD node. In case an
/// operation runs out of memory, it returns an invalid BDD function. Unless
/// explicitly specified otherwise, an `oxidd_bdd_t` parameter may be invalid to
/// permit "chaining" operations without explicit checks in between. In this
/// case, the returned BDD function is also invalid.
///
/// An instance of this type contributes to both the reference count of the
/// referenced node and the manager that stores this node. Unless explicitly
/// stated otherwise, functions taking `oxidd_bdd_t` instances as arguments do
/// not take ownership of them (i.e., do not decrement the reference counters).
/// Returned `oxidd_bdd_t` instances must typically be deallocated using
/// `oxidd_bdd_unref()` to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct bdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl CFunction for bdd_t {
    type CManagerRef = bdd_manager_t;
    type Function = BDDFunction;

    const INVALID: Self = Self {
        _p: std::ptr::null(),
        _i: 0,
    };

    #[inline]
    unsafe fn get(self) -> AllocResult<ManuallyDrop<BDDFunction>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(BDDFunction::from_raw(self._p, self._i)))
        }
    }
}

impl From<BDDFunction> for bdd_t {
    fn from(value: BDDFunction) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<BDDFunction>> for bdd_t {
    #[inline]
    fn from(value: AllocResult<BDDFunction>) -> Self {
        match value {
            Ok(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            Err(_) => Self::INVALID,
        }
    }
}
impl From<Option<BDDFunction>> for bdd_t {
    #[inline]
    fn from(value: Option<BDDFunction>) -> Self {
        match value {
            Some(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            None => Self::INVALID,
        }
    }
}

/// Pair of two `oxidd_bdd_t` instances
#[repr(C)]
pub struct bdd_pair_t {
    /// First component
    first: bdd_t,
    /// Second component
    second: bdd_t,
}

/// Create a new manager for a simple binary decision diagram (BDD)
///
/// @param  inner_node_capacity   Maximum number of inner nodes. `0` means no
///                               limit.
/// @param  apply_cache_capacity  Maximum number of apply cache entries. The
///                               apply cache implementation may round this up
///                               to the next power of two.
/// @param  threads               Number of threads for concurrent operations.
///                               `0` means automatic selection.
///
/// @returns  The BDD manager with reference count 1
#[no_mangle]
pub extern "C" fn oxidd_bdd_manager_new(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> bdd_manager_t {
    bdd_manager_t {
        _p: oxidd::bdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
    }
}

/// Increment the manager reference counter
///
/// No-op if `manager` is invalid.
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_ref(manager: bdd_manager_t) -> bdd_manager_t {
    if !manager._p.is_null() {
        std::mem::forget(manager.get().clone());
    }
    manager
}

/// Decrement the manager reference counter
///
/// No-op if `manager` is invalid.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_unref(manager: bdd_manager_t) {
    if !manager._p.is_null() {
        drop(BDDManagerRef::from_raw(manager._p));
    }
}

/// Increment the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// No-op if `f` is invalid.
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_ref(f: bdd_t) -> bdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// No-op if `f` is invalid.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_unref(f: bdd_t) {
    if !f._p.is_null() {
        drop(BDDFunction::from_raw(f._p, f._i));
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
pub unsafe extern "C" fn oxidd_bdd_manager_run_in_worker_pool(
    manager: bdd_manager_t,
    callback: extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void,
    data: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    crate::util::run_in_worker_pool(&*manager.get(), callback, data)
}

/// Get the manager that stores `f`
///
/// @param  f  A *valid* BDD function
///
/// @returns  A manager reference with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_containing_manager(f: bdd_t) -> bdd_manager_t {
    bdd_manager_t {
        _p: f.get().expect(FUNC_UNWRAP_MSG).manager_ref().into_raw(),
    }
}

/// Get the count of inner nodes stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_num_inner_nodes(manager: bdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}
/// Deprecated alias for `oxidd_bdd_manager_num_inner_nodes()`
///
/// @deprecated  Use `oxidd_bdd_manager_num_inner_nodes()` instead
#[deprecated(
    since = "0.11.0",
    note = "use oxidd_bdd_manager_num_inner_nodes instead"
)]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_num_inner_nodes(manager: bdd_manager_t) -> usize {
    oxidd_bdd_manager_num_inner_nodes(manager)
}

/// Get an approximate count of inner nodes stored in `manager`
///
/// For concurrent implementations, it may be much less costly to determine an
/// approximation of the inner node count that an accurate count
/// (`oxidd_bdd_manager_num_inner_nodes()`).
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  An approximate count of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_approx_num_inner_nodes(manager: bdd_manager_t) -> usize {
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
pub unsafe extern "C" fn oxidd_bdd_manager_num_vars(manager: bdd_manager_t) -> VarNo {
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
pub unsafe extern "C" fn oxidd_bdd_manager_num_named_vars(manager: bdd_manager_t) -> VarNo {
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
///                     number of variables (`oxidd_bdd_num_vars()`) must not
///                     overflow.
///
/// @returns  The range of new variable numbers
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_add_vars(
    manager: bdd_manager_t,
    additional: VarNo,
) -> crate::util::var_no_range_t {
    manager
        .get()
        .with_manager_exclusive(|manager| manager.add_vars(additional))
        .into()
}

/// Add named variables to the decision diagram in `manager`
///
/// This is a shorthand for `oxidd_bdd_add_vars()` and respective
/// `oxidd_bdd_set_var_name()` calls. More details can be found there.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  names    Pointer to an array of (at least) `count` variable names.
///                  Each name must be a null-terminated UTF-8 string or `NULL`.
///                  Both an empty string and `NULL` mean that the variable is
///                  unnamed. Passing `NULL` as an argument is also allowed, in
///                  which case this function is equivalent to
///                  `oxidd_bdd_manager_add_vars()`.
/// @param  count    Count of variables to add. Adding this to the current
///                  number of variables (`oxidd_bdd_num_vars()`) must not
///                  overflow.
///
/// @returns  Result indicating whether renaming was successful or which name is
///           already in use. The `name` field is either `NULL` or one of the
///           pointers of the `names` argument (i.e., it must not be deallocated
///           separately).
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_add_named_vars(
    manager: bdd_manager_t,
    names: *const *const c_char,
    count: VarNo,
) -> crate::util::duplicate_var_name_result_t {
    manager.add_named_vars(names, count)
}
/// Add named variables to the decision diagram in `manager`
///
/// This is a more flexible alternative to `oxidd_bdd_manager_add_named_vars()`
/// allowing a custom iterator.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  iter     Iterator yielding strings. An empty string means that the
///                  variable is unnamed.
///                  The iterator must not yield so many strings that the
///                  variable count (`oxidd_bdd_num_vars()`) overflows.
///
/// @returns  Result indicating whether renaming was successful or which name is
///           already in use. The `name` field is either `NULL` or one of the
///           pointers of the `names` argument (i.e., it must not be deallocated
///           separately).
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_add_named_vars_iter(
    manager: bdd_manager_t,
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
///                  (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The name, or `NULL` for unnamed variables. The caller receives
///           ownership of the allocation and should eventually deallocate the
///           memory using `free()` (from libc).
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_var_name(
    manager: bdd_manager_t,
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
///                  (`oxidd_bdd_manager_num_vars()`).
/// @param  string   Pointer to a C++ `std::string`. The name will be assigned
///                  to this string. For unnamed variables, the string will be
///                  empty.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_var_name_cpp(
    manager: bdd_manager_t,
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
/// retrieved via `oxidd_bdd_manager_name_to_var()` anymore.
///
/// Note that variable names are required to be unique. If labelling `var` as
/// `name` would violate uniqueness, then `var`'s name is left unchanged.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bdd_manager_num_vars()`).
/// @param  name     A null-terminated UTF-8 string to be used as the variable
///                  name.
///                  May also be `NULL` to represent an empty string.
///
/// @returns  `(oxidd_var_no_t) -1` on success, otherwise the variable which
///           already uses `name`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_set_var_name(
    manager: bdd_manager_t,
    var: VarNo,
    name: *const c_char,
) -> VarNo {
    manager.set_var_name(var, &crate::util::c_char_to_str(name))
}
/// Label `var` as `name`
///
/// An empty name means that the variable will become unnamed, and cannot be
/// retrieved via `oxidd_bdd_manager_name_to_var()` anymore.
///
/// Note that variable names are required to be unique. If labelling `var` as
/// `name` would violate uniqueness, then `var`'s name is left unchanged.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bdd_manager_num_vars()`).
/// @param  name     A UTF-8 string to be used as the variable name.
///                  May also be `NULL` to represent an empty string.
/// @param  len      Byte length of `name`
///
/// @returns  `(oxidd_var_no_t) -1` on success, otherwise the variable which
///           already uses `name`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_set_var_name_with_len(
    manager: bdd_manager_t,
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
pub unsafe extern "C" fn oxidd_bdd_manager_name_to_var(
    manager: bdd_manager_t,
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
pub unsafe extern "C" fn oxidd_bdd_manager_name_to_var_with_len(
    manager: bdd_manager_t,
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
///                  (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The corresponding level number
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_var_to_level(
    manager: bdd_manager_t,
    var: VarNo,
) -> LevelNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.var_to_level(var))
}

/// Perform garbage collection
///
/// This method looks for nodes that are neither referenced by a `bdd_function`
/// nor another node and removes them. The method works from top to bottom, so
/// if a node is only referenced by nodes that can be removed, this node will be
/// removed as well.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The count of nodes removed
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_gc(manager: bdd_manager_t) -> usize {
    manager.get().with_manager_shared(|manager| manager.gc())
}

/// Get the count of garbage collections
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The garbage collection count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_gc_count(manager: bdd_manager_t) -> u64 {
    manager
        .get()
        .with_manager_shared(|manager| manager.gc_count())
}

/// Get the variable for the given level
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  level    The level number. Must be less than the level/variable
///                  count (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The corresponding variable number
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_level_to_var(
    manager: bdd_manager_t,
    level: LevelNo,
) -> VarNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.level_to_var(level))
}

/// Get the Boolean function that is true if and only if `var` is true
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The BDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_var(manager: bdd_manager_t, var: VarNo) -> bdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BDDFunction::var(manager, var).into())
}

/// Get the Boolean function that is true if and only if `var` is false
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bdd_manager_num_vars()`).
///
/// @returns  The BDD function representing the negated variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_not_var(manager: bdd_manager_t, var: VarNo) -> bdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BDDFunction::not_var(manager, var).into())
}

/// Get the constant false BDD function `⊥`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_false(manager: bdd_manager_t) -> bdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BDDFunction::f(manager).into())
}

/// Get the constant true BDD function `⊤`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_true(manager: bdd_manager_t) -> bdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BDDFunction::t(manager).into())
}

/// Get the cofactors `(f_true, f_false)` of `f`
///
/// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the top-most
/// variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
/// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
///
/// Structurally, the cofactors are the children. If you only need one of the
/// cofactors, then use `oxidd_bdd_cofactor_true()` or
/// `oxidd_bdd_cofactor_false()`. These functions are slightly more efficient
/// then.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The pair `f_true` and `f_false` if `f` is valid and references an
///           inner node, otherwise a pair of invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_cofactors(f: bdd_t) -> bdd_pair_t {
    if let Ok(f) = f.get() {
        if let Some((t, e)) = f.cofactors() {
            return bdd_pair_t {
                first: t.into(),
                second: e.into(),
            };
        }
    }
    bdd_pair_t {
        first: bdd_t::INVALID,
        second: bdd_t::INVALID,
    }
}

/// Get the cofactor `f_true` of `f`
///
/// This function is slightly more efficient than `oxidd_bdd_cofactors()` in
/// case `f_false` is not needed. For a more detailed description, see
/// `oxidd_bdd_cofactors()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  `f_true` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_cofactor_true(f: bdd_t) -> bdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_true().into()
    } else {
        bdd_t::INVALID
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
pub unsafe extern "C" fn oxidd_bdd_cofactor_false(f: bdd_t) -> bdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_false().into()
    } else {
        bdd_t::INVALID
    }
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
pub unsafe extern "C" fn oxidd_bdd_node_level(f: bdd_t) -> LevelNo {
    if let Ok(f) = f.get() {
        f.with_manager_shared(|manager, edge| manager.get_node(edge).level())
    } else {
        LevelNo::MAX
    }
}
/// Deprecated alias for `oxidd_bdd_node_level()`
///
/// @deprecated  Use `oxidd_bdd_node_level()` instead
#[deprecated(since = "0.11.0", note = "use oxidd_bdd_node_level instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_level(f: bdd_t) -> LevelNo {
    oxidd_bdd_node_level(f)
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
pub unsafe extern "C" fn oxidd_bdd_node_var(f: bdd_t) -> VarNo {
    if let Ok(f) = f.get() {
        f.with_manager_shared(|manager, edge| match manager.get_node(edge) {
            oxidd::Node::Inner(n) => manager.level_to_var(n.level()),
            oxidd::Node::Terminal(_) => VarNo::MAX,
        })
    } else {
        VarNo::MAX
    }
}

/// Compute the BDD for the negation `¬f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_not(f: bdd_t) -> bdd_t {
    op1(f, BDDFunction::not)
}

/// Compute the BDD for the conjunction `lhs ∧ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_and(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::and)
}

/// Compute the BDD for the disjunction `lhs ∨ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_or(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::or)
}

/// Compute the BDD for the negated conjunction `lhs ⊼ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_nand(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::nand)
}

/// Compute the BDD for the negated disjunction `lhs ⊽ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_nor(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::nor)
}

/// Compute the BDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_xor(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::xor)
}

/// Compute the BDD for the equivalence `lhs ↔ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_equiv(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::equiv)
}

/// Compute the BDD for the implication `lhs → rhs` (or `lhs ≤ rhs`)
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_imp(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::imp)
}

/// Compute the BDD for the strict implication `lhs < rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_imp_strict(lhs: bdd_t, rhs: bdd_t) -> bdd_t {
    op2(lhs, rhs, BDDFunction::imp_strict)
}

/// Compute the BDD for the conditional “if `cond` then `then_case` else
/// `else_case`”
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_ite(cond: bdd_t, then_case: bdd_t, else_case: bdd_t) -> bdd_t {
    op3(cond, then_case, else_case, BDDFunction::ite)
}

/// Substitute `vars` in the BDD `f` by `replacement`
///
/// The substitution is performed in a parallel fashion, e.g.:
/// `(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y`
///
/// To create the substitution, use `oxidd_bdd_substitution_new()` and
/// `oxidd_bdd_substitution_add_pair()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_substitute(
    f: bdd_t,
    substitution: *const bdd_substitution_t,
) -> bdd_t {
    if substitution.is_null() {
        return bdd_t::INVALID;
    }
    f.get()
        .and_then(|f| {
            let subst = &*substitution;

            f.substitute(crate::util::Subst {
                id: subst.id,
                vars: &subst.vars,
                replacements: &subst.replacements,
            })
        })
        .into()
}

/// Substitution mapping variables to replacement functions
///
/// The intent behind this struct is to optimize the case where the same
/// substitution is applied multiple times. We would like to re-use apply
/// cache entries across these operations, and therefore, we need a compact
/// identifier for the substitution.
pub struct bdd_substitution_t {
    id: u32,
    vars: Vec<VarNo>,
    replacements: Vec<BDDFunction>,
}

/// Create a new substitution, capable of holding at least `capacity` pairs
/// without reallocating
///
/// Before applying the substitution via `oxidd_bdd_substitute()`, add all the
/// pairs via `oxidd_bdd_substitution_add_pair()`. Do not add more pairs after
/// the first `oxidd_bdd_substitute()` call with this substitution as it may
/// lead to incorrect results.
///
/// @returns  The substitution, to be freed via `oxidd_bdd_substitution_free()`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_substitution_new(capacity: usize) -> *mut bdd_substitution_t {
    Box::into_raw(Box::new(bdd_substitution_t {
        id: oxidd_core::util::new_substitution_id(),
        vars: Vec::with_capacity(capacity),
        replacements: Vec::with_capacity(capacity),
    }))
}

/// Add a pair of a variable `var` and a replacement function `replacement` to
/// `substitution`
///
/// `var` and `replacement` must be valid BDD functions. This function
/// increments the reference counters of both `var` and `replacement` (and they
/// are decremented by `oxidd_bdd_substitution_free()`). The order in which the
/// pairs are added is irrelevant.
///
/// Note that adding a new pair after applying the substitution may lead to
/// incorrect results when applying the substitution again.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_substitution_add_pair(
    substitution: *mut bdd_substitution_t,
    var: VarNo,
    replacement: bdd_t,
) {
    assert!(!substitution.is_null(), "substitution must not be NULL");
    let r = replacement
        .get()
        .expect("the replacement function is invalid");

    let subst = &mut *substitution;
    subst.vars.push(var);
    subst.replacements.push((*r).clone());
}

/// Free the given substitution
///
/// If `substitution` is `NULL`, this is a no-op.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_substitution_free(substitution: *mut bdd_substitution_t) {
    if !substitution.is_null() {
        drop(Box::from_raw(substitution))
    }
}

/// Compute the BDD for `f` with its variables restricted to constant values
/// according to `vars`
///
/// `vars` conceptually is a partial assignment, represented as the conjunction
/// of positive or negative literals, depending on whether the variable should
/// be mapped to true or false.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_restrict(f: bdd_t, vars: bdd_t) -> bdd_t {
    op2(f, vars, BDDFunction::restrict)
}

/// Compute the BDD for the universal quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// universal quantification. Universal quantification `∀x. f(…, x, …)` of a
/// Boolean function `f(…, x, …)` over a single variable `x` is
/// `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_forall(f: bdd_t, vars: bdd_t) -> bdd_t {
    op2(f, vars, BDDFunction::forall)
}

/// Compute the BDD for the existential quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// existential quantification. Existential quantification `∃x. f(…, x, …)` of
/// a Boolean function `f(…, x, …)` over a single variable `x` is
/// `f(…, 0, …) ∨ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_exists(f: bdd_t, vars: bdd_t) -> bdd_t {
    op2(f, vars, BDDFunction::exists)
}
/// Deprecated alias for `oxidd_bdd_exists()`
///
/// @deprecated  Use `oxidd_bdd_exists()` instead
#[deprecated(since = "0.10.0", note = "use oxidd_bdd_exists instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_exist(f: bdd_t, var: bdd_t) -> bdd_t {
    oxidd_bdd_exists(f, var)
}

/// Compute the BDD for the unique quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by unique
/// quantification. Unique quantification `∃!x. f(…, x, …)` of a Boolean
/// function `f(…, x, …)` over a single variable `x` is
/// `f(…, 0, …) ⊕ f(…, 1, …)`.
///
/// Unique quantification is also known as the
/// [Boolean difference](https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors)
/// or
/// [Boolean derivative](https://en.wikipedia.org/wiki/Boolean_differential_calculus).
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_unique(f: bdd_t, vars: bdd_t) -> bdd_t {
    op2(f, vars, BDDFunction::unique)
}

/// Combined application of `op` and `oxidd_bdd_forall()`
///
/// Passing a number as `op` that is not a valid `oxidd_boolean_operator`
/// results in undefined behavior.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function `∀ vars. lhs <op> rhs` with its own reference
///           count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_apply_forall(
    op: BooleanOperator,
    lhs: bdd_t,
    rhs: bdd_t,
    vars: bdd_t,
) -> bdd_t {
    lhs.get()
        .and_then(|f| f.apply_forall(op, &*rhs.get()?, &*vars.get()?))
        .into()
}

/// Combined application of `op` and `oxidd_bdd_exists()`
///
/// Passing a number as `op` that is not a valid `oxidd_boolean_operator`
/// results in undefined behavior.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function `∃ vars. lhs <op> rhs` with its own reference
///           count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_apply_exists(
    op: BooleanOperator,
    lhs: bdd_t,
    rhs: bdd_t,
    vars: bdd_t,
) -> bdd_t {
    lhs.get()
        .and_then(|f| f.apply_exists(op, &*rhs.get()?, &*vars.get()?))
        .into()
}
/// Deprecated alias for `oxidd_bdd_apply_exists()`
///
/// @deprecated  Use `oxidd_bdd_apply_exists()` instead
#[deprecated(since = "0.10.0", note = "use oxidd_bdd_apply_exists instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_apply_exist(
    op: BooleanOperator,
    lhs: bdd_t,
    rhs: bdd_t,
    vars: bdd_t,
) -> bdd_t {
    oxidd_bdd_apply_exists(op, lhs, rhs, vars)
}

/// Combined application of `op` and `oxidd_bdd_unique()`
///
/// Passing a number as `op` that is not a valid `oxidd_boolean_operator`
/// results in undefined behavior.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function `∃! vars. lhs <op> rhs` with its own reference
///           count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_apply_unique(
    op: BooleanOperator,
    lhs: bdd_t,
    rhs: bdd_t,
    vars: bdd_t,
) -> bdd_t {
    lhs.get()
        .and_then(|f| f.apply_unique(op, &*rhs.get()?, &*vars.get()?))
        .into()
}

/// Count nodes in `f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BDD function
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_node_count(f: bdd_t) -> usize {
    f.get().expect(FUNC_UNWRAP_MSG).node_count()
}

/// Check if `f` is satisfiable
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BDD function
///
/// @returns  `true` iff there is a satisfying assignment for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_satisfiable(f: bdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).satisfiable()
}

/// Check if `f` is valid
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BDD function
///
/// @returns  `true` iff there are only satisfying assignments for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_valid(f: bdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).valid()
}

/// Count the satisfying assignments of `f`, assuming `vars` input variables
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f     A *valid* BDD function
/// @param  vars  Number of input variables
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_sat_count_double(f: bdd_t, vars: LevelNo) -> f64 {
    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Pick a satisfying assignment
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BDD function
///
/// @returns  A satisfying assignment if there exists one. The i-th element
///           represents the value for the i-th variable. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           `oxidd_assignment_free()`.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_pick_cube(f: bdd_t) -> assignment_t {
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

/// Pick a satisfying assignment, represented as BDD
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
///           `f` is ⊥), ⊥ is returned.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_pick_cube_dd(f: bdd_t) -> bdd_t {
    f.get().and_then(|f| f.pick_cube_dd(|_, _, _| false)).into()
}

/// Pick a satisfying assignment, represented as BDD, using the literals in
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
pub unsafe extern "C" fn oxidd_bdd_pick_cube_dd_set(f: bdd_t, literal_set: bdd_t) -> bdd_t {
    op2(f, literal_set, BDDFunction::pick_cube_dd_set)
}

/// Evaluate the Boolean function `f` with arguments `args`
///
/// `args` determines the valuation for all variables in the function's domain.
/// The order is irrelevant (except that if the valuation for a variable is
/// given multiple times, the last value counts). Should there be a decision
/// node for a variable not part of the domain, then `false` is used as the
/// decision value.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f         A *valid* BDD function
/// @param  args      Array of pairs `(variable, value)`, where each variable
///                   number must be less than the number of variables in the
///                   manager
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_eval(
    f: bdd_t,
    args: *const var_no_bool_pair_t,
    num_args: usize,
) -> bool {
    let args = std::slice::from_raw_parts(args, num_args);

    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .with_manager_shared(|manager, edge| {
            BDDFunction::eval_edge(manager, edge, args.iter().map(|p| (p.var, p.val)))
        })
}

/// Dump the entire decision diagram represented by `manager` as Graphviz DOT
/// code to a file at `path`
///
/// If a file at `path` exists, it will be truncated, otherwise a new one will
/// be created.
///
/// This function optionally allows to name BDD functions. If `functions` and
/// `function_names` are non-null and `num_function_names` is non-zero, then
/// `functions` and `function_names` are assumed to point to an array of length
/// (at least) `num_function_names`. In this case, the i-th function is labeled
/// with the i-th function name.
///
/// The output may also include nodes that are not reachable from `functions`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager             The manager
/// @param  path                Path at which the DOT file should be written
/// @param  functions           Array of `num_function_names` BDD functions in
///                             `manager` to be labeled.
///                             May be `NULL`, in which case there will be no
///                             labels.
/// @param  function_names      Array of `num_function_names` null-terminated
///                             UTF-8 strings, each labelling the respective
///                             BDD function.
///                             May be `NULL`, in which case there will be no
///                             labels.
/// @param  num_function_names  Count of functions to be labeled
///
/// @returns  `true` on success
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_dump_all_dot_file(
    manager: bdd_manager_t,
    path: *const c_char,
    functions: *const bdd_t,
    function_names: *const *const c_char,
    num_function_names: usize,
) -> bool {
    crate::util::dump_all_dot_file(manager, path, functions, function_names, num_function_names)
}

/// Print statistics to stderr
#[no_mangle]
pub extern "C" fn oxidd_bdd_print_stats() {
    oxidd::bdd::print_stats();
}
