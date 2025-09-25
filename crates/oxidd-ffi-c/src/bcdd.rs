use std::ffi::c_char;
use std::hash::BuildHasherDefault;
use std::mem::{ManuallyDrop, MaybeUninit};

use rustc_hash::FxHasher;

use oxidd::bcdd::{BCDDFunction, BCDDManagerRef};
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

/// Reference to a manager of a binary decision diagram with complement edges
/// (BCDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking `oxidd_bcdd_manager_t`
/// instances as arguments do not take ownership of them (i.e., do not decrement
/// the reference counter). Returned `oxidd_bcdd_manager_t` instances must
/// typically be deallocated using `oxidd_bcdd_manager_unref()` to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct bcdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl CManagerRef for bcdd_manager_t {
    type ManagerRef = BCDDManagerRef;

    #[inline]
    unsafe fn get(self) -> ManuallyDrop<BCDDManagerRef> {
        assert!(!self._p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(BCDDManagerRef::from_raw(self._p))
    }
}

/// Boolean function represented as a binary decision diagram with complement
/// edges (BCDD)
///
/// This is essentially an optional (tagged) reference to a BCDD node. In case
/// an operation runs out of memory, it returns an invalid BCDD function. Unless
/// explicitly specified otherwise, an `oxidd_bcdd_t` parameter may be invalid
/// to permit "chaining" operations without explicit checks in between. In this
/// case, the returned BCDD function is also invalid.
///
/// An instance of this type contributes to both the reference count of the
/// referenced node and the manager that stores this node. Unless explicitly
/// stated otherwise, functions taking `oxidd_bcdd_t` instances as arguments do
/// not take ownership of them (i.e., do not decrement the reference counters).
/// Returned `oxidd_bcdd_t` instances must typically be deallocated using
/// `oxidd_bcdd_unref()` to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct bcdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl CFunction for bcdd_t {
    type CManagerRef = bcdd_manager_t;
    type Function = BCDDFunction;

    const INVALID: Self = Self {
        _p: std::ptr::null(),
        _i: 0,
    };

    #[inline]
    unsafe fn get(self) -> AllocResult<ManuallyDrop<BCDDFunction>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(BCDDFunction::from_raw(self._p, self._i)))
        }
    }
}

impl From<BCDDFunction> for bcdd_t {
    #[inline]
    fn from(value: BCDDFunction) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<BCDDFunction>> for bcdd_t {
    #[inline]
    fn from(value: AllocResult<BCDDFunction>) -> Self {
        match value {
            Ok(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            Err(_) => Self::INVALID,
        }
    }
}
impl From<Option<BCDDFunction>> for bcdd_t {
    #[inline]
    fn from(value: Option<BCDDFunction>) -> Self {
        match value {
            Some(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            None => Self::INVALID,
        }
    }
}

/// Pair of two `oxidd_bcdd_t` instances
#[repr(C)]
pub struct bcdd_pair_t {
    /// First component
    first: bcdd_t,
    /// Second component
    second: bcdd_t,
}

/// Create a new manager for a binary decision diagram with complement edges
/// (BCDD)
///
/// @param  inner_node_capacity   Maximum number of inner nodes. `0` means no
///                               limit.
/// @param  apply_cache_capacity  Maximum number of apply cache entries. The
///                               apply cache implementation may round this up
///                               to the next power of two.
/// @param  threads               Number of threads for concurrent operations.
///                               `0` means automatic selection.
///
/// @returns  The BCDD manager with reference count 1
#[no_mangle]
pub extern "C" fn oxidd_bcdd_manager_new(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> bcdd_manager_t {
    bcdd_manager_t {
        _p: oxidd::bcdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
    }
}

/// Increment the manager reference counter
///
/// No-op if `manager` is invalid.
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_ref(manager: bcdd_manager_t) -> bcdd_manager_t {
    if !manager._p.is_null() {
        std::mem::forget(manager.get().clone());
    }
    manager
}

/// Decrement the manager reference counter
///
/// No-op if `manager` is invalid.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_unref(manager: bcdd_manager_t) {
    if !manager._p.is_null() {
        drop(BCDDManagerRef::from_raw(manager._p));
    }
}

/// Increment the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// No-op if `f` is invalid.
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_ref(f: bcdd_t) -> bcdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// No-op if `f` is invalid.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_unref(f: bcdd_t) {
    if !f._p.is_null() {
        drop(BCDDFunction::from_raw(f._p, f._i));
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
pub unsafe extern "C" fn oxidd_bcdd_manager_run_in_worker_pool(
    manager: bcdd_manager_t,
    callback: extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void,
    data: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    crate::util::run_in_worker_pool(&*manager.get(), callback, data)
}

/// Get the manager that stores `f`
///
/// @param  f  A *valid* BCDD function
///
/// @returns  A manager reference with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_containing_manager(f: bcdd_t) -> bcdd_manager_t {
    bcdd_manager_t {
        _p: f.get().expect(FUNC_UNWRAP_MSG).manager_ref().into_raw(),
    }
}

/// Get the count of inner nodes stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_num_inner_nodes(manager: bcdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}
/// Deprecated alias for `oxidd_bcdd_manager_num_inner_nodes()`
///
/// @deprecated  Use `oxidd_bcdd_manager_num_inner_nodes()` instead
#[deprecated(
    since = "0.11.0",
    note = "use oxidd_bcdd_manager_num_inner_nodes instead"
)]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_num_inner_nodes(manager: bcdd_manager_t) -> usize {
    oxidd_bcdd_manager_num_inner_nodes(manager)
}

/// Get an approximate count of inner nodes stored in `manager`
///
/// For concurrent implementations, it may be much less costly to determine an
/// approximation of the inner node count that an accurate count
/// (`oxidd_bcdd_manager_num_inner_nodes()`).
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  An approximate count of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_approx_num_inner_nodes(
    manager: bcdd_manager_t,
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
pub unsafe extern "C" fn oxidd_bcdd_manager_num_vars(manager: bcdd_manager_t) -> VarNo {
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
pub unsafe extern "C" fn oxidd_bcdd_manager_num_named_vars(manager: bcdd_manager_t) -> VarNo {
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
///                     number of variables (`oxidd_bcdd_num_vars()`) must not
///                     overflow.
///
/// @returns  The range of new variable numbers
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_add_vars(
    manager: bcdd_manager_t,
    additional: VarNo,
) -> crate::util::var_no_range_t {
    manager
        .get()
        .with_manager_exclusive(|manager| manager.add_vars(additional))
        .into()
}

/// Add named variables to the decision diagram in `manager`
///
/// This is a shorthand for `oxidd_bcdd_add_vars()` and respective
/// `oxidd_bcdd_set_var_name()` calls. More details can be found there.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  names    Pointer to an array of (at least) `count` variable names.
///                  Each name must be a null-terminated UTF-8 string or `NULL`.
///                  Both an empty string and `NULL` mean that the variable is
///                  unnamed. Passing `NULL` as an argument is also allowed, in
///                  which case this function is equivalent to
///                  `oxidd_bcdd_manager_add_vars()`.
/// @param  count    Count of variables to add. Adding this to the current
///                  number of variables (`oxidd_bcdd_num_vars()`) must not
///                  overflow.
///
/// @returns  Result indicating whether renaming was successful or which name is
///           already in use. The `name` field is either `NULL` or one of the
///           pointers of the `names` argument (i.e., it must not be deallocated
///           separately).
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_add_named_vars(
    manager: bcdd_manager_t,
    names: *const *const c_char,
    count: VarNo,
) -> crate::util::duplicate_var_name_result_t {
    manager.add_named_vars(names, count)
}
/// Add named variables to the decision diagram in `manager`
///
/// This is a more flexible alternative to `oxidd_bcdd_manager_add_named_vars()`
/// allowing a custom iterator.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  iter     Iterator yielding strings. An empty string means that the
///                  variable is unnamed.
///                  The iterator must not yield so many strings that the
///                  variable count (`oxidd_bcdd_num_vars()`) overflows.
///
/// @returns  Result indicating whether renaming was successful or which name is
///           already in use. The `name` field is either `NULL` or one of the
///           pointers of the `names` argument (i.e., it must not be deallocated
///           separately).
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_add_named_vars_iter(
    manager: bcdd_manager_t,
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
///                  (`oxidd_bcdd_manager_num_vars()`).
/// @param  len      Output parameter for the string length.
///
/// @returns  The name, or `NULL` for unnamed variables. The caller receives
///           ownership of the allocation and should eventually deallocate the
///           memory using `free()` (from libc).
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_var_name(
    manager: bcdd_manager_t,
    var: VarNo,
    len: Option<&mut MaybeUninit<usize>>,
) -> *const c_char {
    manager.get().with_manager_shared(|manager| {
        let name = manager.var_name(var);
        if let Some(len) = len {
            len.write(name.len());
        }
        crate::util::to_c_str(name)
    })
}
#[cfg(feature = "cpp")]
/// Get `var`'s name
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bcdd_manager_num_vars()`).
/// @param  string   Pointer to a C++ `std::string`. The name will be assigned
///                  to this string. For unnamed variables, the string will be
///                  empty.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_var_name_cpp(
    manager: bcdd_manager_t,
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
/// retrieved via `oxidd_bcdd_manager_name_to_var()` anymore.
///
/// Note that variable names are required to be unique. If labelling `var` as
/// `name` would violate uniqueness, then `var`'s name is left unchanged.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bcdd_manager_num_vars()`).
/// @param  name     A UTF-8 string to be used as the variable name.
///                  May also be `NULL` to represent an empty string.
/// @param  len      Length of `name` in bytes (excluding any trailing null
///                  byte)
///
/// @returns  `(oxidd_var_no_t) -1` on success, otherwise the variable which
///           already uses `name`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_set_var_name(
    manager: bcdd_manager_t,
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
/// @param  name     UTF-8 string to look up.
///                  May also be `NULL` to represent an empty string.
/// @param  len      Length of `name` in bytes (excluding any trailing null
///                  byte)
///
/// @returns  The variable number if found, otherwise `(oxidd_var_no_t) -1`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_name_to_var(
    manager: bcdd_manager_t,
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
///                  (`oxidd_bcdd_manager_num_vars()`).
///
/// @returns  The corresponding level number
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_var_to_level(
    manager: bcdd_manager_t,
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
///                  count (`oxidd_bcdd_manager_num_vars()`).
///
/// @returns  The corresponding variable number
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_level_to_var(
    manager: bcdd_manager_t,
    level: LevelNo,
) -> VarNo {
    manager
        .get()
        .with_manager_shared(|manager| manager.level_to_var(level))
}

/// Perform garbage collection
///
/// This method looks for nodes that are neither referenced by a `bcdd_function`
/// nor another node and removes them. The method works from top to bottom, so
/// if a node is only referenced by nodes that can be removed, this node will be
/// removed as well.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The count of nodes removed
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_gc(manager: bcdd_manager_t) -> usize {
    manager.get().with_manager_shared(|manager| manager.gc())
}

/// Get the count of garbage collections
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The garbage collection count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_gc_count(manager: bcdd_manager_t) -> u64 {
    manager
        .get()
        .with_manager_shared(|manager| manager.gc_count())
}

/// Get the Boolean function that is true if and only if `var` is true
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bcdd_manager_num_vars()`).
///
/// @returns  The BCDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_var(manager: bcdd_manager_t, var: VarNo) -> bcdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BCDDFunction::var(manager, var).into())
}

/// Get the Boolean function that is true if and only if `var` is false
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @param  manager  The manager
/// @param  var      The variable number. Must be less than the variable count
///                  (`oxidd_bcdd_manager_num_vars()`).
///
/// @returns  The BCDD function representing the negated variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_not_var(manager: bcdd_manager_t, var: VarNo) -> bcdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BCDDFunction::not_var(manager, var).into())
}

/// Get the constant false BCDD function `⊥`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_false(manager: bcdd_manager_t) -> bcdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BCDDFunction::f(manager).into())
}

/// Get the constant true BCDD function `⊤`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_true(manager: bcdd_manager_t) -> bcdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BCDDFunction::t(manager).into())
}

/// Get the cofactors `(f_true, f_false)` of `f`
///
/// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the top-most
/// variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
/// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
///
/// Structurally, the cofactors are the children with edge tags are adjusted
/// accordingly. If you only need one of the cofactors, then use
/// `oxidd_bcdd_cofactor_true()` or `oxidd_bcdd_cofactor_false()`. These
/// functions are slightly more efficient then.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  The pair `f_true` and `f_false` if `f` is valid and references an
///           inner node, otherwise a pair of invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_cofactors(f: bcdd_t) -> bcdd_pair_t {
    if let Ok(f) = f.get() {
        if let Some((t, e)) = f.cofactors() {
            return bcdd_pair_t {
                first: t.into(),
                second: e.into(),
            };
        }
    }
    bcdd_pair_t {
        first: bcdd_t::INVALID,
        second: bcdd_t::INVALID,
    }
}

/// Get the cofactor `f_true` of `f`
///
/// This function is slightly more efficient than `oxidd_bcdd_cofactors()` in
/// case `f_false` is not needed.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  `f_true` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_cofactor_true(f: bcdd_t) -> bcdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_true().into()
    } else {
        bcdd_t::INVALID
    }
}

/// Get the cofactor `f_false` of `f`
///
/// This function is slightly more efficient than `oxidd_bcdd_cofactors()` in
/// case `f_true` is not needed.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Time complexity: O(1)
///
/// @returns  `f_false` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_cofactor_false(f: bcdd_t) -> bcdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_false().into()
    } else {
        bcdd_t::INVALID
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
pub unsafe extern "C" fn oxidd_bcdd_node_level(f: bcdd_t) -> LevelNo {
    if let Ok(f) = f.get() {
        f.with_manager_shared(|manager, edge| manager.get_node(edge).level())
    } else {
        LevelNo::MAX
    }
}
/// Deprecated alias for `oxidd_bcdd_node_level()`
///
/// @deprecated  Use `oxidd_bcdd_node_level()` instead
#[deprecated(since = "0.11.0", note = "use oxidd_bcdd_node_level instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_level(f: bcdd_t) -> LevelNo {
    oxidd_bcdd_node_level(f)
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
pub unsafe extern "C" fn oxidd_bcdd_node_var(f: bcdd_t) -> VarNo {
    if let Ok(f) = f.get() {
        f.with_manager_shared(|manager, edge| match manager.get_node(edge) {
            oxidd::Node::Inner(n) => manager.level_to_var(n.level()),
            oxidd::Node::Terminal(_) => VarNo::MAX,
        })
    } else {
        VarNo::MAX
    }
}

/// Compute the BCDD for the negation `¬f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_not(f: bcdd_t) -> bcdd_t {
    op1(f, BCDDFunction::not)
}

/// Compute the BCDD for the conjunction `lhs ∧ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_and(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::and)
}

/// Compute the BCDD for the disjunction `lhs ∨ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_or(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::or)
}

/// Compute the BCDD for the negated conjunction `lhs ⊼ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_nand(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::nand)
}

/// Compute the BCDD for the negated disjunction `lhs ⊽ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_nor(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::nor)
}

/// Compute the BCDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_xor(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::xor)
}

/// Compute the BCDD for the equivalence `lhs ↔ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_equiv(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::equiv)
}

/// Compute the BCDD for the implication `lhs → rhs` (or `lhs ≤ rhs`)
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_imp(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::imp)
}

/// Compute the BCDD for the strict implication `lhs < rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_imp_strict(lhs: bcdd_t, rhs: bcdd_t) -> bcdd_t {
    op2(lhs, rhs, BCDDFunction::imp_strict)
}

/// Compute the BCDD for the conditional “if `cond` then `then_case` else
/// `else_case`”
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_ite(
    cond: bcdd_t,
    then_case: bcdd_t,
    else_case: bcdd_t,
) -> bcdd_t {
    op3(cond, then_case, else_case, BCDDFunction::ite)
}

/// Substitute variables in the BCDD `f` according to `substitution`
///
/// The substitution is performed in a parallel fashion, e.g.:
/// `(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y`
///
/// To create the substitution, use `oxidd_bcdd_substitution_new()` and
/// `oxidd_bcdd_substitution_add_pair()`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_substitute(
    f: bcdd_t,
    substitution: *const bcdd_substitution_t,
) -> bcdd_t {
    if substitution.is_null() {
        return bcdd_t::INVALID;
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
pub struct bcdd_substitution_t {
    id: u32,
    vars: Vec<VarNo>,
    replacements: Vec<BCDDFunction>,
}

/// Create a new substitution, capable of holding at least `capacity` pairs
/// without reallocating
///
/// Before applying the substitution via `oxidd_bcdd_substitute()`, add all the
/// pairs via `oxidd_bcdd_substitution_add_pair()`. Do not add more pairs after
/// the first `oxidd_bcdd_substitute()` call with this substitution as it may
/// lead to incorrect results.
///
/// @returns  The substitution, to be freed via `oxidd_bcdd_substitution_free()`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_substitution_new(capacity: usize) -> *mut bcdd_substitution_t {
    Box::into_raw(Box::new(bcdd_substitution_t {
        id: oxidd_core::util::new_substitution_id(),
        vars: Vec::with_capacity(capacity),
        replacements: Vec::with_capacity(capacity),
    }))
}

/// Add a pair of a variable `var` and a replacement function `replacement` to
/// `substitution`
///
/// `var` and `replacement` must be valid BCDD functions. This function
/// increments the reference counters of both `var` and `replacement` (and they
/// are decremented by `oxidd_bcdd_substitution_free()`). The order in which the
/// pairs are added is irrelevant.
///
/// Note that adding a new pair after applying the substitution may lead to
/// incorrect results when applying the substitution again.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_substitution_add_pair(
    substitution: *mut bcdd_substitution_t,
    var: VarNo,
    replacement: bcdd_t,
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
pub unsafe extern "C" fn oxidd_bcdd_substitution_free(substitution: *mut bcdd_substitution_t) {
    if !substitution.is_null() {
        drop(Box::from_raw(substitution))
    }
}

/// Compute the BCDD for `f` with its variables restricted to constant values
/// according to `vars`
///
/// `vars` conceptually is a partial assignment, represented as the conjunction
/// of positive or negative literals, depending on whether the variable should
/// be mapped to true or false.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_restrict(f: bcdd_t, vars: bcdd_t) -> bcdd_t {
    op2(f, vars, BCDDFunction::restrict)
}

/// Compute the BCDD for the universal quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// universal quantification. Universal quantification `∀x. f(…, x, …)` of a
/// Boolean function `f(…, x, …)` over a single variable `x` is
/// `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_forall(f: bcdd_t, var: bcdd_t) -> bcdd_t {
    op2(f, var, BCDDFunction::forall)
}

/// Compute the BCDD for the existential quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// existential quantification. Existential quantification `∃x. f(…, x, …)` of
/// a Boolean function `f(…, x, …)` over a single variable `x` is
/// `f(…, 0, …) ∨ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_exists(f: bcdd_t, var: bcdd_t) -> bcdd_t {
    op2(f, var, BCDDFunction::exists)
}
/// Deprecated alias for `oxidd_bcdd_exists()`
///
/// @deprecated  Use `oxidd_bcdd_exists()` instead
#[deprecated(since = "0.10.0", note = "use oxidd_bcdd_exists instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_exist(f: bcdd_t, var: bcdd_t) -> bcdd_t {
    oxidd_bcdd_exists(f, var)
}

/// Compute the BCDD for the unique quantification of `f` over `vars`
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
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_unique(f: bcdd_t, var: bcdd_t) -> bcdd_t {
    op2(f, var, BCDDFunction::unique)
}

/// Combined application of `op` and `oxidd_bcdd_forall()`
///
/// Passing a number as `op` that is not a valid `oxidd_boolean_operator`
/// results in undefined behavior.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function `∀ vars. lhs <op> rhs` with its own reference
///           count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_apply_forall(
    op: BooleanOperator,
    lhs: bcdd_t,
    rhs: bcdd_t,
    vars: bcdd_t,
) -> bcdd_t {
    lhs.get()
        .and_then(|f| f.apply_forall(op, &*rhs.get()?, &*vars.get()?))
        .into()
}

/// Combined application of `op` and `oxidd_bcdd_exists()`
///
/// Passing a number as `op` that is not a valid `oxidd_boolean_operator`
/// results in undefined behavior.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function `∃ vars. lhs <op> rhs` with its own reference
///           count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_apply_exists(
    op: BooleanOperator,
    lhs: bcdd_t,
    rhs: bcdd_t,
    vars: bcdd_t,
) -> bcdd_t {
    lhs.get()
        .and_then(|f| f.apply_exists(op, &*rhs.get()?, &*vars.get()?))
        .into()
}
/// Deprecated alias for `oxidd_bcdd_apply_exists()`
///
/// @deprecated  Use `oxidd_bcdd_apply_exists()` instead
#[deprecated(since = "0.10.0", note = "use oxidd_bcdd_apply_exists instead")]
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_apply_exist(
    op: BooleanOperator,
    lhs: bcdd_t,
    rhs: bcdd_t,
    vars: bcdd_t,
) -> bcdd_t {
    oxidd_bcdd_apply_exists(op, lhs, rhs, vars)
}

/// Combined application of `op` and `oxidd_bcdd_unique()`
///
/// Passing a number as `op` that is not a valid `oxidd_boolean_operator`
/// results in undefined behavior.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BCDD function `∃! vars. lhs <op> rhs` with its own reference
///           count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_apply_unique(
    op: BooleanOperator,
    lhs: bcdd_t,
    rhs: bcdd_t,
    vars: bcdd_t,
) -> bcdd_t {
    lhs.get()
        .and_then(|f| f.apply_unique(op, &*rhs.get()?, &*vars.get()?))
        .into()
}

/// Count nodes in `f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_node_count(f: bcdd_t) -> usize {
    f.get().expect(FUNC_UNWRAP_MSG).node_count()
}

/// Check if `f` is satisfiable
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  `true` iff there is a satisfying assignment for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_satisfiable(f: bcdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).satisfiable()
}

/// Check if `f` is valid
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  `true` iff there are only satisfying assignments for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_valid(f: bcdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).valid()
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f     A *valid* BCDD function
/// @param  vars  Number of input variables
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_sat_count_double(f: bcdd_t, vars: LevelNo) -> f64 {
    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Pick a satisfying assignment
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  A satisfying assignment if there exists one. The i-th element
///           represents the value for the i-th variable. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           `oxidd_assignment_free()`.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_pick_cube(f: bcdd_t) -> assignment_t {
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

/// Pick a satisfying assignment, represented as BCDD
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
///           `f` is ⊥), ⊥ is returned.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_pick_cube_dd(f: bcdd_t) -> bcdd_t {
    f.get().and_then(|f| f.pick_cube_dd(|_, _, _| false)).into()
}

/// Pick a satisfying assignment, represented as BCDD, using the literals in
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
pub unsafe extern "C" fn oxidd_bcdd_pick_cube_dd_set(f: bcdd_t, literal_set: bcdd_t) -> bcdd_t {
    op2(f, literal_set, BCDDFunction::pick_cube_dd_set)
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
/// @param  f         A *valid* BCDD function
/// @param  args      Array of pairs `(variable, value)`, where each variable
///                   number must be less than the number of variables in the
///                   manager
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_eval(
    f: bcdd_t,
    args: *const var_no_bool_pair_t,
    num_args: usize,
) -> bool {
    let args = std::slice::from_raw_parts(args, num_args);

    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .with_manager_shared(|manager, edge| {
            BCDDFunction::eval_edge(manager, edge, args.iter().map(|p| (p.var, p.val)))
        })
}

/// Dump the entire decision diagram represented by `manager` as Graphviz DOT
/// code to a file at `path`
///
/// If a file at `path` exists, it will be truncated, otherwise a new one will
/// be created.
///
/// This function optionally allows to name BCDD functions. If `functions` and
/// `function_names` are non-null and `num_function_names` is non-zero, then
/// `functions` and `function_names` are assumed to point to an array of length
/// (at least) `num_function_names`. In this case, the i-th function is labeled
/// with the i-th function name.
///
/// The output may also include nodes that are not reachable from ``functions``.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  manager             The manager
/// @param  path                Path at which the DOT file should be written
/// @param  functions           Array of `num_function_names` BCDD functions in
///                             `manager` to be labeled.
///                             May be `NULL`, in which case there will be no
///                             labels.
/// @param  function_names      Array of `num_function_names` null-terminated
///                             UTF-8 strings, each labelling the respective
///                             BCDD function.
///                             May be `NULL`, in which case there will be no
///                             labels.
/// @param  num_function_names  Count of functions to be labeled
///
/// @returns  `true` on success
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_dump_all_dot_file(
    manager: bcdd_manager_t,
    path: *const c_char,
    functions: *const bcdd_t,
    function_names: *const *const c_char,
    num_function_names: usize,
) -> bool {
    crate::util::dump_all_dot_file(manager, path, functions, function_names, num_function_names)
}

/// Print statistics to stderr
#[no_mangle]
pub extern "C" fn oxidd_bcdd_print_stats() {
    oxidd::bcdd::print_stats();
}
