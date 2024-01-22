use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use oxidd::AllocResult;
use oxidd::OutOfMemory;
use rustc_hash::FxHasher;

use oxidd::cbdd::CBDDFunction;
use oxidd::cbdd::CBDDManagerRef;
use oxidd::util::num::Saturating;
use oxidd::util::num::F64;
use oxidd::BooleanFunction;
use oxidd::BooleanFunctionQuant;
use oxidd::Function;
use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd::RawFunction;
use oxidd::RawManagerRef;

use crate::oxidd_assignment_t;

/// cbindgen:ignore
const FUNC_UNWRAP_MSG: &str = "the given function is invalid";

/// Reference to a manager of a binary decision diagram with complement edges
/// (CBDD)
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_cbdd_manager_t {
    __p: *const std::ffi::c_void,
}

impl oxidd_cbdd_manager_t {
    #[inline]
    unsafe fn get(self) -> ManuallyDrop<CBDDManagerRef> {
        assert!(!self.__p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(CBDDManagerRef::from_raw(self.__p))
    }
}

/// Boolean function represented as a binary decision diagram with complement
/// edges (CBDD)
///
/// This is essentially a reference to a CBDD node.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_cbdd_t {
    __p: *const std::ffi::c_void,
    __i: u32,
}

impl oxidd_cbdd_t {
    unsafe fn get(self) -> AllocResult<ManuallyDrop<CBDDFunction>> {
        if self.__p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(CBDDFunction::from_raw(
                self.__p, self.__i,
            )))
        }
    }
}

impl From<CBDDFunction> for oxidd_cbdd_t {
    fn from(value: CBDDFunction) -> Self {
        let (__p, __i) = value.into_raw();
        Self { __p, __i }
    }
}

impl From<AllocResult<CBDDFunction>> for oxidd_cbdd_t {
    fn from(value: AllocResult<CBDDFunction>) -> Self {
        match value {
            Ok(f) => {
                let (__p, __i) = f.into_raw();
                Self { __p, __i }
            }
            Err(_) => Self {
                __p: std::ptr::null(),
                __i: 0,
            },
        }
    }
}

unsafe fn op1(
    f: oxidd_cbdd_t,
    op: impl FnOnce(&CBDDFunction) -> AllocResult<CBDDFunction>,
) -> oxidd_cbdd_t {
    f.get().and_then(|f| op(&f)).into()
}

unsafe fn op2(
    lhs: oxidd_cbdd_t,
    rhs: oxidd_cbdd_t,
    op: impl FnOnce(&CBDDFunction, &CBDDFunction) -> AllocResult<CBDDFunction>,
) -> oxidd_cbdd_t {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

unsafe fn op3(
    f1: oxidd_cbdd_t,
    f2: oxidd_cbdd_t,
    f3: oxidd_cbdd_t,
    op: impl FnOnce(&CBDDFunction, &CBDDFunction, &CBDDFunction) -> AllocResult<CBDDFunction>,
) -> oxidd_cbdd_t {
    f1.get()
        .and_then(|f1| op(&f1, &*f2.get()?, &*f3.get()?))
        .into()
}

/// Level number type
type oxidd_level_no_t = u32;

/// Create a new manager for a binary decision diagram with complement edges
/// (CBDD)
///
/// @param  inner_node_capacity   Maximum number of inner nodes. `0` means no
///                               limit.
/// @param  apply_cache_capacity  Maximum number of apply cache entries. The
///                               apply cache implementation may round this up
///                               to the next power of two.
/// @param  threads               Number of threads for concurrent operations.
///                               `0` means automatic selection.
///
/// @returns  The CBDD manager with reference count 1
#[no_mangle]
pub extern "C" fn oxidd_cbdd_manager_new(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> oxidd_cbdd_manager_t {
    oxidd_cbdd_manager_t {
        __p: oxidd::cbdd::new_manager(inner_node_capacity, apply_cache_capacity, threads)
            .into_raw(),
    }
}

/// Increment the manager reference counter
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_manager_ref(
    manager: oxidd_cbdd_manager_t,
) -> oxidd_cbdd_manager_t {
    std::mem::forget(manager.get().clone());
    manager
}

/// Decrement the manager reference counter
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_manager_unref(manager: oxidd_cbdd_manager_t) {
    if !manager.__p.is_null() {
        drop(CBDDManagerRef::from_raw(manager.__p));
    }
}

/// Increment the reference counter of the given CBDD node
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_ref(f: oxidd_cbdd_t) -> oxidd_cbdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference count of the given CBDD node
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_unref(f: oxidd_cbdd_t) {
    if !f.__p.is_null() {
        drop(CBDDFunction::from_raw(f.__p, f.__i));
    }
}

/// Get the number of inner nodes currently stored in `manager`
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_num_inner_nodes(manager: oxidd_cbdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}

/// Get a fresh variable, i.e. a function that is true if and only if the
/// variable is true. This adds a new level to a decision diagram.
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires an exclusive manager lock.
///
/// @returns  The CBDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_new_var(manager: oxidd_cbdd_manager_t) -> oxidd_cbdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| CBDDFunction::new_var(manager).into())
}

/// Get the constant false CBDD function `⊥`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_false(manager: oxidd_cbdd_manager_t) -> oxidd_cbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| CBDDFunction::f(manager).into())
}

/// Get the constant true CBDD function `⊤`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_true(manager: oxidd_cbdd_manager_t) -> oxidd_cbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| CBDDFunction::t(manager).into())
}

/// Compute the CBDD for the negation `¬f`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_not(f: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op1(f, CBDDFunction::not)
}

/// Compute the CBDD for the conjunction `lhs ∧ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_and(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::and)
}

/// Compute the CBDD for the disjunction `lhs ∨ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_or(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::or)
}

/// Compute the CBDD for the negated conjunction `lhs ⊼ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_nand(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::nand)
}

/// Compute the CBDD for the negated disjunction `lhs ⊽ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_nor(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::nor)
}

/// Compute the CBDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_xor(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::xor)
}

/// Compute the CBDD for the equivalence `lhs ↔ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_equiv(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::equiv)
}

/// Compute the CBDD for the implication `lhs → rhs` (or `self ≤ rhs`)
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_imp(lhs: oxidd_cbdd_t, rhs: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::imp)
}

/// Compute the CBDD for the strict implication `lhs < rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_imp_strict(
    lhs: oxidd_cbdd_t,
    rhs: oxidd_cbdd_t,
) -> oxidd_cbdd_t {
    op2(lhs, rhs, CBDDFunction::imp_strict)
}

/// Compute the CBDD for the conditional `cond ? then_case : else_case`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_ite(
    cond: oxidd_cbdd_t,
    then_case: oxidd_cbdd_t,
    else_case: oxidd_cbdd_t,
) -> oxidd_cbdd_t {
    op3(cond, then_case, else_case, CBDDFunction::ite)
}

/// Compute the CBDD for the universial quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurences of the variables universal
/// quantification. Universial quantification of a boolean function `f(…, x, …)`
/// over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_forall(f: oxidd_cbdd_t, var: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(f, var, CBDDFunction::forall)
}

/// Compute the CBDD for the existential quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurences of the variables by
/// existential quantification. Existential quantification of a boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_exist(f: oxidd_cbdd_t, var: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(f, var, CBDDFunction::exist)
}

/// Compute the CBDD for the unique quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurences of the variables by
/// unique quantification. Unique quantification of a boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The CBDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_unique(f: oxidd_cbdd_t, var: oxidd_cbdd_t) -> oxidd_cbdd_t {
    op2(f, var, CBDDFunction::unique)
}

/// Count nodes in `f`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_node_count(f: oxidd_cbdd_t) -> usize {
    f.get().expect(FUNC_UNWRAP_MSG).node_count()
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The number of satisfying assignments or `UINT64_MAX` if the number
///           or some intermediate result is too large
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_sat_count_uint64(
    f: oxidd_cbdd_t,
    vars: oxidd_level_no_t,
) -> u64 {
    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .sat_count::<Saturating<u64>, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_sat_count_double(
    f: oxidd_cbdd_t,
    vars: oxidd_level_no_t,
) -> f64 {
    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Pick a satisfying assignment
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  A satisfying assignment. If `f` is unsatisfiable, the data pointer
///           is `NULL` and len is 0. In any case, the assignment can be
///           deallocated using oxidd_assignment_free().
#[no_mangle]
pub unsafe extern "C" fn oxidd_cbdd_pick_cube(f: oxidd_cbdd_t) -> oxidd_assignment_t {
    let res = f.get().expect(FUNC_UNWRAP_MSG).pick_cube([], |_, _| false);
    match res {
        Some(mut v) => {
            v.shrink_to_fit();
            let len = v.len();
            let data = v.as_mut_ptr() as _;
            std::mem::forget(v);
            oxidd_assignment_t { data, len }
        }
        None => oxidd_assignment_t {
            data: std::ptr::null_mut(),
            len: 0,
        },
    }
}

/// Print statistics to stderr
#[no_mangle]
pub extern "C" fn oxidd_cbdd_print_stats() {
    oxidd::cbdd::print_stats();
}
