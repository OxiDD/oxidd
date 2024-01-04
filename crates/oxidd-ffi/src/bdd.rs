use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use oxidd::AllocResult;
use oxidd::OutOfMemory;
use rustc_hash::FxHasher;

use oxidd::bdd::BDDFunction;
use oxidd::bdd::BDDManagerRef;
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
const FUNC_UNWRAP_MSG: &'static str = "the given function is invalid";

/// Reference to a manager of a simple binary decision diagram (BDD)
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_bdd_manager_t {
    __p: *const std::ffi::c_void,
}

impl oxidd_bdd_manager_t {
    unsafe fn get(self) -> ManuallyDrop<BDDManagerRef> {
        assert!(!self.__p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(BDDManagerRef::from_raw(self.__p))
    }
}

/// Boolean function represented as a simple binary decision diagram (BDD)
///
/// This is essentially a reference to a BDD node.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_bdd_t {
    __p: *const std::ffi::c_void,
    __i: u32,
}

impl oxidd_bdd_t {
    unsafe fn get(self) -> AllocResult<ManuallyDrop<BDDFunction>> {
        if self.__p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(BDDFunction::from_raw(self.__p, self.__i)))
        }
    }
}

impl From<BDDFunction> for oxidd_bdd_t {
    fn from(value: BDDFunction) -> Self {
        let (__p, __i) = value.into_raw();
        Self { __p, __i }
    }
}

impl From<AllocResult<BDDFunction>> for oxidd_bdd_t {
    fn from(value: AllocResult<BDDFunction>) -> Self {
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
    f: oxidd_bdd_t,
    op: impl FnOnce(&BDDFunction) -> AllocResult<BDDFunction>,
) -> oxidd_bdd_t {
    f.get().and_then(|f| op(&*f)).into()
}

unsafe fn op2(
    lhs: oxidd_bdd_t,
    rhs: oxidd_bdd_t,
    op: impl FnOnce(&BDDFunction, &BDDFunction) -> AllocResult<BDDFunction>,
) -> oxidd_bdd_t {
    lhs.get().and_then(|lhs| op(&*lhs, &*rhs.get()?)).into()
}

unsafe fn op3(
    f1: oxidd_bdd_t,
    f2: oxidd_bdd_t,
    f3: oxidd_bdd_t,
    op: impl FnOnce(&BDDFunction, &BDDFunction, &BDDFunction) -> AllocResult<BDDFunction>,
) -> oxidd_bdd_t {
    f1.get()
        .and_then(|f1| op(&*f1, &*f2.get()?, &*f3.get()?))
        .into()
}

/// Level number type
type oxidd_level_no_t = u32;

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
) -> oxidd_bdd_manager_t {
    oxidd_bdd_manager_t {
        __p: oxidd::bdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
    }
}

/// Increment the manager reference counter
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_ref(
    manager: oxidd_bdd_manager_t,
) -> oxidd_bdd_manager_t {
    std::mem::forget(manager.get().clone());
    manager
}

/// Decrement the manager reference counter
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_manager_unref(manager: oxidd_bdd_manager_t) {
    if !manager.__p.is_null() {
        drop(BDDManagerRef::from_raw(manager.__p));
    }
}

/// Increment the reference counter of the given BDD node
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_ref(f: oxidd_bdd_t) -> oxidd_bdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference count of the given BDD node
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_unref(f: oxidd_bdd_t) {
    if !f.__p.is_null() {
        drop(BDDFunction::from_raw(f.__p, f.__i));
    }
}

/// Get the number of inner nodes currently stored in `manager`
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_num_inner_nodes(manager: oxidd_bdd_manager_t) -> usize {
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
/// @returns  The BDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_new_var(manager: oxidd_bdd_manager_t) -> oxidd_bdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| BDDFunction::new_var(manager).into())
}

/// Get the constant false BDD function `⊥`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_false(manager: oxidd_bdd_manager_t) -> oxidd_bdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BDDFunction::f(manager).into())
}

/// Get the constant true BDD function `⊤`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_true(manager: oxidd_bdd_manager_t) -> oxidd_bdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BDDFunction::t(manager).into())
}

/// Compute the BDD for the negation `¬f`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_not(f: oxidd_bdd_t) -> oxidd_bdd_t {
    op1(f, BDDFunction::not)
}

/// Compute the BDD for the conjunction `lhs ∧ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_and(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::and)
}

/// Compute the BDD for the disjunction `lhs ∨ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_or(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::or)
}

/// Compute the BDD for the negated conjunction `lhs ⊼ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_nand(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::nand)
}

/// Compute the BDD for the negated disjunction `lhs ⊽ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_nor(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::nor)
}

/// Compute the BDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_xor(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::xor)
}

/// Compute the BDD for the equivalence `lhs ↔ rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_equiv(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::equiv)
}

/// Compute the BDD for the implication `lhs → rhs` (or `self ≤ rhs`)
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_imp(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::imp)
}

/// Compute the BDD for the strict implication `lhs < rhs`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_imp_strict(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::imp_strict)
}

/// Compute the BDD for the conditional `cond ? then_case : else_case`
///
/// This function does not decrement the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_ite(
    cond: oxidd_bdd_t,
    then_case: oxidd_bdd_t,
    else_case: oxidd_bdd_t,
) -> oxidd_bdd_t {
    op3(cond, then_case, else_case, BDDFunction::ite)
}

/// Compute the BDD for the universial quantification of `f` over `vars`
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
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_forall(f: oxidd_bdd_t, var: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, var, BDDFunction::forall)
}

/// Compute the BDD for the existential quantification of `f` over `vars`
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
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_exist(f: oxidd_bdd_t, var: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, var, BDDFunction::exist)
}

/// Compute the BDD for the unique quantification of `f` over `vars`
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
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_unique(f: oxidd_bdd_t, var: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, var, BDDFunction::unique)
}

/// Count nodes in `f`
///
/// This function does not decrement the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_node_count(f: oxidd_bdd_t) -> usize {
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
pub unsafe extern "C" fn oxidd_bdd_sat_count_uint64(f: oxidd_bdd_t, vars: oxidd_level_no_t) -> u64 {
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
pub unsafe extern "C" fn oxidd_bdd_sat_count_double(f: oxidd_bdd_t, vars: oxidd_level_no_t) -> f64 {
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
pub unsafe extern "C" fn oxidd_bdd_pick_cube(f: oxidd_bdd_t) -> oxidd_assignment_t {
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
pub extern "C" fn oxidd_bdd_print_stats() {
    oxidd::bdd::print_stats();
}
