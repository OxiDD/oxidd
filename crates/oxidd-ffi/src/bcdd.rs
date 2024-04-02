use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use rustc_hash::FxHasher;

use oxidd::bcdd::BCDDFunction;
use oxidd::bcdd::BCDDManagerRef;
use oxidd::util::num::F64;
use oxidd::util::AllocResult;
use oxidd::util::Borrowed;
use oxidd::util::OutOfMemory;
use oxidd::BooleanFunction;
use oxidd::BooleanFunctionQuant;
use oxidd::Edge;
use oxidd::Function;
use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd::RawFunction;
use oxidd::RawManagerRef;

use crate::util::oxidd_assignment_t;
use crate::util::oxidd_level_no_t;

/// cbindgen:ignore
const FUNC_UNWRAP_MSG: &str = "the given function is invalid";

/// Reference to a manager of a binary decision diagram with complement edges
/// (BCDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking oxidd_bcdd_manager_t
/// instances as arguments do not take ownership of them (i.e. do not decrement
/// the reference counter). Returned oxidd_bcdd_manager_t instances must
/// typically be deallocated using oxidd_bcdd_manager_unref() to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_bcdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl oxidd_bcdd_manager_t {
    #[inline]
    unsafe fn get(self) -> ManuallyDrop<BCDDManagerRef> {
        assert!(!self._p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(BCDDManagerRef::from_raw(self._p))
    }
}

/// Boolean function represented as a binary decision diagram with complement
/// edges (BCDD)
///
/// /// This is essentially an optional (tagged) reference to a BCDD node. In
/// case an operation runs out of memory, it returns an invalid BCDD function.
/// Unless explicitly specified otherwise, an oxidd_bcdd_t parameter may be
/// invalid to permit "chaining" operations without explicit checks in between.
/// In this case, the returned BCDD function is also invalid.
///
/// An instance of this type contributes to both the reference count of the
/// referenced node and the manager that stores this node. Unless explicitly
/// stated otherwise, functions taking oxidd_bcdd_t instances as arguments do
/// not take ownership of them (i.e. do not decrement the reference counters).
/// Returned oxidd_bcdd_t instances must typically be deallocated using
/// oxidd_bcdd_unref() to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_bcdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl oxidd_bcdd_t {
    const INVALID: Self = Self {
        _p: std::ptr::null(),
        _i: 0,
    };

    unsafe fn get(self) -> AllocResult<ManuallyDrop<BCDDFunction>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(BCDDFunction::from_raw(self._p, self._i)))
        }
    }
}

impl From<BCDDFunction> for oxidd_bcdd_t {
    fn from(value: BCDDFunction) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<BCDDFunction>> for oxidd_bcdd_t {
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
impl From<Option<BCDDFunction>> for oxidd_bcdd_t {
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
pub struct oxidd_bcdd_pair_t {
    /// First component
    first: oxidd_bcdd_t,
    /// Second component
    second: oxidd_bcdd_t,
}

unsafe fn op1(
    f: oxidd_bcdd_t,
    op: impl FnOnce(&BCDDFunction) -> AllocResult<BCDDFunction>,
) -> oxidd_bcdd_t {
    f.get().and_then(|f| op(&f)).into()
}

unsafe fn op2(
    lhs: oxidd_bcdd_t,
    rhs: oxidd_bcdd_t,
    op: impl FnOnce(&BCDDFunction, &BCDDFunction) -> AllocResult<BCDDFunction>,
) -> oxidd_bcdd_t {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

unsafe fn op3(
    f1: oxidd_bcdd_t,
    f2: oxidd_bcdd_t,
    f3: oxidd_bcdd_t,
    op: impl FnOnce(&BCDDFunction, &BCDDFunction, &BCDDFunction) -> AllocResult<BCDDFunction>,
) -> oxidd_bcdd_t {
    f1.get()
        .and_then(|f1| op(&f1, &*f2.get()?, &*f3.get()?))
        .into()
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
) -> oxidd_bcdd_manager_t {
    oxidd_bcdd_manager_t {
        _p: oxidd::bcdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
    }
}

/// Increment the manager reference counter
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_ref(
    manager: oxidd_bcdd_manager_t,
) -> oxidd_bcdd_manager_t {
    std::mem::forget(manager.get().clone());
    manager
}

/// Decrement the manager reference counter
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_manager_unref(manager: oxidd_bcdd_manager_t) {
    if !manager._p.is_null() {
        drop(BCDDManagerRef::from_raw(manager._p));
    }
}

/// Increment the reference counter of the node referenced by `f` as well as the
/// manager storing the node
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_ref(f: oxidd_bcdd_t) -> oxidd_bcdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference counter of the node referenced by `f` as well as the
/// manager storing the node
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_unref(f: oxidd_bcdd_t) {
    if !f._p.is_null() {
        drop(BCDDFunction::from_raw(f._p, f._i));
    }
}

/// Get the number of inner nodes currently stored in `manager`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_num_inner_nodes(manager: oxidd_bcdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}

/// Get a fresh variable, i.e. a function that is true if and only if the
/// variable is true. This adds a new level to a decision diagram.
///
/// Locking behavior: acquires an exclusive manager lock.
///
/// @returns  The BCDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_new_var(manager: oxidd_bcdd_manager_t) -> oxidd_bcdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| BCDDFunction::new_var(manager).into())
}

/// Get the constant false BCDD function `⊥`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_false(manager: oxidd_bcdd_manager_t) -> oxidd_bcdd_t {
    manager
        .get()
        .with_manager_shared(|manager| BCDDFunction::f(manager).into())
}

/// Get the constant true BCDD function `⊤`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_true(manager: oxidd_bcdd_manager_t) -> oxidd_bcdd_t {
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
/// oxidd_bcdd_cofactor_true() or oxidd_bcdd_cofactor_false(). These functions
/// are slightly more efficient then.
///
/// Locking behavior: acquires a shared manager lock.
///
/// Runtime complexity: O(1)
///
/// @returns  The pair `f_true` and `f_false` if `f` is valid and references an
///           inner node, otherwise a pair of invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_cofactors(f: oxidd_bcdd_t) -> oxidd_bcdd_pair_t {
    if let Ok(f) = f.get() {
        if let Some((t, e)) = f.cofactors() {
            return oxidd_bcdd_pair_t {
                first: t.into(),
                second: e.into(),
            };
        }
    }
    oxidd_bcdd_pair_t {
        first: oxidd_bcdd_t::INVALID,
        second: oxidd_bcdd_t::INVALID,
    }
}

/// Get the cofactor `f_true` of `f`
///
/// This function is slightly more efficient than oxidd_bcdd_cofactors() in case
/// `f_false` is not needed.
///
/// Locking behavior: acquires a shared manager lock.
///
/// Runtime complexity: O(1)
///
/// @returns  `f_true` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_cofactor_true(f: oxidd_bcdd_t) -> oxidd_bcdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_true().into()
    } else {
        oxidd_bcdd_t::INVALID
    }
}

/// Get the cofactor `f_false` of `f`
///
/// This function is slightly more efficient than oxidd_bcdd_cofactors() in case
/// `f_true` is not needed.
///
/// Locking behavior: acquires a shared manager lock.
///
/// Runtime complexity: O(1)
///
/// @returns  `f_false` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_cofactor_false(f: oxidd_bcdd_t) -> oxidd_bcdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_false().into()
    } else {
        oxidd_bcdd_t::INVALID
    }
}

/// Compute the BCDD for the negation `¬f`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_not(f: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op1(f, BCDDFunction::not)
}

/// Compute the BCDD for the conjunction `lhs ∧ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_and(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::and)
}

/// Compute the BCDD for the disjunction `lhs ∨ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_or(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::or)
}

/// Compute the BCDD for the negated conjunction `lhs ⊼ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_nand(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::nand)
}

/// Compute the BCDD for the negated disjunction `lhs ⊽ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_nor(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::nor)
}

/// Compute the BCDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_xor(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::xor)
}

/// Compute the BCDD for the equivalence `lhs ↔ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_equiv(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::equiv)
}

/// Compute the BCDD for the implication `lhs → rhs` (or `lhs ≤ rhs`)
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_imp(lhs: oxidd_bcdd_t, rhs: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::imp)
}

/// Compute the BCDD for the strict implication `lhs < rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_imp_strict(
    lhs: oxidd_bcdd_t,
    rhs: oxidd_bcdd_t,
) -> oxidd_bcdd_t {
    op2(lhs, rhs, BCDDFunction::imp_strict)
}

/// Compute the BCDD for the conditional `cond ? then_case : else_case`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_ite(
    cond: oxidd_bcdd_t,
    then_case: oxidd_bcdd_t,
    else_case: oxidd_bcdd_t,
) -> oxidd_bcdd_t {
    op3(cond, then_case, else_case, BCDDFunction::ite)
}

/// Compute the BCDD for `f` with its variables restricted to constant values
/// according to `vars`
///
/// `vars` conceptually is a partial assignment, represented as the conjunction
/// of positive or negative literals, depending on whether the variable should
/// be mapped to true or false.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_restrict(f: oxidd_bcdd_t, vars: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(f, vars, BCDDFunction::restrict)
}

/// Compute the BCDD for the universal quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// universal quantification. Universal quantification of a Boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_forall(f: oxidd_bcdd_t, var: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(f, var, BCDDFunction::forall)
}

/// Compute the BCDD for the existential quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// existential quantification. Existential quantification of a Boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∨ f(…, 1, …)`.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_exist(f: oxidd_bcdd_t, var: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(f, var, BCDDFunction::exist)
}

/// Compute the BCDD for the unique quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// unique quantification. Unique quantification of a Boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ⊕ f(…, 1, …)`.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The BCDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_unique(f: oxidd_bcdd_t, var: oxidd_bcdd_t) -> oxidd_bcdd_t {
    op2(f, var, BCDDFunction::unique)
}

/// Count nodes in `f`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_node_count(f: oxidd_bcdd_t) -> usize {
    f.get().expect(FUNC_UNWRAP_MSG).node_count()
}

/// Check if `f` is satisfiable
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  `true` iff there is a satisfying assignment for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_satisfiable(f: oxidd_bcdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).satisfiable()
}

/// Check if `f` is valid
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  `true` iff there are only satisfying assignments for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_valid(f: oxidd_bcdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).valid()
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f     A *valid* BCDD function
/// @param  vars  Number of input variables
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_sat_count_double(
    f: oxidd_bcdd_t,
    vars: oxidd_level_no_t,
) -> f64 {
    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Pick a satisfying assignment
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f  A *valid* BCDD function
///
/// @returns  A satisfying assignment if there exists one. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           oxidd_assignment_free().
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_pick_cube(f: oxidd_bcdd_t) -> oxidd_assignment_t {
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

/// Pair of a BCDD function and a Boolean
#[repr(C)]
pub struct oxidd_bcdd_bool_pair_t {
    /// The function
    func: oxidd_bcdd_t,
    /// The Boolean value
    val: bool,
}

/// Evaluate the Boolean function `f` with arguments `args`
///
/// `args` determines the valuation for all variables. Missing values are
/// assumed to be false. The order is irrelevant. All elements must point to
/// inner nodes.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f         A *valid* BCDD function
/// @param  args      Array of pairs `(variable, value)`, where `variable` is
///                   valid
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_eval(
    f: oxidd_bcdd_t,
    args: *const oxidd_bcdd_bool_pair_t,
    num_args: usize,
) -> bool {
    let args = std::slice::from_raw_parts(args, num_args);

    /// `Borrowed<T>` is represented like `T` and the compiler still checks that
    /// `T: 'b`
    #[inline(always)]
    unsafe fn extend_lifetime<'b, T>(x: Borrowed<'_, T>) -> Borrowed<'b, T> {
        std::mem::transmute(x)
    }

    f.get()
        .expect(FUNC_UNWRAP_MSG)
        .with_manager_shared(|manager, edge| {
            BCDDFunction::eval_edge(
                manager,
                edge,
                args.iter().map(|p| {
                    let v = p.func.get().expect(FUNC_UNWRAP_MSG);
                    let borrowed = v.as_edge(manager).borrowed();
                    // SAFETY: We can extend the lifetime since the node is also referenced via
                    // `assignment` which outlives even the `with_manager_shared` closure
                    (extend_lifetime(borrowed), p.val)
                }),
            )
        })
}

/// Print statistics to stderr
#[no_mangle]
pub extern "C" fn oxidd_bcdd_print_stats() {
    oxidd::bcdd::print_stats();
}
