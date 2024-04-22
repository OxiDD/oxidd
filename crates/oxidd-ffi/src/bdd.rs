use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use oxidd::util::Borrowed;
use rustc_hash::FxHasher;

use oxidd::bdd::BDDFunction;
use oxidd::bdd::BDDManagerRef;
use oxidd::util::num::F64;
use oxidd::util::AllocResult;
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

/// Reference to a manager of a simple binary decision diagram (BDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking oxidd_bdd_manager_t
/// instances as arguments do not take ownership of them (i.e. do not decrement
/// the reference counter). Returned oxidd_bdd_manager_t instances must
/// typically be deallocated using oxidd_bdd_manager_unref() to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_bdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl oxidd_bdd_manager_t {
    unsafe fn get(self) -> ManuallyDrop<BDDManagerRef> {
        assert!(!self._p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(BDDManagerRef::from_raw(self._p))
    }
}

/// Boolean function represented as a simple binary decision diagram (BDD)
///
/// This is essentially an optional reference to a BDD node. In case an
/// operation runs out of memory, it returns an invalid BDD function. Unless
/// explicitly specified otherwise, an oxidd_bdd_t parameter may be invalid to
/// permit "chaining" operations without explicit checks in between. In this
/// case, the returned BDD function is also invalid.
///
/// An instance of this type contributes to both the reference count of the
/// referenced node and the manager that stores this node. Unless explicitly
/// stated otherwise, functions taking oxidd_bdd_t instances as arguments do not
/// take ownership of them (i.e. do not decrement the reference counters).
/// Returned oxidd_bdd_t instances must typically be deallocated using
/// oxidd_bdd_unref() to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_bdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl oxidd_bdd_t {
    const INVALID: Self = Self {
        _p: std::ptr::null(),
        _i: 0,
    };

    unsafe fn get(self) -> AllocResult<ManuallyDrop<BDDFunction>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(BDDFunction::from_raw(self._p, self._i)))
        }
    }
}

impl From<BDDFunction> for oxidd_bdd_t {
    fn from(value: BDDFunction) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<BDDFunction>> for oxidd_bdd_t {
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
impl From<Option<BDDFunction>> for oxidd_bdd_t {
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
pub struct oxidd_bdd_pair_t {
    /// First component
    first: oxidd_bdd_t,
    /// Second component
    second: oxidd_bdd_t,
}

unsafe fn op1(
    f: oxidd_bdd_t,
    op: impl FnOnce(&BDDFunction) -> AllocResult<BDDFunction>,
) -> oxidd_bdd_t {
    f.get().and_then(|f| op(&f)).into()
}

unsafe fn op2(
    lhs: oxidd_bdd_t,
    rhs: oxidd_bdd_t,
    op: impl FnOnce(&BDDFunction, &BDDFunction) -> AllocResult<BDDFunction>,
) -> oxidd_bdd_t {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

unsafe fn op3(
    f1: oxidd_bdd_t,
    f2: oxidd_bdd_t,
    f3: oxidd_bdd_t,
    op: impl FnOnce(&BDDFunction, &BDDFunction, &BDDFunction) -> AllocResult<BDDFunction>,
) -> oxidd_bdd_t {
    f1.get()
        .and_then(|f1| op(&f1, &*f2.get()?, &*f3.get()?))
        .into()
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
) -> oxidd_bdd_manager_t {
    oxidd_bdd_manager_t {
        _p: oxidd::bdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
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
    if !manager._p.is_null() {
        drop(BDDManagerRef::from_raw(manager._p));
    }
}

/// Increment the reference counter of the node referenced by `f` as well as
/// the manager storing the node
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_ref(f: oxidd_bdd_t) -> oxidd_bdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference counter of the node referenced by `f` as well as
/// the manager storing the node
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_unref(f: oxidd_bdd_t) {
    if !f._p.is_null() {
        drop(BDDFunction::from_raw(f._p, f._i));
    }
}

/// Get the manager that stores `f`
///
/// @param  f  A *valid* BDD function
///
/// @returns  A manager reference with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_containing_manager(f: oxidd_bdd_t) -> oxidd_bdd_manager_t {
    oxidd_bdd_manager_t {
        _p: f.get().expect(FUNC_UNWRAP_MSG).manager_ref().into_raw(),
    }
}

/// Get the number of inner nodes currently stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_num_inner_nodes(manager: oxidd_bdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}

/// Get a fresh variable, i.e., a function that is true if and only if the
/// variable is true. This adds a new level to a decision diagram.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
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
/// Locking behavior: acquires the manager's lock for shared access.
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
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_true(manager: oxidd_bdd_manager_t) -> oxidd_bdd_t {
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
/// cofactors, then use oxidd_bdd_cofactor_true() or oxidd_bdd_cofactor_false().
/// These functions are slightly more efficient then.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Runtime complexity: O(1)
///
/// @returns  The pair `f_true` and `f_false` if `f` is valid and references an
///           inner node, otherwise a pair of invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_cofactors(f: oxidd_bdd_t) -> oxidd_bdd_pair_t {
    if let Ok(f) = f.get() {
        if let Some((t, e)) = f.cofactors() {
            return oxidd_bdd_pair_t {
                first: t.into(),
                second: e.into(),
            };
        }
    }
    oxidd_bdd_pair_t {
        first: oxidd_bdd_t::INVALID,
        second: oxidd_bdd_t::INVALID,
    }
}

/// Get the cofactor `f_true` of `f`
///
/// This function is slightly more efficient than oxidd_bdd_cofactors() in case
/// `f_false` is not needed. For a more detailed description, see
/// oxidd_bdd_cofactors().
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Runtime complexity: O(1)
///
/// @returns  `f_true` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_cofactor_true(f: oxidd_bdd_t) -> oxidd_bdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_true().into()
    } else {
        oxidd_bdd_t::INVALID
    }
}

/// Get the cofactor `f_false` of `f`
///
/// This function is slightly more efficient than oxidd_bdd_cofactors() in case
/// `f_true` is not needed. For a more detailed description, see
/// oxidd_bdd_cofactors().
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Runtime complexity: O(1)
///
/// @returns  `f_false` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_cofactor_false(f: oxidd_bdd_t) -> oxidd_bdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_false().into()
    } else {
        oxidd_bdd_t::INVALID
    }
}

/// Compute the BDD for the negation `¬f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_not(f: oxidd_bdd_t) -> oxidd_bdd_t {
    op1(f, BDDFunction::not)
}

/// Compute the BDD for the conjunction `lhs ∧ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_and(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::and)
}

/// Compute the BDD for the disjunction `lhs ∨ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_or(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::or)
}

/// Compute the BDD for the negated conjunction `lhs ⊼ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_nand(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::nand)
}

/// Compute the BDD for the negated disjunction `lhs ⊽ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_nor(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::nor)
}

/// Compute the BDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_xor(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::xor)
}

/// Compute the BDD for the equivalence `lhs ↔ rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_equiv(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::equiv)
}

/// Compute the BDD for the implication `lhs → rhs` (or `lhs ≤ rhs`)
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_imp(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::imp)
}

/// Compute the BDD for the strict implication `lhs < rhs`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_imp_strict(lhs: oxidd_bdd_t, rhs: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(lhs, rhs, BDDFunction::imp_strict)
}

/// Compute the BDD for the conditional `cond ? then_case : else_case`
///
/// Locking behavior: acquires the manager's lock for shared access.
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
pub unsafe extern "C" fn oxidd_bdd_restrict(f: oxidd_bdd_t, vars: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, vars, BDDFunction::restrict)
}

/// Compute the BDD for the universal quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// universal quantification. Universal quantification of a Boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_forall(f: oxidd_bdd_t, vars: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, vars, BDDFunction::forall)
}

/// Compute the BDD for the existential quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// existential quantification. Existential quantification of a Boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∨ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_exist(f: oxidd_bdd_t, vars: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, vars, BDDFunction::exist)
}

/// Compute the BDD for the unique quantification of `f` over `vars`
///
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. This operation removes all occurrences of the variables by
/// unique quantification. Unique quantification of a Boolean function
/// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ⊕ f(…, 1, …)`.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The BDD function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_unique(f: oxidd_bdd_t, vars: oxidd_bdd_t) -> oxidd_bdd_t {
    op2(f, vars, BDDFunction::unique)
}

/// Count nodes in `f`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f  A *valid* BDD function
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_node_count(f: oxidd_bdd_t) -> usize {
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
pub unsafe extern "C" fn oxidd_bdd_satisfiable(f: oxidd_bdd_t) -> bool {
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
pub unsafe extern "C" fn oxidd_bdd_valid(f: oxidd_bdd_t) -> bool {
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
pub unsafe extern "C" fn oxidd_bdd_sat_count_double(f: oxidd_bdd_t, vars: oxidd_level_no_t) -> f64 {
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
/// @returns  A satisfying assignment if there exists one. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           oxidd_assignment_free().
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

/// Pair of a BDD function and a Boolean
#[repr(C)]
pub struct oxidd_bdd_bool_pair_t {
    /// The function
    func: oxidd_bdd_t,
    /// The Boolean value
    val: bool,
}

/// Evaluate the Boolean function `f` with arguments `args`
///
/// `args` determines the valuation for all variables. Missing values are
/// assumed to be false. The order is irrelevant. All elements must point to
/// inner nodes.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @param  f         A *valid* BDD function
/// @param  args      Array of pairs `(variable, value)`, where `variable` is
///                   valid
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bdd_eval(
    f: oxidd_bdd_t,
    args: *const oxidd_bdd_bool_pair_t,
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
            BDDFunction::eval_edge(
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
pub extern "C" fn oxidd_bdd_print_stats() {
    oxidd::bdd::print_stats();
}
