use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use rustc_hash::FxHasher;

use oxidd::util::num::F64;
use oxidd::util::AllocResult;
use oxidd::util::Borrowed;
use oxidd::util::OutOfMemory;
use oxidd::zbdd::ZBDDFunction;
use oxidd::zbdd::ZBDDManagerRef;
use oxidd::BooleanFunction;
use oxidd::BooleanVecSet;
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

/// Reference to a manager of a zero-suppressed decision diagram (ZBDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking oxidd_zbdd_manager_t
/// instances as arguments do not take ownership of them (i.e. do not decrement
/// the reference counter). Returned oxidd_zbdd_manager_t instances must
/// typically be deallocated using oxidd_zbdd_manager_unref() to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_zbdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl oxidd_zbdd_manager_t {
    unsafe fn get(self) -> ManuallyDrop<ZBDDManagerRef> {
        assert!(!self._p.is_null(), "the given manager is invalid");
        ManuallyDrop::new(ZBDDManagerRef::from_raw(self._p))
    }
}

/// Boolean function represented as a zero-suppressed decision diagram (ZBDD)
///
/// This is essentially an optional reference to a ZBDD node. In case an
/// operation runs out of memory, it returns an invalid ZBDD function. Unless
/// explicitly specified otherwise, an oxidd_zbdd_t parameter may be invalid to
/// permit "chaining" operations without explicit checks in between. In this
/// case, the returned ZBDD function is also invalid.
///
/// An instance of this type contributes to both the reference count of the
/// referenced node and the manager that stores this node. Unless explicitly
/// stated otherwise, functions taking oxidd_zbdd_t instances as arguments do
/// not take ownership of them (i.e. do not decrement the reference counters).
/// Returned oxidd_zbdd_t instances must typically be deallocated using
/// oxidd_zbdd_unref() to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_zbdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl oxidd_zbdd_t {
    const INVALID: Self = Self {
        _p: std::ptr::null(),
        _i: 0,
    };

    unsafe fn get(self) -> AllocResult<ManuallyDrop<ZBDDFunction>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(ZBDDFunction::from_raw(self._p, self._i)))
        }
    }
}

impl From<ZBDDFunction> for oxidd_zbdd_t {
    fn from(value: ZBDDFunction) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<ZBDDFunction>> for oxidd_zbdd_t {
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
impl From<Option<ZBDDFunction>> for oxidd_zbdd_t {
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
pub struct oxidd_zbdd_pair_t {
    /// First component
    first: oxidd_zbdd_t,
    /// Second component
    second: oxidd_zbdd_t,
}

unsafe fn op1(
    f: oxidd_zbdd_t,
    op: impl FnOnce(&ZBDDFunction) -> AllocResult<ZBDDFunction>,
) -> oxidd_zbdd_t {
    f.get().and_then(|f| op(&f)).into()
}

unsafe fn op2(
    lhs: oxidd_zbdd_t,
    rhs: oxidd_zbdd_t,
    op: impl FnOnce(&ZBDDFunction, &ZBDDFunction) -> AllocResult<ZBDDFunction>,
) -> oxidd_zbdd_t {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

unsafe fn op3(
    f1: oxidd_zbdd_t,
    f2: oxidd_zbdd_t,
    f3: oxidd_zbdd_t,
    op: impl FnOnce(&ZBDDFunction, &ZBDDFunction, &ZBDDFunction) -> AllocResult<ZBDDFunction>,
) -> oxidd_zbdd_t {
    f1.get()
        .and_then(|f1| op(&f1, &*f2.get()?, &*f3.get()?))
        .into()
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
) -> oxidd_zbdd_manager_t {
    oxidd_zbdd_manager_t {
        _p: oxidd::zbdd::new_manager(inner_node_capacity, apply_cache_capacity, threads).into_raw(),
    }
}

/// Increment the manager reference counter
///
/// @returns  `manager`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_ref(
    manager: oxidd_zbdd_manager_t,
) -> oxidd_zbdd_manager_t {
    std::mem::forget(manager.get().clone());
    manager
}

/// Decrement the manager reference counter
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_manager_unref(manager: oxidd_zbdd_manager_t) {
    drop(ZBDDManagerRef::from_raw(manager._p));
}

/// Increment the reference counter of the node referenced by `f` as well as
/// the manager storing the node
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_ref(f: oxidd_zbdd_t) -> oxidd_zbdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference counter of the node referenced by `f` as well as the
/// manager storing the node
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_unref(f: oxidd_zbdd_t) {
    drop(ZBDDFunction::from_raw(f._p, f._i));
}

/// Get the manager that stores `f`
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  A manager reference with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_containing_manager(f: oxidd_zbdd_t) -> oxidd_zbdd_manager_t {
    oxidd_zbdd_manager_t {
        _p: f.get().expect(FUNC_UNWRAP_MSG).manager_ref().into_raw(),
    }
}

/// Get the number of inner nodes currently stored in `manager`
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_num_inner_nodes(manager: oxidd_zbdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}

/// Get a fresh variable in the form of a singleton set. This adds a new level
/// to a decision diagram.
///
/// Locking behavior: acquires an exclusive manager lock.
///
/// @returns  The ZBDD set representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_new_singleton(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| ZBDDFunction::new_singleton(manager).into())
}

/// Get the ZBDD Boolean function v for the singleton set {v}
///
/// `var` must be a singleton set.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  singleton  The singleton set {v}
///
/// @returns  The ZBDD Boolean function v
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_var_boolean_function(singleton: oxidd_zbdd_t) -> oxidd_zbdd_t {
    let res = singleton.get().and_then(|singleton| {
        singleton.with_manager_shared(|manager, edge| {
            Ok(ZBDDFunction::from_edge(
                manager,
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
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set referencing the new node
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_make_node(
    var: oxidd_zbdd_t,
    hi: oxidd_zbdd_t,
    lo: oxidd_zbdd_t,
) -> oxidd_zbdd_t {
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
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_empty(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::empty(manager).into())
}

/// Get the ZBDD set {∅}
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_base(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
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
/// cofactors, then use oxidd_zbdd_cofactor_true() or
/// oxidd_zbdd_cofactor_false(). These functions are slightly more efficient
/// then.
///
/// Note that the domain of f is 𝔹ⁿ⁺¹ while the domain of f<sub>true</sub> and
/// f<sub>false</sub> is 𝔹ⁿ. (Remember that, e.g., g(x₀) = x₀ and
/// g'(x₀, x₁) = x₀ have different representations as ZBDDs.)
///
/// Locking behavior: acquires a shared manager lock.
///
/// Runtime complexity: O(1)
///
/// @returns  The pair `f_true` and `f_false` if `f` is valid and references an
///           inner node, otherwise a pair of invalid functions.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_cofactors(f: oxidd_zbdd_t) -> oxidd_zbdd_pair_t {
    if let Ok(f) = f.get() {
        if let Some((t, e)) = f.cofactors() {
            return oxidd_zbdd_pair_t {
                first: t.into(),
                second: e.into(),
            };
        }
    }
    oxidd_zbdd_pair_t {
        first: oxidd_zbdd_t::INVALID,
        second: oxidd_zbdd_t::INVALID,
    }
}

/// Get the cofactor `f_true` of `f`
///
/// This function is slightly more efficient than oxidd_zbdd_cofactors() in case
/// `f_false` is not needed. For a more detailed description, see
/// oxidd_zbdd_cofactors().
///
/// Locking behavior: acquires a shared manager lock.
///
/// Runtime complexity: O(1)
///
/// @returns  `f_true` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_cofactor_true(f: oxidd_zbdd_t) -> oxidd_zbdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_true().into()
    } else {
        oxidd_zbdd_t::INVALID
    }
}

/// Get the cofactor `f_false` of `f`
///
/// This function is slightly more efficient than oxidd_bdd_cofactors() in case
/// `f_true` is not needed. For a more detailed description, see
/// oxidd_bdd_cofactors().
///
/// Locking behavior: acquires a shared manager lock.
///
/// Runtime complexity: O(1)
///
/// @returns  `f_false` if `f` is valid and references an inner node, otherwise
///           an invalid function.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_cofactor_false(f: oxidd_zbdd_t) -> oxidd_zbdd_t {
    if let Ok(f) = f.get() {
        f.cofactor_false().into()
    } else {
        oxidd_zbdd_t::INVALID
    }
}

/// Get the set of subsets of `set` not containing `var`, formally
/// `{s ∈ set | var ∉ s}`
///
/// `var` must be a singleton set.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_subset0(set: oxidd_zbdd_t, var: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(set, var, ZBDDFunction::subset0)
}

/// Get the set of subsets of `set` containing `var`, formally
/// `{s ∈ set | var ∈ s}`
///
/// `var` must be a singleton set.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_subset1(set: oxidd_zbdd_t, var: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(set, var, ZBDDFunction::subset1)
}

/// Get the set of subsets derived from `set` by adding `var` to the
/// subsets that do not contain `var`, and removing `var` from the subsets
/// that contain `var`, formally
/// `{s ∪ {var} | s ∈ set ∧ var ∉ s} ∪ {s ∖ {var} | s ∈ set ∧ var ∈ s}`
///
/// `var` must be a singleton set.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_change(set: oxidd_zbdd_t, var: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(set, var, ZBDDFunction::change)
}

/// Compute the ZBDD for the union `lhs ∪ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_union(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::union)
}

/// Compute the ZBDD for the intersection `lhs ∩ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_intsec(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::intsec)
}

/// Compute the ZBDD for the set difference `lhs ∖ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_diff(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::diff)
}

/// Get a fresh variable, i.e., a Boolean function that is true if and only if
/// the variable is true. This adds a new level to a decision diagram.
///
/// Locking behavior: acquires an exclusive manager lock.
///
/// @returns  The ZBDD Boolean function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_new_var(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| ZBDDFunction::new_var(manager).into())
}

/// Get the constant false ZBDD Boolean function `⊥`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_false(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::f(manager).into())
}

/// Get the constant true ZBDD Boolean function `⊤`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_true(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDFunction::t(manager).into())
}

/// Compute the ZBDD for the negation `¬f`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_not(f: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op1(f, ZBDDFunction::not)
}

/// Compute the ZBDD for the conjunction `lhs ∧ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_and(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::and)
}

/// Compute the ZBDD for the disjunction `lhs ∨ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_or(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::or)
}

/// Compute the ZBDD for the negated conjunction `lhs ⊼ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_nand(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::nand)
}

/// Compute the ZBDD for the negated disjunction `lhs ⊽ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_nor(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::nor)
}

/// Compute the ZBDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_xor(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::xor)
}

/// Compute the ZBDD for the equivalence `lhs ↔ rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_equiv(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::equiv)
}

/// Compute the ZBDD for the implication `lhs → rhs` (or `lhs ≤ rhs`)
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_imp(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::imp)
}

/// Compute the ZBDD for the strict implication `lhs < rhs`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_imp_strict(
    lhs: oxidd_zbdd_t,
    rhs: oxidd_zbdd_t,
) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDFunction::imp_strict)
}

/// Compute the ZBDD for the conditional `cond ? then_case : else_case`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_ite(
    cond: oxidd_zbdd_t,
    then_case: oxidd_zbdd_t,
    else_case: oxidd_zbdd_t,
) -> oxidd_zbdd_t {
    op3(cond, then_case, else_case, ZBDDFunction::ite)
}

/// Count nodes in `f`
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f     A *valid* ZBDD function
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_node_count(f: oxidd_zbdd_t) -> usize {
    f.get().expect(FUNC_UNWRAP_MSG).node_count()
}

/// Check if `f` is satisfiable
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  `true` iff there is a satisfying assignment for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_satisfiable(f: oxidd_zbdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).satisfiable()
}

/// Check if `f` is valid
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f  A *valid* ZBDD function
///
/// @returns  `true` iff there are only satisfying assignments for `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_valid(f: oxidd_zbdd_t) -> bool {
    f.get().expect(FUNC_UNWRAP_MSG).valid()
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f     A *valid* ZBDD function
/// @param  vars  Number of input variables
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_sat_count_double(
    f: oxidd_zbdd_t,
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
/// @param  f  A *valid* ZBDD function
///
/// @returns  A satisfying assignment if there exists one. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           oxidd_assignment_free().
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_pick_cube(f: oxidd_zbdd_t) -> oxidd_assignment_t {
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

/// Pair of a ZBDD function and a Boolean
#[repr(C)]
pub struct oxidd_zbdd_bool_pair_t {
    /// The ZBDD
    func: oxidd_zbdd_t,
    /// The Boolean value
    val: bool,
}

/// Evaluate the Boolean function `f` with arguments `args`
///
/// `args` determines the valuation for all variables. Missing values are
/// assumed to be false. The order is irrelevant. All elements must point to
/// inner nodes. Note that the domain of `f` is treated somewhat implicitly, it
/// contains at least all `args` and all variables in the support of the ZBDD
/// `f`. Unlike BDDs, extending the domain changes the semantics of ZBDDs.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @param  f         A *valid* ZBDD function
/// @param  args      Array of pairs `(variable, value)`, where `variable` is
///                   valid
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_eval(
    f: oxidd_zbdd_t,
    args: *const oxidd_zbdd_bool_pair_t,
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
            ZBDDFunction::eval_edge(
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
pub extern "C" fn oxidd_zbdd_print_stats() {
    oxidd::zbdd::print_stats();
}
