use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use rustc_hash::FxHasher;

use oxidd::bcdd::{BCDDFunction, BCDDManagerRef};
use oxidd::util::num::F64;
use oxidd::util::{AllocResult, Borrowed, OutOfMemory};
use oxidd::{
    BooleanFunction, BooleanFunctionQuant, Edge, Function, FunctionSubst, Manager, ManagerRef,
    RawFunction, RawManagerRef,
};

// We need to use the following items from `oxidd_core` since cbindgen only
// parses `oxidd_ffi` and `oxidd_core`:
use oxidd_core::LevelNo;

use crate::util::{assignment_t, FUNC_UNWRAP_MSG};

/// Reference to a manager of a binary decision diagram with complement edges
/// (BCDD)
///
/// An instance of this type contributes to the manager's reference counter.
/// Unless explicitly stated otherwise, functions taking oxidd_bcdd_manager_t
/// instances as arguments do not take ownership of them (i.e., do not decrement
/// the reference counter). Returned oxidd_bcdd_manager_t instances must
/// typically be deallocated using oxidd_bcdd_manager_unref() to avoid memory
/// leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct bcdd_manager_t {
    /// Internal pointer value, `NULL` iff this reference is invalid
    _p: *const std::ffi::c_void,
}

impl bcdd_manager_t {
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
/// not take ownership of them (i.e., do not decrement the reference counters).
/// Returned oxidd_bcdd_t instances must typically be deallocated using
/// oxidd_bcdd_unref() to avoid memory leaks.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct bcdd_t {
    /// Internal pointer value, `NULL` iff this function is invalid
    _p: *const std::ffi::c_void,
    /// Internal index value
    _i: usize,
}

/// cbindgen:ignore
impl bcdd_t {
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

#[inline]
unsafe fn op1(f: bcdd_t, op: impl FnOnce(&BCDDFunction) -> AllocResult<BCDDFunction>) -> bcdd_t {
    f.get().and_then(|f| op(&f)).into()
}

#[inline]
unsafe fn op2(
    lhs: bcdd_t,
    rhs: bcdd_t,
    op: impl FnOnce(&BCDDFunction, &BCDDFunction) -> AllocResult<BCDDFunction>,
) -> bcdd_t {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

#[inline]
unsafe fn op3(
    f1: bcdd_t,
    f2: bcdd_t,
    f3: bcdd_t,
    op: impl FnOnce(&BCDDFunction, &BCDDFunction, &BCDDFunction) -> AllocResult<BCDDFunction>,
) -> bcdd_t {
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

/// Get the number of inner nodes currently stored in `manager`
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// @returns  The number of inner nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_num_inner_nodes(manager: bcdd_manager_t) -> usize {
    manager
        .get()
        .with_manager_shared(|manager| manager.num_inner_nodes())
}

/// Get a fresh variable, i.e., a function that is true if and only if the
/// variable is true. This adds a new level to a decision diagram.
///
/// Locking behavior: acquires the manager's lock for exclusive access.
///
/// @returns  The BCDD function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_new_var(manager: bcdd_manager_t) -> bcdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| BCDDFunction::new_var(manager).into())
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
/// oxidd_bcdd_cofactor_true() or oxidd_bcdd_cofactor_false(). These functions
/// are slightly more efficient then.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Runtime complexity: O(1)
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
/// This function is slightly more efficient than oxidd_bcdd_cofactors() in case
/// `f_false` is not needed.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Runtime complexity: O(1)
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
/// This function is slightly more efficient than oxidd_bcdd_cofactors() in case
/// `f_true` is not needed.
///
/// Locking behavior: acquires the manager's lock for shared access.
///
/// Runtime complexity: O(1)
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

/// Compute the BCDD for the conditional `cond ? then_case : else_case`
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
/// To create the substitution, use oxidd_bcdd_substitution_new() and
/// oxidd_bcdd_substitution_add_pair().
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
                pairs: &subst.pairs,
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
    pairs: Vec<(BCDDFunction, BCDDFunction)>,
}

/// Create a new substitution, capable of holding at least `capacity` pairs
/// without reallocating
///
/// Before applying the substitution via oxidd_bcdd_substitute(), add all the
/// pairs via oxidd_bcdd_substitution_add_pair(). Do not add more pairs after
/// the first oxidd_bcdd_substitute() call with this substitution as it may lead
/// to incorrect results.
///
/// @returns  The substitution, to be freed via oxidd_bcdd_substitution_free()
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_substitution_new(capacity: usize) -> *mut bcdd_substitution_t {
    Box::into_raw(Box::new(bcdd_substitution_t {
        id: oxidd_core::util::new_substitution_id(),
        pairs: Vec::with_capacity(capacity),
    }))
}

/// Add a pair of a variable `var` and a replacement function `replacement` to
/// `substitution`
///
/// `var` and `replacement` must be valid BCDD functions. This function
/// increments the reference counters of both `var` and `replacement` (and they
/// are decremented by oxidd_bcdd_substitution_free()). The order in which the
/// pairs are added is irrelevant.
///
/// Note that adding a new pair after applying the substitution may lead to
/// incorrect results when applying the substitution again.
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_substitution_add_pair(
    substitution: *mut bcdd_substitution_t,
    var: bcdd_t,
    replacement: bcdd_t,
) {
    assert!(!substitution.is_null(), "substitution must not be NULL");
    let v = var.get().expect("the variable function is invalid");
    let r = replacement
        .get()
        .expect("the replacement function is invalid");

    let subst = &mut *substitution;
    subst.pairs.push(((*v).clone(), (*r).clone()))
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
pub unsafe extern "C" fn oxidd_bcdd_exist(f: bcdd_t, var: bcdd_t) -> bcdd_t {
    op2(f, var, BCDDFunction::exist)
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
/// @returns  A satisfying assignment if there exists one. If `f` is
///           unsatisfiable, the data pointer is `NULL` and len is 0. In any
///           case, the assignment can be deallocated using
///           oxidd_assignment_free().
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_pick_cube(f: bcdd_t) -> assignment_t {
    let res = f.get().expect(FUNC_UNWRAP_MSG).pick_cube([], |_, _| false);
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

/// Pair of a BCDD function and a Boolean
#[repr(C)]
pub struct bcdd_bool_pair_t {
    /// The function
    func: bcdd_t,
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
/// @param  f         A *valid* BCDD function
/// @param  args      Array of pairs `(variable, value)`, where `variable` is
///                   valid
/// @param  num_args  Length of `args`
///
/// @returns  `f` evaluated with `args`
#[no_mangle]
pub unsafe extern "C" fn oxidd_bcdd_eval(
    f: bcdd_t,
    args: *const bcdd_bool_pair_t,
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
                    // `args` which outlives even the `with_manager_shared` closure
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
