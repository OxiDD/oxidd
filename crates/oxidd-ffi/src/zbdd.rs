use std::hash::BuildHasherDefault;
use std::mem::ManuallyDrop;

use oxidd::AllocResult;
use rustc_hash::FxHasher;

use oxidd::util::num::Saturating;
use oxidd::util::num::F64;
use oxidd::zbdd::ZBDDManagerRef;
use oxidd::zbdd::ZBDDSet;
use oxidd::BooleanFunction;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd::OutOfMemory;
use oxidd::RawFunction;
use oxidd::RawManagerRef;

use crate::oxidd_assignment_t;

/// cbindgen:ignore
const SET_UNWRAP_MSG: &str = "the given set is invalid";

/// Reference to a manager of a zero-suppressed decision diagram (ZBDD)
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_zbdd_manager_t {
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
/// This is essentially a reference to a ZBDD node.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct oxidd_zbdd_t {
    _p: *const std::ffi::c_void,
    _i: usize,
}

impl oxidd_zbdd_t {
    unsafe fn get(self) -> AllocResult<ManuallyDrop<ZBDDSet>> {
        if self._p.is_null() {
            Err(OutOfMemory)
        } else {
            Ok(ManuallyDrop::new(ZBDDSet::from_raw(self._p, self._i)))
        }
    }
}

impl From<ZBDDSet> for oxidd_zbdd_t {
    fn from(value: ZBDDSet) -> Self {
        let (_p, _i) = value.into_raw();
        Self { _p, _i }
    }
}

impl From<AllocResult<ZBDDSet>> for oxidd_zbdd_t {
    fn from(value: AllocResult<ZBDDSet>) -> Self {
        match value {
            Ok(f) => {
                let (_p, _i) = f.into_raw();
                Self { _p, _i }
            }
            Err(_) => Self {
                _p: std::ptr::null(),
                _i: 0,
            },
        }
    }
}

unsafe fn op1(f: oxidd_zbdd_t, op: impl FnOnce(&ZBDDSet) -> AllocResult<ZBDDSet>) -> oxidd_zbdd_t {
    f.get().and_then(|f| op(&f)).into()
}

unsafe fn op2(
    lhs: oxidd_zbdd_t,
    rhs: oxidd_zbdd_t,
    op: impl FnOnce(&ZBDDSet, &ZBDDSet) -> AllocResult<ZBDDSet>,
) -> oxidd_zbdd_t {
    lhs.get().and_then(|lhs| op(&lhs, &*rhs.get()?)).into()
}

unsafe fn op3(
    f1: oxidd_zbdd_t,
    f2: oxidd_zbdd_t,
    f3: oxidd_zbdd_t,
    op: impl FnOnce(&ZBDDSet, &ZBDDSet, &ZBDDSet) -> AllocResult<ZBDDSet>,
) -> oxidd_zbdd_t {
    f1.get()
        .and_then(|f1| op(&f1, &*f2.get()?, &*f3.get()?))
        .into()
}

/// Level number type
type oxidd_level_no_t = u32;

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

/// Increment the reference counter of the given ZBDD node
///
/// @returns  `f`
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_ref(f: oxidd_zbdd_t) -> oxidd_zbdd_t {
    std::mem::forget(f.get().clone());
    f
}

/// Decrement the reference count of the given ZBDD node
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_unref(f: oxidd_zbdd_t) {
    drop(ZBDDSet::from_raw(f._p, f._i));
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
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires an exclusive manager lock.
///
/// @returns  The ZBDD Boolean function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_new_singleton(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| ZBDDSet::new_singleton(manager).into())
}

/// Create a new ZBDD node at the level of `var` with the given `hi` and `lo`
/// edge
///
/// `var` must be a singleton set.
///
/// This function takes ownership of `hi` and `lo`. The reference counter of
/// `var` remains unchanged.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set representing the variable.
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
                .map(|e| ZBDDSet::from_edge(manager, e))
        })
    });
    res.into()
}

/// Get the ZBDD set `∅`
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_empty(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDSet::empty(manager).into())
}

/// Get the ZBDD set `{∅}`
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_base(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDSet::base(manager).into())
}

/// Get the set of subsets of `self` not containing `var`, formally
/// `{s ∈ set | var ∉ s}`
///
/// `var` must be a singleton set.
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_subset0(set: oxidd_zbdd_t, var: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(set, var, ZBDDSet::subset0)
}

/// Get the set of subsets of `self` containing `var`, formally
/// `{s ∈ self | var ∈ s}`
///
/// `var` must be a singleton set.
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_subset1(set: oxidd_zbdd_t, var: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(set, var, ZBDDSet::subset1)
}

/// Get the set of subsets derived from `self` by adding `var` to the
/// subsets that do not contain `var`, and removing `var` from the subsets
/// that contain `var`, formally
/// `{s ∪ {var} | s ∈ self ∧ var ∉ s} ∪ {s ∖ {var} | s ∈ self ∧ var ∈ s}`
///
/// `var` must be a singleton set.
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_change(set: oxidd_zbdd_t, var: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(set, var, ZBDDSet::change)
}

/// Compute the ZBDD for the union `lhs ∪ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_union(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::union)
}

/// Compute the ZBDD for the intersection `lhs ∩ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_intsec(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::intsec)
}

/// Compute the ZBDD for the set difference `lhs ∖ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD set with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_diff(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::diff)
}

/// Get a fresh variable, i.e. a Boolean function that is true if and only if
/// the variable is true. This adds a new level to a decision diagram.
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires an exclusive manager lock.
///
/// @returns  The ZBDD Boolean function representing the variable.
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_new_var(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_exclusive(|manager| ZBDDSet::new_var(manager).into())
}

/// Get the constant false ZBDD Boolean function `⊥`
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_false(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDSet::f(manager).into())
}

/// Get the constant true ZBDD Boolean function `⊤`
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_true(manager: oxidd_zbdd_manager_t) -> oxidd_zbdd_t {
    manager
        .get()
        .with_manager_shared(|manager| ZBDDSet::t(manager).into())
}

/// Compute the ZBDD for the negation `¬f`
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_not(f: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op1(f, ZBDDSet::not)
}

/// Compute the ZBDD for the conjunction `lhs ∧ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_and(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::and)
}

/// Compute the ZBDD for the disjunction `lhs ∨ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_or(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::or)
}

/// Compute the ZBDD for the negated conjunction `lhs ⊼ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_nand(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::nand)
}

/// Compute the ZBDD for the negated disjunction `lhs ⊽ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_nor(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::nor)
}

/// Compute the ZBDD for the exclusive disjunction `lhs ⊕ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_xor(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::xor)
}

/// Compute the ZBDD for the equivalence `lhs ↔ rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_equiv(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::equiv)
}

/// Compute the ZBDD for the implication `lhs → rhs` (or `self ≤ rhs`)
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_imp(lhs: oxidd_zbdd_t, rhs: oxidd_zbdd_t) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::imp)
}

/// Compute the ZBDD for the strict implication `lhs < rhs`
///
/// This function does not change the reference counters of its arguments.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The ZBDD Boolean function with its own reference count
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_imp_strict(
    lhs: oxidd_zbdd_t,
    rhs: oxidd_zbdd_t,
) -> oxidd_zbdd_t {
    op2(lhs, rhs, ZBDDSet::imp_strict)
}

/// Compute the ZBDD for the conditional `cond ? then_case : else_case`
///
/// This function does not change the reference counters of its arguments.
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
    op3(cond, then_case, else_case, ZBDDSet::ite)
}

/// Count nodes in `f`
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The node count including the two terminal nodes
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_node_count(f: oxidd_zbdd_t) -> usize {
    f.get().expect(SET_UNWRAP_MSG).node_count()
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The number of satisfying assignments or `UINT64_MAX` if the number
///           or some intermediate result is too large
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_sat_count_uint64(
    f: oxidd_zbdd_t,
    vars: oxidd_level_no_t,
) -> u64 {
    f.get()
        .expect(SET_UNWRAP_MSG)
        .sat_count::<Saturating<u64>, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Count the number of satisfying assignments, assuming `vars` input variables
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  The number of satisfying assignments
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_sat_count_double(
    f: oxidd_zbdd_t,
    vars: oxidd_level_no_t,
) -> f64 {
    f.get()
        .expect(SET_UNWRAP_MSG)
        .sat_count::<F64, BuildHasherDefault<FxHasher>>(vars, &mut Default::default())
        .0
}

/// Pick a satisfying assignment
///
/// This function does not change the reference counters of its argument.
///
/// Locking behavior: acquires a shared manager lock.
///
/// @returns  A satisfying assignment. If `f` is unsatisfiable, the data pointer
///           is `NULL` and len is 0. In any case, the assignment can be
///           deallocated using oxidd_assignment_free().
#[no_mangle]
pub unsafe extern "C" fn oxidd_zbdd_pick_cube(f: oxidd_zbdd_t) -> oxidd_assignment_t {
    let res = f.get().expect(SET_UNWRAP_MSG).pick_cube([], |_, _| false);
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
pub extern "C" fn oxidd_zbdd_print_stats() {
    oxidd::zbdd::print_stats();
}
