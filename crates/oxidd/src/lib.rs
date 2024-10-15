//! Default instantiation of `oxidd-rules-*` using `oxidd-manager-index/pointer`
//! and an apply cache implementation from `oxidd-cache`
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::upper_case_acronyms)]
#![warn(missing_docs)]

#[cfg(not(any(feature = "manager-index", feature = "manager-pointer")))]
std::compile_error!(
    "Either feature `manager-index` or `manager-pointer` must be enabled for this crate"
);

pub use oxidd_core::function::{
    BooleanFunction, BooleanFunctionQuant, BooleanOperator, BooleanVecSet, Function, FunctionSubst,
    NumberBase, PseudoBooleanFunction, TVLFunction,
};
pub use oxidd_core::util::{Subst, Substitution};
pub use oxidd_core::{
    Edge, HasWorkers, InnerNode, LevelNo, Manager, ManagerRef, NodeID, WorkerPool,
};

pub mod util;

#[deprecated = "use AllocResult from the oxidd::util module"]
#[doc(hidden)]
pub type AllocResult<T> = util::AllocResult<T>;
#[deprecated = "use OptBool from the oxidd::util module"]
#[doc(hidden)]
pub type OptBool = util::OptBool;
#[deprecated = "use OutOfMemory from the oxidd::util module"]
#[doc(hidden)]
pub type OutOfMemory = util::OutOfMemory;
#[deprecated = "use SatCountCache from the oxidd::util module"]
#[doc(hidden)]
pub type SatCountCache<N, S> = util::SatCountCache<N, S>;

#[cfg(feature = "bcdd")]
pub mod bcdd;
#[cfg(feature = "bdd")]
pub mod bdd;
#[cfg(feature = "mtbdd")]
pub mod mtbdd;
#[cfg(feature = "tdd")]
pub mod tdd;
#[cfg(feature = "zbdd")]
pub mod zbdd;

/// [`Function`] that can be converted into raw values. For use via the FFI
/// only.
#[doc(hidden)]
pub trait RawFunction {
    /// Convert `self` into a raw pointer and index, e.g. for usage in a
    /// foreign function interface.
    ///
    /// This method does not change any reference counters. To avoid a
    /// memory leak, use [`Self::from_raw()`] to convert pointer and index
    /// back into a `Self`.
    fn into_raw(self) -> (*const std::ffi::c_void, usize);

    /// Convert `raw` into `Self`
    ///
    /// # Safety
    ///
    /// `raw` and `index` must have been obtained via [`Self::into_raw()`]. This
    /// function does not change any reference counters, so calling this
    /// function multiple times for the same pointer may lead to use after free
    /// bugs depending on the usage of the returned `Self`.
    unsafe fn from_raw(ptr: *const std::ffi::c_void, index: usize) -> Self;
}

/// [`ManagerRef`] that can be converted into a raw value. For use via the FFI
/// only.
#[doc(hidden)]
pub trait RawManagerRef {
    /// Convert `self` into a raw pointer, e.g. for usage in a foreign
    /// function interface.
    ///
    /// This method does not change any reference counters. To avoid a
    /// memory leak, use [`Self::from_raw()`] to convert the pointer back
    /// into manager reference.
    fn into_raw(self) -> *const std::ffi::c_void;

    /// Convert `raw` into a `Self`
    ///
    /// # Safety
    ///
    /// `raw` must have been obtained via [`Self::into_raw()`]. This function
    /// does not change any reference counters, so calling this function
    /// multiple times for the same pointer may lead to use after free bugs
    /// depending on the usage of the returned manager reference.
    unsafe fn from_raw(raw: *const std::ffi::c_void) -> Self;
}

#[cfg(all(feature = "manager-pointer", not(miri)))]
const PAGE_SIZE: usize = 2 * 1024 * 1024;
#[cfg(all(feature = "manager-pointer", miri))]
const PAGE_SIZE: usize = 4 * 1024;
