//! Default instantiation of `oxidd-rules-*` using `oxidd-manager-index/pointer`
//! and an apply cache implementation from `oxidd-cache`
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::upper_case_acronyms)]

#[cfg(not(any(feature = "manager-index", feature = "manager-pointer")))]
std::compile_error!(
    "Either feature `manager-index` or `manager-pointer` must be enabled for this crate"
);

pub use oxidd_core::function::BooleanFunction;
pub use oxidd_core::function::BooleanFunctionQuant;
pub use oxidd_core::function::BooleanVecSet;
pub use oxidd_core::function::Function;
pub use oxidd_core::function::NumberBase;
pub use oxidd_core::function::PseudoBooleanFunction;
pub use oxidd_core::function::TVLFunction;
pub use oxidd_core::util::AllocResult;
pub use oxidd_core::util::IsFloatingPoint;
pub use oxidd_core::util::OptBool;
pub use oxidd_core::util::OutOfMemory;
pub use oxidd_core::util::SatCountNumber;
pub use oxidd_core::LevelNo;
pub use oxidd_core::Manager;
pub use oxidd_core::ManagerRef;
pub use oxidd_core::NodeID;

pub mod util;

#[cfg(feature = "bdd")]
pub mod bdd;
#[cfg(feature = "cbdd")]
pub mod cbdd;
#[cfg(feature = "mtbdd")]
pub mod mtbdd;
#[cfg(feature = "tdd")]
pub mod tdd;
#[cfg(feature = "zbdd")]
pub mod zbdd;

pub trait RawFunction {
    /// Convert `self` into a raw pointer and index, e.g. for usage in a
    /// foreign function interface.
    ///
    /// This method does not change any reference counters. To avoid a
    /// memory leak, use [`Self::from_raw()`] to convert pointer and index
    /// back into a `Self`.
    fn into_raw(self) -> (*const std::ffi::c_void, u32);

    /// Convert `raw` into `Self`
    ///
    /// # Safety
    ///
    /// `raw` and `index` must have been obtained via [`Self::into_raw()`]. This
    /// function does not change any reference counters, so calling this
    /// function multiple times for the same pointer may lead to use after free
    /// bugs depending on the usage of the returned `Self`.
    unsafe fn from_raw(ptr: *const std::ffi::c_void, index: u32) -> Self;
}

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
