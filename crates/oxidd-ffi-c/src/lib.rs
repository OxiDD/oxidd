//! All items of this crate are not intended to be used from other Rust crates
//! but only from other languages. Consult the C/C++ API documentation for
//! details on the provided types and functions.
#![allow(non_camel_case_types)]
#![allow(clippy::forget_non_drop)]

// Some general notes: cbindgen will prefix all types with `oxidd_`, and
// additionally rename:
//
// - `BooleanOperator` into `oxidd_boolean_operator`
// - `LevelNo` into `oxidd_level_no_t`
//
// Renaming is not applied in docstrings, we manually enter the C symbol name
// there. Furthermore, renaming does not work for function names, we manually
// prefix them with `oxidd_`.

// These modules are not `pub` because they are not intended to be used from
// Rust code.
mod util;

mod bcdd;
mod bdd;
mod zbdd;
