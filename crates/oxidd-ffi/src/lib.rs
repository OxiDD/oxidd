//! All items of this crate are not intended to be used from other Rust crates
//! but only from other languages. Consult the C/C++ API documentation for
//! details on the provided types and functions.
#![allow(non_camel_case_types)]
#![allow(clippy::forget_non_drop)]

// These modules are not `pub` because they are not intended to be used from
// Rust code.
mod util;

mod bcdd;
mod bdd;
mod zbdd;
