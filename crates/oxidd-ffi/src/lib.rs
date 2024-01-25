#![allow(non_camel_case_types)]
#![allow(clippy::forget_non_drop)]

// These modules are not `pub` because they are not intended to be used from
// Rust code.
mod bdd;
mod cbdd;
mod zbdd;

/// Boolean assignment
///
/// `data` is a pointer to `len` values. A value can be either 0 (false), 1
/// (true), or -1 (don't care).
#[repr(C)]
#[doc(hidden)]
pub struct oxidd_assignment_t {
    data: *mut i8,
    len: usize,
}

/// Free the given assignment
///
/// To uphold Rust's invariants, all values in the assignment must be 0, 1, or
/// -1.
#[no_mangle]
#[doc(hidden)]
pub unsafe extern "C" fn oxidd_assignment_free(assignment: oxidd_assignment_t) {
    drop(Vec::from_raw_parts(
        assignment.data,
        assignment.len,
        assignment.len,
    ))
}
