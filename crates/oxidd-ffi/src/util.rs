/// Level number type
#[allow(non_camel_case_types)]
pub type oxidd_level_no_t = u32;

/// Boolean assignment
///
/// `data` is a pointer to `len` values. A value can be either 0 (false), 1
/// (true), or -1 (don't care).
#[repr(C)]
pub struct oxidd_assignment_t {
    /// Pointer to the data array of length `len`
    ///
    /// Must never be modified
    pub data: *mut i8,
    /// Length of the assignment
    ///
    /// Must never be modified
    pub len: usize,
}

/// Free the given assignment
///
/// To uphold Rust's invariants, all values in the assignment must be 0, 1, or
/// -1. `assignment.data` (i.e. the pointer value itself) and
#[no_mangle]
pub unsafe extern "C" fn oxidd_assignment_free(assignment: oxidd_assignment_t) {
    if !assignment.data.is_null() {
        drop(Vec::from_raw_parts(
            assignment.data,
            assignment.len,
            assignment.len,
        ))
    }
}
