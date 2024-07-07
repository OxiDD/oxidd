use oxidd_core::util::Substitution;

/// cbindgen:ignore
pub const FUNC_UNWRAP_MSG: &str = "the given function is invalid";

/// Boolean assignment
///
/// `data` is a pointer to `len` values. A value can be either 0 (false), 1
/// (true), or -1 (don't care).
#[repr(C)]
pub struct assignment_t {
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
/// -1. `assignment.data` (i.e., the pointer value itself) and
/// `assignment.length` must not be modified.
///
/// In case `assignment.data` is `NULL`, this is a no-op.
#[no_mangle]
pub unsafe extern "C" fn oxidd_assignment_free(assignment: assignment_t) {
    if !assignment.data.is_null() {
        drop(Vec::from_raw_parts(
            assignment.data,
            assignment.len,
            assignment.len,
        ))
    }
}

pub(crate) struct Subst<'a, V, R> {
    pub id: u32,
    pub pairs: &'a [(V, R)],
}

impl<'a, V, R> Substitution for Subst<'a, V, R> {
    type Var = &'a V;
    type Replacement = &'a R;

    #[inline]
    fn id(&self) -> u32 {
        self.id
    }
    #[inline]
    fn pairs(&self) -> impl ExactSizeIterator<Item = (&'a V, &'a R)> {
        self.pairs.iter().map(|(v, r)| (v, r))
    }
}
