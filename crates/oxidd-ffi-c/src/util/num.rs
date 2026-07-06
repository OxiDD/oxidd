use std::mem::ManuallyDrop;

use oxidd::util::num::Natural;

use crate::util::{partial_ordering, string_t};

/// Natural number, specifically well-suited for counting satisfying assignments
///
/// When counting the number of satisfying assignments in a decision diagram,
/// the numbers often have many trailing zeros (in binary representation). Based
/// on this observation, we decompose numbers as *m* × 2^<sup>e</sup> with a
/// mantissa *m* and an exponent *e*. We require that both *m* and *e* are
/// natural numbers. Further, *m* is odd unless the represented number is zero,
/// in which case both *m* and *e* are zero.
///
/// The exponent *e* is stored as a `uint64_t` and the mantissa *m* as an array
/// of `uint64_t` "digits." In case *m* fits into a single `uint64_t` digit, *m*
/// is stored inline, i.e., without any heap allocation.
///
/// Conceptually, this type is similar to arbitrary precision floats. However,
/// we do not require the user to choose the precision, instead we always
/// represent numbers exactly.
///
/// To account for computation errors (e.g., an exponent that cannot be
/// represented as a `uint64_t`). this type includes a NaN value. It is
/// undefined if this value is larger or smaller than actual natural numbers.
#[repr(C)]
pub struct natural_t {
    /// Unless this pointer is `NULL`, it points to an array of `len` `uint64_t`
    /// "digits." The array starts with the least significant digit (i.e.,
    /// little endian). If the most significant digit is 0, the most significant
    /// bit of the second-most significant digit is 1. The least significant bit
    /// of the least significant digit in the array is always 1. Further, the
    /// number represented by the array (i.e., disregarding `shl`) is
    /// greater than `UINT64_MAX`.
    ///
    /// Allowing the most significant to be 0 avoids the need to re-allocate
    /// memory in the addition of two numbers: We can predict the sum's bit
    /// width almost exactly in constant time, we only don't know whether there
    /// is a final carry if the operands have different bit width. Hence, we
    /// just always assume that there will be a carry and potentially allocate
    /// one more digit than needed.
    ptr: *mut u64,
    /// If `ptr` is not `NULL`, then `len` represents the length of the array to
    /// which `ptr` points. Otherwise the number represented by the
    /// `oxidd_natural_t` is `len << shl`. If `len != 0`, then the least
    /// significant bit of `len` is 1.
    len: u64,
    /// Trailing zeros of the represented number (in binary representation). If
    /// `UINT64_MAX`, this indicates a computation error (NaN).
    shl: u64,
}

impl From<Natural> for natural_t {
    fn from(value: Natural) -> Self {
        let (ptr, len, shl) = value.into_raw_parts();
        Self { ptr, len, shl }
    }
}

impl Clone for natural_t {
    fn clone(&self) -> Self {
        if self.ptr.is_null() {
            return Self { ..*self };
        }
        let slice = std::ptr::slice_from_raw_parts(self.ptr, self.len as usize);
        // SAFETY: `self.ptr` is not dangling, thus the pointer is valid and we
        // have shared access to the slice
        let clone: *mut [u64] = Box::into_raw(unsafe { &*slice }.into());
        Self {
            ptr: clone.cast(),
            ..*self
        }
    }
}

impl Drop for natural_t {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let slice = std::ptr::slice_from_raw_parts_mut(self.ptr, self.len as usize);
            // SAFETY: ptr is not null, thus the pointer is valid and we own the slice
            drop(unsafe { Box::from_raw(slice) });
        }
    }
}

/// Free the given natural number
#[unsafe(no_mangle)]
extern "C" fn oxidd_natural_free(num: natural_t) {
    drop(num);
}

/// Check if `lhs` and `rhs` are equal
///
/// @param  lhs  The left-hand side operand
/// @param  rhs  The right-hand side operand
///
/// @returns  `true` if, and only if `lhs` and `rhs` are equal
#[unsafe(no_mangle)]
extern "C" fn oxidd_natural_eq(lhs: &natural_t, rhs: &natural_t) -> bool {
    let lhs = ManuallyDrop::new(unsafe { Natural::from_raw_parts(lhs.ptr, lhs.len, lhs.shl) });
    let rhs = ManuallyDrop::new(unsafe { Natural::from_raw_parts(rhs.ptr, rhs.len, rhs.shl) });
    lhs == rhs
}

/// Compare `lhs` with `rhs`
///
/// @param  lhs  The left-hand side operand
/// @param  rhs  The right-hand side operand
///
/// @returns  Whether `lhs` is less than, equal to, greater than, or
///           incomparable to `rhs`
#[unsafe(no_mangle)]
extern "C" fn oxidd_natural_cmp(lhs: &natural_t, rhs: &natural_t) -> partial_ordering {
    let lhs = ManuallyDrop::new(unsafe { Natural::from_raw_parts(lhs.ptr, lhs.len, lhs.shl) });
    let rhs = ManuallyDrop::new(unsafe { Natural::from_raw_parts(rhs.ptr, rhs.len, rhs.shl) });
    lhs.partial_cmp(&rhs).into()
}

/// Convert `num` to a string in decimal representation
#[unsafe(no_mangle)]
extern "C" fn oxidd_natural_to_string(num: &natural_t) -> string_t {
    let num = ManuallyDrop::new(unsafe { Natural::from_raw_parts(num.ptr, num.len, num.shl) });
    num.to_string().into()
}

/// Clone `num`
///
/// The returned `oxidd_natural_t` must be deallocated independently of `num`,
/// see [`oxidd_natural_free()`].
#[unsafe(no_mangle)]
extern "C" fn oxidd_natural_clone(num: &natural_t) -> natural_t {
    num.clone()
}
