// spell-checker:ignore trunc

use std::cmp::Ordering;
use std::fmt::Write;
use std::mem::ManuallyDrop;
use std::ops::{Add, Shl, Shr};
use std::ptr::NonNull;
use std::{fmt, mem};

use crate::util::IsFloatingPoint;

/// Natural number, specifically well-suited for counting satisfying assignments
///
/// When counting the number of satisfying assignments in a decision diagram,
/// the numbers often have many trailing zeros (in binary representation). Based
/// on this observation, we decompose numbers as *m* × 2^<sup>e</sup> with a
/// mantissa *m* and an exponent *e*. We require that both *m* and *e* are
/// natural numbers. Further, *m* is odd unless the represented number is zero,
/// in which case both *m* and *e* are zero.
///
/// The exponent *e* is stored as a [`u64`] and the mantissa *m* as an array of
/// [`u64`] "digits." In case *m* fits into a single [`u64`] digit, *m* is
/// stored inline, i.e., without any heap allocation.
///
/// Conceptually, this type is similar to arbitrary precision floats. However,
/// we do not require the user to choose the precision, instead we always
/// represent numbers exactly.
///
/// To account for computation errors (e.g., an exponent that cannot be
/// represented as a [`u64`]). this type includes a NaN value. It is undefined
/// if this value is larger or smaller than actual natural numbers. Therefore,
/// this type does not implement [`Ord`] but only [`PartialOrd`].
///
/// Also note that the implementation of [`Shr`] is tailored to the application
/// in counting satisfying assignments, where it is used as an exact division by
/// (a power of) two. If a 1-bit would get lost by a shift to the right, i.e.,
/// the division result would not be exact, the computation result is NaN to
/// make the error obvious. A context for such an error may be pretending that a
/// Boolean function has a smaller domain than it actually has.
#[derive(Eq)]
pub struct Natural {
    /// Unless this pointer is `DANGLING`, it points to an array of `len` `u64`
    /// "digits." The array starts with the least significant digit (i.e.,
    /// little endian). If the most significant digit is 0, the most significant
    /// bit of the second-most significant digit is 1. The least significant bit
    /// of the least significant digit in the array is always 1. Further, the
    /// number represented by the array (i.e., disregarding `shl`) is
    /// greater than `u64::MAX`.
    ///
    /// Allowing the most significant to be 0 avoids the need to re-allocate
    /// memory in the addition of two numbers: We can predict the sum's bit
    /// width almost exactly in constant time, we only don't know whether there
    /// is a final carry if the operands have different bit width. Hence, we
    /// just always assume that there will be a carry and potentially allocate
    /// one more digit than needed.
    ptr: std::ptr::NonNull<u64>,
    /// If `ptr` is not `DANGLING`, then `len` represents the length of
    /// the array to which `ptr` points. Otherwise the number represented by the
    /// [`Natural`] is `len << shl`. If `len != 0`, then the least significant
    /// bit of `len` is 1.
    len: u64,
    /// Trailing zeros of the represented number (in binary representation). If
    /// `u64::MAX`, this indicates a computation error (NaN).
    shl: u64,
}

/// An unaligned, dangling pointer
const DANGLING: NonNull<u64> = {
    assert!(
        std::mem::align_of::<u64>() > 1,
        "The implementation of `Natural` assumes that `u64` has at least 2 byte alignment"
    );
    NonNull::without_provenance(std::num::NonZeroUsize::new(1).unwrap())
};

#[inline]
fn shl_amount(value: u64) -> u32 {
    if value == 0 {
        0
    } else {
        value.trailing_zeros()
    }
}

#[inline]
fn bit_width(digits: &[u64], shl: u64) -> u128 {
    (u64::BITS as u128 * digits.len() as u128) - digits.last().unwrap().leading_zeros() as u128
        + shl as u128
}

impl Natural {
    /// The number 0
    pub const ZERO: Self = Self {
        ptr: DANGLING,
        len: 0,
        shl: 0,
    };
    /// cbindgen:ignore
    const NAN: Self = Self {
        ptr: DANGLING,
        len: 0,
        shl: u64::MAX,
    };

    /// Decompose a [`Natural`] into its raw parts
    ///
    /// The first tuple component is either [`std::ptr::null_mut()`] and the
    /// mantissa is just the second tuple component, or the first component
    /// points to an array with the number of elements specified by the second
    /// component. This array contains the mantissa digits, starting with the
    /// least significant one.
    ///
    /// The third component is the exponent, or [`u64::MAX`] to represent a
    /// failed computation (NaN).
    ///
    /// This method is primarily intended for use with foreign function
    /// interfaces and thus hidden from the documentation.
    ///
    /// See [`Self::from_raw_parts()`] for the inverse operation.
    #[inline]
    #[doc(hidden)]
    pub fn into_raw_parts(self) -> (*mut u64, u64, u64) {
        let ptr = if self.ptr == DANGLING {
            std::ptr::null_mut()
        } else {
            self.ptr.as_ptr()
        };
        let this = ManuallyDrop::new(self);
        (ptr, this.len, this.shl)
    }

    /// Create a [`Natural`] from its raw parts
    ///
    /// If `ptr` is [`std::ptr::null_mut()`], then `len` is the mantissa.
    /// Otherwise `ptr` points to `len` digits (starting with the least
    /// significant one) describing the mantissa. `exp` is the exponent, or
    /// [`u64::MAX`] to represent a failed computation (NaN).
    ///
    /// This method is primarily intended for use with foreign function
    /// interfaces and thus hidden from the documentation.
    ///
    /// See [`Self::into_raw_parts()`] for the inverse operation.
    #[inline]
    #[doc(hidden)]
    pub unsafe fn from_raw_parts(ptr: *mut u64, len: u64, exp: u64) -> Self {
        let ptr = NonNull::new(ptr).unwrap_or(DANGLING);
        Self { ptr, len, shl: exp }
    }

    #[inline]
    fn from_mantissa_single_with_shl(mantissa: u64, shl: u64) -> Self {
        Self {
            ptr: DANGLING,
            len: mantissa,
            shl,
        }
    }

    #[inline]
    fn from_mantissa_with_shl(mantissa: Box<[u64]>, shl: u64) -> Self {
        let len = mantissa.len() as u64;
        let ptr = NonNull::new(Box::into_raw(mantissa).cast()).unwrap();
        Self { ptr, len, shl }
    }

    /// Create a `Natural` from a sequence of digits
    ///
    /// Here, `le` stands for *little endian*, i.e., the first element of
    /// `digits` is the least significant digit.
    pub fn from_le_digits(digits: &[u64]) -> Self {
        let mut digits = digits;
        while let [r @ .., 0] = digits {
            digits = r;
        }
        let mut shl_digits = 0u64;
        while let [0, r @ ..] = digits {
            digits = r;
            shl_digits += 1;
        }
        match digits {
            [] => Self::ZERO,
            &[d] => {
                debug_assert_ne!(d, 0);
                let shl = d.trailing_zeros();
                Self {
                    ptr: DANGLING,
                    len: d >> shl,
                    shl: (shl as u64).saturating_add(shl_digits.saturating_mul(u64::BITS as u64)),
                }
            }
            &[lsd, .., msd] => {
                debug_assert_ne!(lsd, 0);
                let shr_bits = lsd.trailing_zeros();
                let shl = shl_digits.saturating_mul(u64::BITS as u64);
                if shr_bits == 0 {
                    return Self {
                        ptr: NonNull::new(Box::<[u64]>::into_raw(digits.into()).cast()).unwrap(),
                        len: digits.len() as u64,
                        shl,
                    };
                }
                let shl = shl.saturating_add(shr_bits as u64);

                let len = digits.len() - ((shr_bits + msd.leading_zeros()) / u64::BITS) as usize;
                if len == 1 {
                    return Self {
                        ptr: DANGLING,
                        len: msd.rotate_right(shr_bits) | (lsd >> shr_bits),
                        shl,
                    };
                }

                let lower_mask = u64::MAX >> shr_bits;
                let upper_mask = !lower_mask;
                let mut v = Vec::with_capacity(len);
                let mut lower = lsd >> shr_bits;
                v.extend(digits[1..].iter().map(|&d| {
                    let rot = d.rotate_right(shr_bits);
                    let d = lower | (rot & upper_mask);
                    lower = rot & lower_mask;
                    d
                }));
                if v.len() != len {
                    v.push(lower);
                }
                debug_assert_eq!(v.len(), len);

                Self {
                    ptr: NonNull::new(Box::into_raw(v.into_boxed_slice()).cast()).unwrap(),
                    len: len as u64,
                    shl,
                }
            }
        }
    }

    /// Get the mantissa
    ///
    /// The returned sequence starts with the least significant digit and has
    /// minimal length, but at least one element. So unless the mantissa is 0,
    /// the most significant digit (i.e., the last element) is non-zero.
    #[inline]
    pub fn mantissa(&self) -> &[u64] {
        let digits = if self.ptr == DANGLING {
            std::slice::from_ref(&self.len)
        } else {
            // SAFETY: `self.ptr` is not `DANGLING`
            let digits =
                unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) };
            // SAFETY: follows from the representation invariant
            if *unsafe { digits.last().unwrap_unchecked() } == 0 {
                &digits[..self.len as usize - 1]
            } else {
                digits
            }
        };
        // SAFETY: follows from the representation invariant
        unsafe { std::hint::assert_unchecked(!digits.is_empty()) };
        digits
    }

    /// Get the mantissa
    ///
    /// Note that unlike [`Self::mantissa()`], this method returns a full view
    /// on the array (i.e., it does not remove any leading zero).
    #[inline]
    fn mantissa_raw(&self) -> &[u64] {
        let digits = if self.ptr == DANGLING {
            std::slice::from_ref(&self.len)
        } else {
            // SAFETY: `self.ptr` is not `DANGLING`
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
        };
        // SAFETY: follows from the representation invariant
        unsafe { std::hint::assert_unchecked(!digits.is_empty()) };
        digits
    }

    /// Get a mutable reference to the mantissa
    ///
    /// Note that unlike [`Self::mantissa()`], this method returns a full view
    /// on the array (i.e., it does not remove any leading zero).
    #[inline]
    fn mantissa_mut(&mut self) -> &mut [u64] {
        let digits = if self.ptr == DANGLING {
            std::slice::from_mut(&mut self.len)
        } else {
            // SAFETY: `self.ptr` is not `DANGLING`
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize) }
        };
        // SAFETY: follows from the representation invariant
        unsafe { std::hint::assert_unchecked(!digits.is_empty()) };
        digits
    }

    /// Get the exponent
    #[inline(always)]
    pub fn exp(&self) -> u64 {
        self.shl
    }

    /// Check if the number is NaN
    #[inline(always)]
    pub fn is_nan(&self) -> bool {
        self.shl == u64::MAX
    }

    /// Compute the number of bits needed to represent the number explicitly,
    /// that is *1 + floor(log₂(self))*.
    pub fn bit_width(&self) -> u128 {
        bit_width(self.mantissa_raw(), self.shl)
    }
}

impl Drop for Natural {
    fn drop(&mut self) {
        if self.ptr != DANGLING {
            let slice = std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize);
            // SAFETY: ptr is not dangling, thus the pointer is valid and we own the slice
            drop(unsafe { Box::from_raw(slice) });
        }
    }
}

// SAFETY: we uniquely own the referenced memory
unsafe impl Send for Natural {}
unsafe impl Sync for Natural {}

impl Clone for Natural {
    fn clone(&self) -> Self {
        if self.ptr == DANGLING {
            return Self { ..*self };
        }
        let slice = std::ptr::slice_from_raw_parts(self.ptr.as_ptr(), self.len as usize);
        // SAFETY: `self.ptr` is not dangling, thus the pointer is valid and we
        // have shared access to the slice
        let clone: *mut [u64] = Box::into_raw(unsafe { &*slice }.into());
        Self {
            ptr: NonNull::new(clone.cast()).unwrap(),
            ..*self
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.shl = source.shl;
        if self.ptr == DANGLING {
            if source.ptr != DANGLING {
                let slice = std::ptr::slice_from_raw_parts(source.ptr.as_ptr(), self.len as usize);
                // SAFETY: `source.ptr` is not dangling, thus the pointer is
                // valid and we have shared access to the slice
                let clone: *mut [u64] = Box::into_raw(unsafe { &*slice }.into());
                self.ptr = NonNull::new(clone.cast()).unwrap();
            }
            self.len = source.len;
            return;
        }

        let dst = std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize);
        if source.ptr != DANGLING {
            // SAFETY: `source.ptr` is not dangling, thus the pointer is valid
            // and we have shared access to the slice
            let src =
                unsafe { std::slice::from_raw_parts(source.ptr.as_ptr(), source.len as usize) };
            if self.len == source.len {
                // SAFETY: `self.ptr` is not dangling, thus the pointer is valid
                // and we own the slice
                unsafe { &mut *dst }.copy_from_slice(src);
                return;
            }
            self.ptr = NonNull::new(Box::<[u64]>::into_raw(src.into()).cast()).unwrap();
        }
        self.len = source.len;

        // SAFETY: `self.ptr` is not dangling, thus the pointer is valid and we
        // own the slice
        drop(unsafe { Box::from_raw(dst) });
    }
}

impl PartialEq for Natural {
    fn eq(&self, other: &Self) -> bool {
        if self.shl != other.shl {
            return false;
        }
        if self.is_nan() {
            return true; // both are `None`-alike
        }
        self.mantissa() == other.mantissa()
    }
}
impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }

        let l_digits = self.mantissa();
        let r_digits = other.mantissa();

        let l_bw = bit_width(l_digits, self.shl);
        let r_bw = bit_width(r_digits, other.shl);
        if l_bw != r_bw {
            return Some(l_bw.cmp(&r_bw));
        }

        let (&l_msd, mut l_digits) = l_digits.split_last().unwrap();
        let (&r_msd, mut r_digits) = r_digits.split_last().unwrap();
        let l_shl = l_msd.leading_zeros();
        let r_shl = r_msd.leading_zeros();
        let mut l = l_msd << l_shl;
        let mut r = r_msd << r_shl;

        let l_upper_mask = u64::MAX << l_shl;
        let l_lower_mask = !l_upper_mask;
        let r_upper_mask = u64::MAX << r_shl;
        let r_lower_mask = !r_upper_mask;
        while let ([l_rest @ .., l_next], [r_rest @ .., r_next]) = (l_digits, r_digits) {
            let l_rot = l_next.rotate_left(l_shl);
            let r_rot = r_next.rotate_left(r_shl);
            let l_digit = l | (l_rot & l_lower_mask);
            let r_digit = r | (r_rot & r_lower_mask);

            if l_digit != r_digit {
                return Some(l_digit.cmp(&r_digit));
            }

            l = l_rot & l_upper_mask;
            r = r_rot & r_upper_mask;
            l_digits = l_rest;
            r_digits = r_rest;
        }

        Some(match (l_digits, r_digits) {
            ([], []) => l.cmp(&r),
            ([.., l_next], _) => {
                debug_assert!(r_digits.is_empty());
                let cmp = (l | (l_next.rotate_left(l_shl) & l_lower_mask)).cmp(&r);
                // The rest of `l_digits` contains at least one non-zero bit,
                // while the remaining bits (after unfolding `other.shl`) are
                // all zero.
                cmp.then(Ordering::Greater)
            }
            (_, [.., r_next]) => {
                let cmp = l.cmp(&(r | (r_next.rotate_left(r_shl) & r_lower_mask)));
                cmp.then(Ordering::Less)
            }
        })
    }
}

impl std::hash::Hash for Natural {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.is_nan() {
            1.hash(state);
        } else {
            self.shl.hash(state);
            self.mantissa().hash(state);
        }
    }
}

/// cbindgen:ignore
impl IsFloatingPoint for Natural {
    const FLOATING_POINT: bool = false;
    const MIN_EXP: i32 = 0;
}

impl From<u128> for Natural {
    #[inline]
    fn from(value: u128) -> Self {
        if value == 0 {
            return Self::ZERO;
        }
        let leading = value.leading_zeros();
        let shl = value.trailing_zeros();
        if leading - shl <= u64::BITS {
            Self::from_mantissa_single_with_shl((value >> shl) as u64, shl as u64)
        } else {
            let value = value >> shl;
            Self::from_mantissa_with_shl(
                [value as u64, (value >> u64::BITS) as u64].into(),
                shl as u64,
            )
        }
    }
}
impl From<u64> for Natural {
    #[inline]
    fn from(value: u64) -> Self {
        let shl = shl_amount(value);
        Self::from_mantissa_single_with_shl(value >> shl, shl as u64)
    }
}
impl From<u32> for Natural {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self::from(value as u64)
    }
}
impl From<u16> for Natural {
    #[inline(always)]
    fn from(value: u16) -> Self {
        Self::from(value as u64)
    }
}
impl From<u8> for Natural {
    #[inline(always)]
    fn from(value: u8) -> Self {
        Self::from(value as u64)
    }
}

impl Add for Natural {
    type Output = Self;
    fn add(mut self, mut rhs: Self) -> Self {
        if rhs.len == 0 {
            return self;
        }
        if self.len == 0 {
            return rhs;
        }

        if self.shl > rhs.shl {
            mem::swap(&mut self, &mut rhs);
        }

        let l_shl = self.shl;
        let r_shl = rhs.shl;
        let l_digits = self.mantissa_mut();
        let r_digits = rhs.mantissa_mut();

        let l_bit_width = bit_width(l_digits, l_shl);
        let r_bit_width = bit_width(r_digits, r_shl);
        // target bit width (may be one less if `l_bit_width != r_bit_width`)
        let bit_width = std::cmp::max(l_bit_width, r_bit_width) + 1;

        if l_shl < r_shl {
            if r_shl == u64::MAX {
                return Self::NAN;
            }

            // mantissa bit/digit count; possibly one too large each
            let bit_len = bit_width - l_shl as u128;
            let len = bit_len.div_ceil(u64::BITS as u128) as usize;
            debug_assert_ne!(len, 0);

            let start_digit = ((r_shl - l_shl) / u64::BITS as u64) as usize;
            let start_bit = (r_shl - l_shl) as u32 % u64::BITS;
            let upper_mask = u64::MAX << start_bit;
            let lower_mask = !upper_mask;

            if bit_len <= (u64::BITS + 1) as u128 {
                debug_assert_eq!(start_digit, 0);
                debug_assert_eq!(self.ptr, DANGLING);
                debug_assert_eq!(rhs.ptr, DANGLING);
                debug_assert_eq!(rhs.len.rotate_left(start_bit) & lower_mask, 0);
                let (d, carry) = self.len.overflowing_add(rhs.len << start_bit);
                return if carry {
                    Self::from_mantissa_with_shl([d, 1].into(), l_shl)
                } else {
                    Self::from_mantissa_single_with_shl(d, l_shl)
                };
            }

            if len == l_digits.len() {
                // we can (likely) update in-place
                debug_assert!(r_digits.len() + start_digit <= l_digits.len());
                let mut lower = 0;
                let mut carry = false;
                for (l, r) in l_digits[start_digit..].iter_mut().zip(&r_digits[..]) {
                    let rot = r.rotate_left(start_bit);
                    (*l, carry) = l.carrying_add(rot & upper_mask | lower, carry);
                    lower = rot & lower_mask;
                }
                let i = start_digit + r_digits.len();
                if let Some(l) = l_digits.get_mut(i) {
                    (*l, carry) = l.carrying_add(lower, carry);
                    lower = 0;
                    for l in &mut l_digits[i + 1..] {
                        (*l, carry) = l.overflowing_add(carry as u64);
                    }
                }
                debug_assert_eq!(lower, 0);
                debug_assert!(!carry);
                return self;
            }

            let mut vec = Vec::with_capacity(len);
            if start_digit >= l_digits.len() {
                vec.extend_from_slice(l_digits);
                if start_digit > l_digits.len() {
                    vec.extend((l_digits.len()..start_digit).map(|_| 0));
                }
                if start_bit == 0 {
                    vec.extend_from_slice(r_digits);
                } else {
                    let mut lower = 0;
                    vec.extend(r_digits.iter().map(|&r| {
                        let rot = r.rotate_left(start_bit);
                        rot & upper_mask | mem::replace(&mut lower, rot & lower_mask)
                    }));
                    if lower != 0 {
                        vec.push(lower);
                    }
                }
            } else {
                vec.extend_from_slice(&l_digits[..start_digit]);
                let mut lower = 0;
                let mut carry = false;
                let zipped = l_digits[start_digit..].iter().zip(r_digits.iter());
                vec.extend(zipped.map(|(l, r)| {
                    let rot = r.rotate_left(start_bit);
                    let res = l.carrying_add(rot & upper_mask | lower, carry);
                    carry = res.1;
                    lower = rot & lower_mask;
                    res.0
                }));
                let l_i = start_digit + r_digits.len(); // assuming `r_digits` is exhausted
                let r_i = l_digits.len() - start_digit; // assuming `l_digits` is exhausted
                if let Some(l) = l_digits.get(l_i) {
                    let res = l.carrying_add(lower, carry);
                    vec.push(res.0);
                    carry = res.1;
                    lower = 0;
                    vec.extend(l_digits[l_i + 1..].iter().map(|l| {
                        let res = l.overflowing_add(carry as u64);
                        carry = res.1;
                        res.0
                    }));
                } else if r_i < r_digits.len() {
                    vec.extend(r_digits[r_i..].iter().map(|r| {
                        let rot = r.rotate_left(start_bit);
                        let res = (rot & upper_mask | lower).overflowing_add(carry as u64);
                        lower = rot & lower_mask;
                        carry = res.1;
                        res.0
                    }));
                }
                if vec.len() != vec.capacity() {
                    vec.push(lower + carry as u64);
                } else {
                    debug_assert_eq!(lower + carry as u64, 0);
                }
            }

            debug_assert_eq!(vec.capacity(), vec.len());
            return Self::from_mantissa_with_shl(vec.into_boxed_slice(), l_shl);
        }

        debug_assert_eq!(l_shl, r_shl);

        // the least significant bits cancel each other
        // -> determine the new left shift
        let (mut lsd, mut carry) = l_digits[0].overflowing_add(r_digits[0]);
        let mut i_in = 1;
        while lsd == 0 {
            let l = l_digits.get(i_in).copied().unwrap_or_default();
            let r = r_digits.get(i_in).copied().unwrap_or_default();
            i_in += 1;
            debug_assert!(carry);
            (lsd, carry) = l.carrying_add(r, true);
        }
        let bit_shr = shl_amount(lsd);
        let Some(shl) = l_shl
            .checked_add(bit_shr as u64)
            .and_then(|sum| sum.checked_add((i_in as u64 - 1).checked_mul(u64::BITS as u64)?))
        else {
            return Self::NAN;
        };
        // mantissa bit/digit count; possibly one too large each
        let bit_len = bit_width - shl as u128;
        debug_assert_ne!(bit_len, 0);
        let len = bit_len.div_ceil(u64::BITS as u128) as usize;

        let l = l_digits.get(i_in).copied().unwrap_or_default();
        let r = r_digits.get(i_in).copied().unwrap_or_default();
        i_in += 1;
        let (next_lsd, mut carry) = l.carrying_add(r, carry);
        let mut rot = next_lsd.rotate_right(bit_shr);

        let lower_mask = u64::MAX >> bit_shr;
        let upper_mask = !lower_mask;
        let d = (lsd >> bit_shr) | (rot & upper_mask);

        // `l_digits` and `r_digits` may be arbitrarily long, while the sum may
        // fit into a single digit. We also account for the case in which
        // `bit_len` is over-estimated by a bit.
        if bit_len <= (u64::BITS + 1) as u128 && rot & lower_mask == 0 {
            debug_assert!(!carry);
            return Self::from_mantissa_single_with_shl(d, shl);
        }
        debug_assert!(len >= 2);

        if len == l_digits.len() {
            // we can (likely) update in-place
            l_digits[0] = d;
            let mut i_out = 1;
            while let Some(l) = l_digits.get(i_in)
                && let Some(r) = r_digits.get(i_in)
            {
                i_in += 1;
                let res = l.carrying_add(*r, carry);
                let next_rot = res.0.rotate_right(bit_shr);
                carry = res.1;
                l_digits[i_out] = (rot & lower_mask) | (next_rot & upper_mask);
                i_out += 1;
                rot = next_rot;
            }

            if i_in < r_digits.len() {
                // This loop can run for up to `l_digits.len()` iterations,
                // e.g., if the position of sum's least significant bit with
                // value 1 corresponds to the position of `l_digits`' most
                // significant bit.
                for r in &r_digits[i_in..] {
                    let res = r.overflowing_add(carry as u64);
                    let next_rot = res.0.rotate_right(bit_shr);
                    carry = res.1;
                    l_digits[i_out] = (rot & lower_mask) | (next_rot & upper_mask);
                    i_out += 1;
                    if i_out == len {
                        self.shl = shl;
                        return self;
                    }
                    rot = next_rot;
                }
            } else {
                while let Some(l) = l_digits.get(i_in) {
                    i_in += 1;
                    let res = l.overflowing_add(carry as u64);
                    let next_rot = res.0.rotate_right(bit_shr);
                    carry = res.1;
                    l_digits[i_out] = (rot & lower_mask) | (next_rot & upper_mask);
                    i_out += 1;
                    rot = next_rot;
                }
            }

            if bit_shr == 0 {
                debug_assert_eq!(lower_mask, u64::MAX);
                l_digits[i_out] = rot;
                if i_out + 1 != l_digits.len() {
                    i_out += 1;
                    l_digits[i_out] = carry as u64;
                }
            } else {
                l_digits[i_out] = (rot & lower_mask) | (carry as u64).rotate_right(bit_shr);
            }
            debug_assert_eq!(i_out + 1, l_digits.len());
            self.shl = shl;
            return self;
        }

        let mut vec = Vec::with_capacity(len);
        vec.push(d);
        let (long, short) = if l_digits.len() >= r_digits.len() {
            (l_digits, r_digits)
        } else {
            (r_digits, l_digits)
        };
        if i_in < short.len() {
            let zipped = short[i_in..].iter().zip(&long[i_in..]);
            vec.extend(zipped.map(|(l, r)| {
                let res = l.carrying_add(*r, carry);
                let next_rot = res.0.rotate_right(bit_shr);
                carry = res.1;
                (mem::replace(&mut rot, next_rot) & lower_mask) | (next_rot & upper_mask)
            }));
            i_in = short.len();
        }
        if i_in < long.len() {
            vec.extend(long[i_in..].iter().map(|x| {
                let res = x.overflowing_add(carry as u64);
                let next_rot = res.0.rotate_right(bit_shr);
                carry = res.1;
                (mem::replace(&mut rot, next_rot) & lower_mask) | (next_rot & upper_mask)
            }));
        }

        if bit_shr == 0 {
            debug_assert_eq!(lower_mask, u64::MAX);
            vec.push(rot);
            if carry {
                vec.push(1);
            }
        } else {
            let d = (rot & lower_mask) | (carry as u64).rotate_right(bit_shr);
            if d != 0 {
                vec.push(d);
            }
        }
        if vec.len() < vec.capacity() {
            vec.push(0);
        }
        debug_assert_eq!(vec.len(), vec.capacity());
        Self::from_mantissa_with_shl(vec.into_boxed_slice(), shl)
    }
}

impl Shl<u64> for Natural {
    type Output = Self;
    #[inline]
    fn shl(mut self, rhs: u64) -> Self {
        if self.len != 0 {
            self.shl = self.shl.saturating_add(rhs);
        }
        self
    }
}
impl Shl<u32> for Natural {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: u32) -> Self {
        self.shl(rhs as u64)
    }
}

impl Shr<u64> for Natural {
    type Output = Self;
    #[inline]
    fn shr(mut self, rhs: u64) -> Self {
        if self.shl >= rhs {
            if self.shl != u64::MAX {
                self.shl -= rhs;
            }
        } else {
            self.shl = u64::MAX;
        }
        self
    }
}
impl Shr<u32> for Natural {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: u32) -> Self {
        self.shr(rhs as u64)
    }
}

fn to_u_big(value: &[u64]) -> dashu_int::UBig {
    // `UBig::from_words` doesn't optimize these cases:
    match *value {
        [d] => return d.into(),
        [d1, d2] => return (d1 as u128 | ((d2 as u128) << 64)).into(),
        _ => {}
    }

    const IS_MULTIPLE: bool =
        mem::size_of::<u64>().is_multiple_of(mem::size_of::<dashu_int::Word>());
    const SCALE: usize = mem::size_of::<u64>() / mem::size_of::<dashu_int::Word>();
    const ALIGN_MATCH: bool = mem::align_of::<dashu_int::Word>() <= mem::align_of::<u64>();
    // spell-checker:ignore CONV
    #[cfg(target_endian = "little")]
    const FAST_CONV: bool = ALIGN_MATCH && IS_MULTIPLE;
    #[cfg(target_endian = "big")]
    const FAST_CONV: bool = ALIGN_MATCH && IS_MULTIPLE && SCALE == 1;

    if FAST_CONV {
        // SAFETY: `dashu_int::Word` is a primitive integer type with equal or
        // smaller alignment and fits precisely `SCALE` times into a `u64`.
        dashu_int::UBig::from_words(unsafe {
            std::slice::from_raw_parts(
                value.as_ptr().cast::<dashu_int::Word>(),
                value.len() * SCALE,
            )
        })
    } else {
        #[cfg(target_endian = "big")]
        let value: Box<[u64]> = value.iter().map(|d| d.swap_bytes()).collect();
        #[cfg(target_endian = "big")]
        let value = value.as_slice();

        // SAFETY: reinterpreting a `&[u64]` as `&[u8]` is safe
        dashu_int::UBig::from_le_bytes(unsafe {
            std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), mem::size_of_val(value))
        })
    }
}

impl TryFrom<&Natural> for dashu_int::UBig {
    type Error = super::NotRepresentable;

    fn try_from(value: &Natural) -> Result<Self, Self::Error> {
        if value.shl > std::cmp::min(1 << 40, usize::MAX as u64) {
            return Err(super::NotRepresentable); // covers NaN
        }
        let mut mantissa = to_u_big(value.mantissa());
        mantissa <<= value.shl as usize;
        Ok(mantissa)
    }
}

impl TryFrom<&Natural> for u128 {
    type Error = super::NotRepresentable;

    fn try_from(value: &Natural) -> Result<Self, Self::Error> {
        if value.shl < u128::BITS as u64
            && let mantissa = value.mantissa_raw()
            && mantissa.len() <= 3
            && mantissa.len() as u32 * u64::BITS - mantissa.last().unwrap().leading_zeros()
                + value.shl as u32
                <= u128::BITS
        {
            let upper = mantissa.get(1).copied().unwrap_or_default();
            return Ok(mantissa[0] as u128 | ((upper as u128) << u64::BITS));
        }
        Err(super::NotRepresentable)
    }
}
impl TryFrom<&Natural> for u64 {
    type Error = super::NotRepresentable;

    fn try_from(value: &Natural) -> Result<Self, Self::Error> {
        if value.shl < u64::BITS as u64
            && value.ptr == DANGLING
            && value.shl as u32 <= value.len.leading_zeros()
        {
            return Ok(value.len << value.shl);
        }
        Err(super::NotRepresentable)
    }
}
impl From<&Natural> for f64 {
    fn from(value: &Natural) -> Self {
        const ZERO_OFFSET: u32 = 1023;
        const MANTISSA_BITS: u32 = f64::MANTISSA_DIGITS - 1; // don't count the 1 from the integral part
        const EXP_BITS: u32 = u64::BITS - MANTISSA_BITS - 1;

        if value.is_nan() {
            return f64::NAN;
        }
        let mantissa = value.mantissa();
        let msd = *mantissa.last().unwrap();
        if msd == 0 {
            return 0.;
        }
        let leading_zeros = msd.leading_zeros();
        let bit_width = (mantissa.len() as u64)
            .saturating_mul(u64::BITS as u64)
            .saturating_add(value.shl)
            - leading_zeros as u64;
        // 1 has bit width 1 but is 1×2⁰ → exponent is one less
        if bit_width > f64::MAX_EXP as u64 {
            return f64::INFINITY;
        }
        let exp = bit_width as u32 + ZERO_OFFSET - 1;
        debug_assert!(exp < (1 << EXP_BITS) - 1);

        let msd2 = if let [.., msd2, _] = *mantissa {
            msd2
        } else {
            0
        };
        // fractional part with the most significant bit truncated away; needs
        // to be shifted to the right
        let frac_trunc_msb = if msd != 1 {
            msd << (leading_zeros + 1) | msd2 >> (u64::BITS - (leading_zeros + 1))
        } else {
            msd2
        };
        let shr = u64::BITS - MANTISSA_BITS;
        let frac_trunc = frac_trunc_msb >> shr;
        let frac_rounded = frac_trunc
            + if bit_width - value.shl == f64::MANTISSA_DIGITS as u64 + 1 {
                frac_trunc & 1 // tie -> round to even
            } else {
                (frac_trunc_msb >> (shr - 1)) & 1
            };
        debug_assert!(frac_rounded <= 1 << MANTISSA_BITS);
        // `frac_rounded` may now be `1 << MANTISSA_BITS`, but this is
        // fine: we will just get the next exponent or +inf.
        f64::from_bits(((exp as u64) << MANTISSA_BITS) + frac_rounded)
    }
}

fn fmt_nan(f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if let Some(width) = f.width()
        && width >= 2
    {
        let w = width - 1;
        let c = f.fill();
        match f.align() {
            Some(fmt::Alignment::Left) => {
                f.write_char('?')?;
                for _ in 0..w {
                    f.write_char(c)?;
                }
            }
            Some(fmt::Alignment::Center) => {
                let first_half = w / 2;
                for _ in 0..first_half {
                    f.write_char(c)?;
                }
                f.write_char('?')?;
                for _ in 0..(w - first_half) {
                    f.write_char(c)?;
                }
            }
            _ => {
                for _ in 0..w {
                    f.write_char(c)?;
                }
                f.write_char('?')?;
            }
        }
        Ok(())
    } else {
        f.write_char('?')
    }
}

impl fmt::Display for Natural {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match dashu_int::UBig::try_from(self) {
            Ok(num) => num.fmt(f),
            Err(_) => fmt_nan(f),
        }
    }
}

/// Re-implementation of [`fmt::Formatter::pad_integral()`] that does not force
/// us to write all digits to a buffer first
#[inline]
fn pad_integral(
    f: &mut fmt::Formatter<'_>,
    digits: u128,
    prefix: &str,
    write_digits: impl FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    let prefix_width = f.alternate() as usize * prefix.len() + f.sign_plus() as usize;
    let min_digits = f.width().unwrap_or(0).saturating_sub(prefix_width);
    let mut pad = match usize::try_from(digits) {
        Ok(digits) => min_digits.saturating_sub(digits),
        Err(_) => 0,
    };

    if pad != 0 && f.sign_aware_zero_pad() {
        for _ in 0..pad {
            f.write_char('0')?;
        }
        pad = 0;
    }

    if f.sign_plus() {
        f.write_char('+')?;
    }
    if f.alternate() {
        f.write_str(prefix)?;
    }

    let fill_char = f.fill();
    if pad != 0 {
        let pad_front = match f.align() {
            Some(fmt::Alignment::Left) => 0,
            Some(fmt::Alignment::Center) => pad / 2,
            _ => pad,
        };
        pad -= pad_front;
        for _ in 0..pad_front {
            f.write_char(fill_char)?;
        }
    }

    write_digits(f)?;

    for _ in 0..pad {
        f.write_char(fill_char)?;
    }

    Ok(())
}

impl fmt::Binary for Natural {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            return fmt_nan(f);
        }

        let mantissa = self.mantissa();
        let bit_width = bit_width(mantissa, self.shl);

        pad_integral(f, bit_width, "0b", move |f| {
            let msd = *mantissa.last().unwrap();
            if msd == 0 {
                return f.write_char('0');
            }

            let mut pos = u64::BITS - msd.leading_zeros();
            for &d in mantissa.iter().rev() {
                while pos != 0 {
                    pos -= 1;
                    f.write_char(if d & (1 << pos) != 0 { '1' } else { '0' })?;
                }
                pos = u64::BITS;
            }

            for _ in 0..self.shl {
                f.write_char('0')?;
            }
            Ok(())
        })
    }
}

impl Natural {
    #[inline] // for constant propagation
    fn fmt_pow2(
        &self,
        f: &mut fmt::Formatter<'_>,
        bits_per_digit: u32,
        prefix: &str,
        to_char: impl Fn(u8) -> char,
    ) -> fmt::Result {
        debug_assert!(bits_per_digit < u64::BITS);
        if self.is_nan() {
            return fmt_nan(f);
        }

        let mut mantissa = self.mantissa();
        let bit_width = bit_width(mantissa, self.shl);
        let digits = bit_width.div_ceil(bits_per_digit as u128);
        let rem_bits = (bit_width % bits_per_digit as u128) as u32;

        pad_integral(f, digits, prefix, move |f| {
            let mut msd = *mantissa.split_off_last().unwrap();
            if msd == 0 {
                return f.write_char('0');
            }
            // 0001010101 0000
            //  ^  |  |   |  |

            let rem_bits = if rem_bits == 0 {
                bits_per_digit
            } else {
                rem_bits
            };
            let mut offset = (u64::BITS - msd.leading_zeros()) as i32 - rem_bits as i32;
            let mut done = false;
            while !done {
                let digit = if offset >= 0 {
                    msd >> offset
                } else {
                    let upper = msd << offset.abs();
                    offset += u64::BITS as i32;

                    if let Some(v) = mantissa.split_off_last() {
                        msd = *v;
                        upper | (msd >> offset)
                    } else if offset == (u64::BITS - bits_per_digit) as i32 {
                        break;
                    } else {
                        done = true;
                        upper
                    }
                } & ((1 << bits_per_digit) - 1);
                f.write_char(to_char(digit as u8))?;
                offset -= bits_per_digit as i32;
            }

            let trailing_zeros = self.shl / bits_per_digit as u64;
            for _ in 0..trailing_zeros {
                f.write_char('0')?;
            }

            Ok(())
        })
    }
}

impl fmt::Octal for Natural {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_pow2(f, 3, "0o", |d| (b'0' + d) as char)
    }
}
impl fmt::LowerHex for Natural {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_pow2(f, 4, "0x", |d| {
            (if d < 10 { b'0' + d } else { b'a' + (d - 10) }) as char
        })
    }
}
impl fmt::UpperHex for Natural {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_pow2(f, 4, "0x", |d| {
            (if d < 10 { b'0' + d } else { b'A' + (d - 10) }) as char
        })
    }
}

impl fmt::Debug for Natural {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            return fmt_nan(f);
        }

        let mantissa = self.mantissa();
        if let [d] = mantissa {
            d.fmt(f)?;
        } else {
            to_u_big(mantissa).fmt(f)?;
        }
        f.write_str(" * 2^")?;
        self.shl.fmt(f)
    }
}

#[cfg(test)]
mod test {
    use super::{DANGLING, Natural};
    use std::fmt::Write as _;

    fn check_inv(i: &Natural) {
        if i.ptr == DANGLING {
            if i.len == 0 {
                assert_eq!(i.shl, 0);
            } else {
                assert_eq!(i.len.trailing_zeros(), 0);
            }
        } else {
            assert_ne!(i.len, 0);
            assert_ne!(i.len, 1);
            // SAFETY: `i.ptr` is not `DANGLING`
            let digits = unsafe { std::slice::from_raw_parts(i.ptr.as_ptr(), i.len as usize) };
            if *digits.last().unwrap() == 0 {
                assert_eq!(digits[digits.len() - 2] >> (u64::BITS - 1), 1);
            }
            assert_eq!(digits[0] & 1, 1);
        }
    }

    #[test]
    fn test_from_clone_and_shift_small() {
        let zero = Natural::from(0u32);
        check_inv(&zero);
        assert_eq!(zero, Natural::ZERO);

        let clone = zero.clone();
        check_inv(&clone);
        assert_eq!(clone, Natural::ZERO);

        let mut clone = clone << 1u64;
        check_inv(&clone);
        assert_eq!(clone, Natural::ZERO);

        let one = Natural::from(1u64);
        check_inv(&one);
        assert_ne!(one, Natural::ZERO);
        assert!(one > Natural::ZERO);

        let two = Natural::from(2u64);
        check_inv(&two);
        assert_ne!(two, one);
        assert!(two > one);

        clone.clone_from(&one);
        check_inv(&clone);
        assert_eq!(clone, one);

        let clone = clone << 1u32;
        check_inv(&clone);
        assert_eq!(clone, two);
    }

    #[test]
    fn test_from_le_digits() {
        for slice in [[].as_slice(), &[0], &[0, 0]] {
            let num = Natural::from_le_digits(slice);
            check_inv(&num);
            assert_eq!(num, Natural::ZERO);
        }

        let one = Natural::from(1u32);
        for slice in [[1].as_slice(), &[1, 0]] {
            let num = Natural::from_le_digits(slice);
            check_inv(&num);
            assert_eq!(num, one);
        }

        let two = Natural::from(2u32);
        let num = Natural::from_le_digits(&[2]);
        check_inv(&num);
        assert_eq!(num, two);

        let a = one.clone() << u64::BITS;
        let b = Natural::from_le_digits(&[0, 1]);
        check_inv(&b);
        assert_eq!(a, b);

        let a = Natural::from(0b11u32) << (2 * u64::BITS - 1);
        let b = Natural::from_le_digits(&[0, 1 << (u64::BITS - 1), 1]);
        check_inv(&b);
        assert_eq!(a, b);

        let half_bits = u64::BITS / 2;
        let a = Natural::from_le_digits(&[!0 << half_bits, !0, !0 >> half_bits]);
        check_inv(&a);
        let b = Natural::from_le_digits(&[!0, !0]);
        check_inv(&b);
        let b = b << half_bits;
        check_inv(&b);
        assert_eq!(a, b);

        let half1_bits = u64::BITS / 2 - 1;
        let a = Natural::from_le_digits(&[!0 << half1_bits, !0, !0 >> half1_bits]);
        check_inv(&a);
        let b = Natural::from_le_digits(&[!0, !0, 0b11]);
        check_inv(&b);
        let b = b << half1_bits;
        check_inv(&b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_add() {
        for shl in [0, 1, u64::BITS - 1] {
            let case = |lhs: Natural, rhs: Natural, sum: Natural| {
                let lhs = lhs << shl;
                let rhs = rhs << shl;
                let expected = sum << shl;
                let actual = lhs.clone() + rhs.clone();
                check_inv(&actual);
                assert_eq!(actual, expected);
                let actual_rev = rhs + lhs;
                check_inv(&actual_rev);
                assert_eq!(actual_rev, expected);
            };

            // --- zero operand ---
            case(1u32.into(), Natural::ZERO, 1u32.into());
            case(u64::MAX.into(), Natural::ZERO, u64::MAX.into());
            case(
                Natural::from_le_digits(&[1, 1]),
                Natural::ZERO,
                Natural::from_le_digits(&[1, 1]),
            );

            // --- same shl for both operands ---
            case(1u32.into(), 1u32.into(), 2u32.into());

            let a = Natural::from_le_digits(&[!0, 1]);
            case(a.clone(), a.clone(), a << 1u32);

            case(
                u64::MAX.into(),
                1u32.into(),
                Natural::from(1u32) << u64::BITS,
            );

            case(
                u64::MAX.into(),
                u64::MAX.into(),
                Natural::from_le_digits(&[u64::MAX << 1, 1]),
            );

            case(
                u64::MAX.into(),
                Natural::from_le_digits(&[1, !0, 1]),
                Natural::from(1u32) << (2 * u64::BITS + 1),
            );

            case(
                Natural::from_le_digits(&[1, 1 << (u64::BITS - 1)]),
                Natural::from_le_digits(&[u64::MAX, 1 << (u64::BITS - 1)]),
                Natural::from_le_digits(&[1, 1]) << u64::BITS,
            );

            case(
                Natural::from_le_digits(&[1, 1]),
                Natural::from_le_digits(&[!0, !0, 1]),
                Natural::from_le_digits(&[1, 0b10]) << u64::BITS,
            );

            case(
                Natural::from_le_digits(&[1, 1]),
                Natural::from_le_digits(&[!0, !0, !0]),
                Natural::from_le_digits(&[1, 0, 1]) << u64::BITS,
            );

            case(
                Natural::from_le_digits(&[1, 1 << (u64::BITS - 1)]),
                Natural::from_le_digits(&[!0, !0, !0, 1]),
                Natural::from_le_digits(&[1, 0b100]) << (2 * u64::BITS - 1),
            );

            case(
                Natural::from_le_digits(&[1, 1 << (u64::BITS - 1)]),
                Natural::from_le_digits(&[!0, !0, !0, u64::MAX >> 1]),
                Natural::from_le_digits(&[1, 0, 1]) << (2 * u64::BITS - 1),
            );

            case(
                Natural::from_le_digits(&[1, 0, 1]),
                1u32.into(),
                Natural::from_le_digits(&[2, 0, 1]),
            );

            // --- different shl ---
            case(1u32.into(), 2u32.into(), 3u32.into());

            let max_bit: u64 = 1 << (u64::BITS - 1);
            case(
                (1 | max_bit).into(),
                (1 ^ u64::MAX).into(),
                Natural::from_le_digits(&[u64::MAX ^ max_bit, 1]),
            );

            case(
                Natural::from_le_digits(&[0, 1]),
                1u32.into(),
                Natural::from_le_digits(&[1, 1]),
            );

            case(
                Natural::from_le_digits(&[0, 2]),
                1u32.into(),
                Natural::from_le_digits(&[1, 2]),
            );

            case(
                Natural::from_le_digits(&[1, 1]),
                Natural::from_le_digits(&[0, 2]),
                Natural::from_le_digits(&[1, 3]),
            );

            case(
                Natural::from_le_digits(&[1, 1]),
                Natural::from_le_digits(&[0, 2, 1]),
                Natural::from_le_digits(&[1, 3, 1]),
            );

            case(
                Natural::from_le_digits(&[1, 1]),
                Natural::from_le_digits(&[0, 2, 2]),
                Natural::from_le_digits(&[1, 3, 2]),
            );

            case(
                Natural::from_le_digits(&[1, u64::MAX]),
                Natural::from_le_digits(&[0, 2, 1]),
                Natural::from_le_digits(&[1, 1, 2]),
            );

            case(
                Natural::from_le_digits(&[!0]),
                Natural::from_le_digits(&[2, !0]),
                Natural::from_le_digits(&[1, 0, 1]),
            );
        }
    }

    #[test]
    fn test_to_f64() {
        for val in 0..0x10000u64 {
            assert_eq!(f64::from(&Natural::from(val)), val as f64);
        }
        for shl in 52..60 {
            for val in ((1u64 << shl) - 0xf)..=((1 << shl) + 0xf) {
                assert_eq!(f64::from(&Natural::from(val)), val as f64, "{val}");
            }
        }
        assert_eq!(
            f64::from(&Natural::from_mantissa_single_with_shl(1, 1023)),
            2f64.powi(1023),
        );
        assert_eq!(
            f64::from(&Natural::from_mantissa_single_with_shl(1, 1024)),
            f64::INFINITY,
        );
    }

    #[test]
    fn fmt_bin() -> std::fmt::Result {
        let mut buf = String::with_capacity(128);

        write!(buf, "{:b}", Natural::from(0u32))?;
        assert_eq!(&buf, "0");
        buf.clear();

        write!(buf, "{:#b}", Natural::from(0u32))?;
        assert_eq!(&buf, "0b0");
        buf.clear();

        write!(buf, "{:b}", Natural::from(1u32))?;
        assert_eq!(&buf, "1");
        buf.clear();

        write!(buf, "{:b}", Natural::from(2u32))?;
        assert_eq!(&buf, "10");
        buf.clear();

        write!(buf, "{:b}", Natural::from(3u32))?;
        assert_eq!(&buf, "11");
        buf.clear();

        write!(buf, "{:b}", Natural::from(4u32))?;
        assert_eq!(&buf, "100");
        buf.clear();

        write!(buf, "{:b}", Natural::from_le_digits(&[1, 1]))?;
        assert_eq!(buf, format!("{:b}", (1u128 << 64) | 1));
        buf.clear();

        Ok(())
    }

    #[test]
    fn fmt_oct() -> std::fmt::Result {
        let mut buf = String::with_capacity(16);

        write!(buf, "{:o}", Natural::from(0u32))?;
        assert_eq!(&buf, "0");
        buf.clear();

        write!(buf, "{:#o}", Natural::from(0u32))?;
        assert_eq!(&buf, "0o0");
        buf.clear();

        write!(buf, "{:o}", Natural::from(1u32))?;
        assert_eq!(&buf, "1");
        buf.clear();

        write!(buf, "{:o}", Natural::from(9u32))?;
        assert_eq!(&buf, "11");
        buf.clear();

        write!(buf, "{:o}", Natural::from(0o71u32))?;
        assert_eq!(&buf, "71");
        buf.clear();

        write!(buf, "{:o}", Natural::from(0o72u32))?;
        assert_eq!(&buf, "72");
        buf.clear();

        write!(buf, "{:o}", Natural::from(0o74u32))?;
        assert_eq!(&buf, "74");
        buf.clear();

        Ok(())
    }

    #[test]
    fn fmt_hex() -> std::fmt::Result {
        let mut buf = String::with_capacity(16);

        write!(buf, "{:x}", Natural::from(0u32))?;
        assert_eq!(&buf, "0");
        buf.clear();

        write!(buf, "{:#x}", Natural::from(0u32))?;
        assert_eq!(&buf, "0x0");
        buf.clear();

        write!(buf, "{:x}", Natural::from(1u32))?;
        assert_eq!(&buf, "1");
        buf.clear();

        write!(buf, "{:x}", Natural::from(0xau32))?;
        assert_eq!(&buf, "a");
        buf.clear();

        write!(buf, "{:X}", Natural::from(0xau32))?;
        assert_eq!(&buf, "A");
        buf.clear();

        write!(buf, "{:x}", Natural::from(0xf1u32))?;
        assert_eq!(&buf, "f1");
        buf.clear();

        write!(buf, "{:x}", Natural::from(0xf2u32))?;
        assert_eq!(&buf, "f2");
        buf.clear();

        write!(buf, "{:x}", Natural::from(0xf4u32))?;
        assert_eq!(&buf, "f4");
        buf.clear();

        write!(buf, "{:x}", Natural::from(0xf8u32))?;
        assert_eq!(&buf, "f8");
        buf.clear();

        Ok(())
    }
}
