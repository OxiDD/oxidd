//! Number types useful for counting satisfying assignments

use std::ops::AddAssign;
use std::ops::ShlAssign;
use std::ops::ShrAssign;
use std::ops::SubAssign;

use crate::util::IsFloatingPoint;

/// Natural numbers with saturating arithmetic
///
/// In contrast to [`std::num::Saturating`], `T::MAX` represents an
/// out-of-bounds value, and a subsequent subtraction or right shift does not
/// change this value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Saturating<T>(pub T);

/// cbindgen:ignore
impl<T> IsFloatingPoint for Saturating<T> {
    const FLOATING_POINT: bool = false;
    const MIN_EXP: i32 = 0;
}

impl<T: From<u32>> From<u32> for Saturating<T> {
    #[inline]
    fn from(value: u32) -> Self {
        Self(T::from(value))
    }
}

impl<'a> AddAssign<&'a Self> for Saturating<u64> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}
impl<'a> AddAssign<&'a Self> for Saturating<u128> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl<'a> SubAssign<&'a Self> for Saturating<u64> {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        if self.0 != u64::MAX {
            self.0 -= rhs.0;
        }
    }
}
impl<'a> SubAssign<&'a Self> for Saturating<u128> {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        if self.0 != u128::MAX {
            self.0 -= rhs.0;
        }
    }
}

impl ShrAssign<u32> for Saturating<u64> {
    #[inline]
    fn shr_assign(&mut self, rhs: u32) {
        if self.0 != u64::MAX {
            self.0 >>= rhs;
        }
    }
}
impl ShrAssign<u32> for Saturating<u128> {
    #[inline]
    fn shr_assign(&mut self, rhs: u32) {
        if self.0 != u128::MAX {
            self.0 >>= rhs;
        }
    }
}

impl ShlAssign<u32> for Saturating<u64> {
    #[inline]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 = match self.0.checked_shl(rhs) {
            Some(v) => v,
            None => u64::MAX,
        }
    }
}
impl ShlAssign<u32> for Saturating<u128> {
    #[inline]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 = match self.0.checked_shl(rhs) {
            Some(v) => v,
            None => u128::MAX,
        }
    }
}

/// Floating point number like [`f64`], but with [`ShlAssign<u32>`] and
/// [`ShrAssign<u32>`].
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct F64(pub f64);

impl From<u32> for F64 {
    fn from(value: u32) -> Self {
        Self(value as f64)
    }
}

/// cbindgen:ignore
impl IsFloatingPoint for F64 {
    const FLOATING_POINT: bool = true;
    const MIN_EXP: i32 = f64::MIN_EXP;
}

impl<'a> AddAssign<&'a Self> for F64 {
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0 += rhs.0;
    }
}
impl<'a> SubAssign<&'a Self> for F64 {
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.0 -= rhs.0;
    }
}

impl ShlAssign<u32> for F64 {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 *= (rhs as f64).exp2()
    }
}
impl ShrAssign<u32> for F64 {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 *= (-(rhs as f64)).exp2()
    }
}
