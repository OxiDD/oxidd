//! Number types useful for counting satisfying assignments

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

macro_rules! impl_assign_via_copy {
    ($t:ty) => {
        impl<'a> ::std::ops::AddAssign<&'a Self> for $t {
            #[inline]
            fn add_assign(&mut self, rhs: &'a Self) {
                *self = *self + *rhs;
            }
        }
        impl<'a> ::std::ops::SubAssign<&'a Self> for $t {
            #[inline]
            fn sub_assign(&mut self, rhs: &'a Self) {
                *self = *self - *rhs;
            }
        }
        impl<'a> ::std::ops::ShlAssign<u32> for $t {
            #[inline]
            fn shl_assign(&mut self, rhs: u32) {
                *self = *self << rhs;
            }
        }
        impl<'a> ::std::ops::ShrAssign<u32> for $t {
            #[inline]
            fn shr_assign(&mut self, rhs: u32) {
                *self = *self >> rhs;
            }
        }
    };
}

macro_rules! impl_saturating {
    ($t:ty) => {
        impl ::std::ops::Add for Saturating<$t> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self(self.0.saturating_add(rhs.0))
            }
        }

        impl ::std::ops::Sub for Saturating<$t> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(if self.0 == <$t>::MAX {
                    <$t>::MAX
                } else {
                    self.0 - rhs.0
                })
            }
        }

        impl ::std::ops::Shl<u32> for Saturating<$t> {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: u32) -> Self {
                Self(self.0.checked_shl(rhs).unwrap_or(<$t>::MAX))
            }
        }

        impl ::std::ops::Shr<u32> for Saturating<$t> {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: u32) -> Self {
                Self(if self.0 == <$t>::MAX {
                    <$t>::MAX
                } else {
                    self.0 >> rhs
                })
            }
        }

        impl_assign_via_copy!(Saturating<$t>);
    };
}

impl_saturating!(u64);
impl_saturating!(u128);

/// Floating point number like [`f64`], but with [`ShlAssign<u32>`] and
/// [`ShrAssign<u32>`].
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct F64(pub f64);

impl From<u32> for F64 {
    #[inline]
    fn from(value: u32) -> Self {
        Self(value as f64)
    }
}

/// cbindgen:ignore
impl IsFloatingPoint for F64 {
    const FLOATING_POINT: bool = true;
    const MIN_EXP: i32 = f64::MIN_EXP;
}

impl std::ops::Add for F64 {
    type Output = F64;
    #[inline]
    fn add(self, rhs: F64) -> F64 {
        F64(self.0 + rhs.0)
    }
}
impl std::ops::Sub for F64 {
    type Output = F64;
    #[inline]
    fn sub(self, rhs: F64) -> F64 {
        F64(self.0 - rhs.0)
    }
}

impl std::ops::Shl<u32> for F64 {
    type Output = F64;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn shl(self, rhs: u32) -> F64 {
        F64(self.0 * (rhs as f64).exp2())
    }
}
impl std::ops::Shr<u32> for F64 {
    type Output = F64;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn shr(self, rhs: u32) -> F64 {
        F64(self.0 * (-(rhs as f64)).exp2())
    }
}

impl_assign_via_copy!(F64);
