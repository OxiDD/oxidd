use std::fmt;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use oxidd_core::function::NumberBase;

/// [`f64`], but strongly normalizing and thus implementing [`Eq`] and [`Hash`].
/// In particular, `Float64::nan() == Float64::nan()` holds. Internally, all NaN
/// values are normalized to [`f64::NAN`] and `-0.0` is normalized to `0.0`.
#[derive(Clone, Copy)]
pub struct F64(f64);

impl PartialEq for F64 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for F64 {}
impl PartialOrd for F64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.0.to_bits() == f64::NAN.to_bits() {
            if other.0.to_bits() == f64::NAN.to_bits() {
                Some(std::cmp::Ordering::Equal)
            } else {
                None
            }
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

impl Hash for F64 {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for F64 {
    #[inline]
    fn from(value: f64) -> Self {
        Self(if value.is_nan() {
            f64::NAN
        } else if value.to_bits() == (-0.0f64).to_bits() {
            0.0
        } else {
            value
        })
    }
}
impl From<F64> for f64 {
    #[inline(always)]
    fn from(value: F64) -> Self {
        value.0
    }
}

impl NumberBase for F64 {
    #[inline]
    fn zero() -> Self {
        Self(0.)
    }
    #[inline]
    fn one() -> Self {
        Self(1.)
    }
    #[inline]
    fn nan() -> Self {
        Self(f64::NAN)
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        Self::from(self.0 + rhs.0)
    }
    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        Self::from(self.0 - rhs.0)
    }
    #[inline]
    fn mul(&self, rhs: &Self) -> Self {
        Self::from(self.0 * rhs.0)
    }
    #[inline]
    fn div(&self, rhs: &Self) -> Self {
        Self::from(self.0 / rhs.0)
    }
}

impl<Tag: Default> oxidd_dump::ParseTagged<Tag> for F64 {
    fn parse(s: &str) -> Option<(Self, Tag)> {
        let val = match s {
            "nan" | "NaN" | "NAN" => Self(f64::NAN),
            "-∞" | "-inf" | "-infinity" | "-Inf" | "-Infinity" | "-INF" | "-INFINITY"
            | "MinusInf" => Self(f64::NEG_INFINITY),
            "∞" | "inf" | "infinity" | "Inf" | "Infinity" | "INF" | "INFINITY" | "+∞" | "+inf"
            | "+infinity" | "+Inf" | "+Infinity" | "+INF" | "+INFINITY" | "PlusInf" => {
                Self(f64::INFINITY)
            }
            _ => Self(f64::from_str(s).ok()?),
        };
        Some((val, Tag::default()))
    }
}

impl Display for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.to_bits() == f64::NAN.to_bits() {
            f.write_str("NaN")
        } else if self.0.to_bits() == f64::NEG_INFINITY.to_bits() {
            f.write_str("-∞")
        } else if self.0.to_bits() == f64::INFINITY.to_bits() {
            f.write_str("+∞")
        } else {
            self.0.fmt(f)
        }
    }
}
impl fmt::Debug for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl oxidd_dump::AsciiDisplay for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.to_bits() == f64::NAN.to_bits() {
            f.write_str("NaN")
        } else if self.0.to_bits() == f64::NEG_INFINITY.to_bits() {
            f.write_str("-INF")
        } else if self.0.to_bits() == f64::INFINITY.to_bits() {
            f.write_str("+INF")
        } else {
            self.0.fmt(f)
        }
    }
}

impl Add for F64 {
    type Output = F64;

    #[inline]
    fn add(self, rhs: Self) -> F64 {
        Self::from(self.0 + rhs.0)
    }
}

impl Sub for F64 {
    type Output = F64;

    #[inline]
    fn sub(self, rhs: Self) -> F64 {
        Self::from(self.0 - rhs.0)
    }
}

impl Mul for F64 {
    type Output = F64;

    #[inline]
    fn mul(self, rhs: Self) -> F64 {
        Self::from(self.0 * rhs.0)
    }
}

impl Div for F64 {
    type Output = F64;

    #[inline]
    fn div(self, rhs: Self) -> F64 {
        Self::from(self.0 / rhs.0)
    }
}

super::impl_ref_op!(F64, Add, add);
super::impl_ref_op!(F64, Sub, sub);
super::impl_ref_op!(F64, Mul, mul);
super::impl_ref_op!(F64, Div, div);

#[cfg(test)]
mod test {
    use oxidd_core::function::NumberBase;

    use super::F64;

    #[test]
    fn nan_eq_ord() {
        let nan = F64::nan();
        assert!(nan <= nan);
        assert!(nan == nan);
        assert!(nan >= nan);
    }

    #[test]
    fn zero_ord() {
        assert!(F64::from(-0.0) <= F64::from(0.0));
        assert!(F64::from(-0.0) == F64::from(0.0));
        assert!(F64::from(-0.0) >= F64::from(0.0));
    }
}
