use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use oxidd_core::function::NumberBase;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum I64 {
    NaN,
    MinusInf,
    Num(i64),
    PlusInf,
}

impl From<i64> for I64 {
    fn from(value: i64) -> Self {
        Self::Num(value)
    }
}

impl NumberBase for I64 {
    #[inline]
    fn zero() -> Self {
        Self::Num(0)
    }
    #[inline]
    fn one() -> Self {
        Self::Num(1)
    }
    #[inline]
    fn nan() -> Self {
        Self::NaN
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }
    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        self - rhs
    }
    #[inline]
    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }
    #[inline]
    fn div(&self, rhs: &Self) -> Self {
        self / rhs
    }
}

impl PartialOrd for I64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use I64::*;
        match (self, other) {
            (Num(lhs), Num(rhs)) => Some(lhs.cmp(rhs)),
            (NaN, NaN) | (MinusInf, MinusInf) | (PlusInf, PlusInf) => Some(Ordering::Equal),
            (NaN, _) | (_, NaN) => None,
            (MinusInf, _) | (_, PlusInf) => Some(Ordering::Less),
            (_, MinusInf) | (PlusInf, _) => Some(Ordering::Greater),
        }
    }
}

impl<Tag: Default> oxidd_dump::ParseTagged<Tag> for I64 {
    fn parse(s: &str) -> Option<(Self, Tag)> {
        let val = match s {
            "nan" | "NaN" | "NAN" => Self::NaN,
            "-∞" | "-inf" | "-infinity" | "-Inf" | "-Infinity" | "-INF" | "-INFINITY"
            | "MinusInf" => Self::MinusInf,
            "∞" | "inf" | "infinity" | "Inf" | "Infinity" | "INF" | "INFINITY" | "+∞" | "+inf"
            | "+infinity" | "+Inf" | "+Infinity" | "+INF" | "+INFINITY" | "PlusInf" => {
                Self::PlusInf
            }
            _ => Self::Num(i64::from_str(s).ok()?),
        };
        Some((val, Tag::default()))
    }
}

impl Display for I64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            I64::NaN => f.write_str("NaN"),
            I64::MinusInf => f.write_str("-∞"),
            I64::Num(n) => n.fmt(f),
            I64::PlusInf => f.write_str("+∞"),
        }
    }
}
impl fmt::Debug for I64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl oxidd_dump::AsciiDisplay for I64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            I64::NaN => f.write_str("NaN"),
            I64::MinusInf => f.write_str("-Inf"),
            I64::Num(n) => n.fmt(f),
            I64::PlusInf => f.write_str("+Inf"),
        }
    }
}

impl Add for I64 {
    type Output = I64;

    #[inline]
    fn add(self, rhs: Self) -> I64 {
        use I64::*;
        match (self, rhs) {
            (Num(lhs), Num(rhs)) => match lhs.checked_add(rhs) {
                Some(n) => Num(n),
                None => {
                    if lhs > 0 && rhs > 0 || lhs < 0 && rhs < 0 {
                        PlusInf
                    } else {
                        MinusInf
                    }
                }
            },
            (NaN, _) | (_, NaN) | (MinusInf, PlusInf) | (PlusInf, MinusInf) => NaN,
            (MinusInf, _) | (_, MinusInf) => MinusInf,
            (PlusInf, _) | (_, PlusInf) => PlusInf,
        }
    }
}

impl Sub for I64 {
    type Output = I64;

    #[inline]
    fn sub(self, rhs: Self) -> I64 {
        use I64::*;
        match (self, rhs) {
            (Num(lhs), Num(rhs)) => match lhs.checked_sub(rhs) {
                Some(n) => Num(n),
                None => {
                    if lhs > 0 && rhs < 0 || lhs < 0 && rhs > 0 {
                        PlusInf
                    } else {
                        MinusInf
                    }
                }
            },
            (NaN, _) | (_, NaN) | (MinusInf, MinusInf) | (PlusInf, PlusInf) => NaN,
            (MinusInf, _) | (_, PlusInf) => MinusInf,
            (PlusInf, _) | (_, MinusInf) => PlusInf,
        }
    }
}

impl Mul for I64 {
    type Output = I64;

    #[inline]
    fn mul(self, rhs: Self) -> I64 {
        use I64::*;
        match (self, rhs) {
            (Num(lhs), Num(rhs)) => match lhs.checked_mul(rhs) {
                Some(n) => Num(n),
                None => {
                    if lhs > 0 && rhs > 0 || lhs < 0 && rhs < 0 {
                        PlusInf
                    } else {
                        MinusInf
                    }
                }
            },
            (NaN, _) | (_, NaN) | (MinusInf, PlusInf) | (PlusInf, MinusInf) => NaN,
            (MinusInf, _) | (_, MinusInf) => MinusInf,
            (PlusInf, _) | (_, PlusInf) => PlusInf,
        }
    }
}

impl Div for I64 {
    type Output = I64;

    #[inline]
    fn div(self, rhs: Self) -> I64 {
        use I64::*;
        match (self, rhs) {
            (Num(lhs), Num(rhs)) => {
                if rhs == 0 {
                    match lhs.cmp(&0) {
                        Ordering::Less => MinusInf,
                        Ordering::Equal => NaN,
                        Ordering::Greater => PlusInf,
                    }
                } else if lhs == i64::MIN && rhs == -1 {
                    PlusInf
                } else {
                    Num(lhs / rhs)
                }
            }
            (Num(_), MinusInf | PlusInf) => Num(0),
            (PlusInf, Num(_)) => PlusInf,
            (MinusInf, Num(_)) => MinusInf,
            _ => NaN,
        }
    }
}

super::impl_ref_op!(I64, Add, add);
super::impl_ref_op!(I64, Sub, sub);
super::impl_ref_op!(I64, Mul, mul);
super::impl_ref_op!(I64, Div, div);
