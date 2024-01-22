use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use oxidd_core::function::NumberBase;
use oxidd_dump::dddmp::AsciiDisplay;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Int64 {
    NaN,
    MinusInf,
    Num(i64),
    PlusInf,
}

impl From<i64> for Int64 {
    fn from(value: i64) -> Self {
        Self::Num(value)
    }
}

impl NumberBase for Int64 {
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

impl PartialOrd for Int64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Int64::*;
        match (self, other) {
            (Num(lhs), Num(rhs)) => Some(lhs.cmp(rhs)),
            (NaN, NaN) | (MinusInf, MinusInf) | (PlusInf, PlusInf) => Some(Ordering::Equal),
            (NaN, _) | (_, NaN) => None,
            (MinusInf, _) | (_, PlusInf) => Some(Ordering::Less),
            (_, MinusInf) | (PlusInf, _) => Some(Ordering::Greater),
        }
    }
}

impl FromStr for Int64 {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "nan" | "NaN" | "NAN" => Self::NaN,
            "-∞" | "-inf" | "-infinity" | "-Inf" | "-Infinity" | "-INF" | "-INFINITY"
            | "MinusInf" => Self::MinusInf,
            "∞" | "inf" | "infinity" | "Inf" | "Infinity" | "INF" | "INFINITY" | "+∞" | "+inf"
            | "+infinity" | "+Inf" | "+Infinity" | "+INF" | "+INFINITY" | "PlusInf" => {
                Self::PlusInf
            }
            _ => Self::Num(i64::from_str(s)?),
        })
    }
}

impl Display for Int64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int64::NaN => f.write_str("NaN"),
            Int64::MinusInf => f.write_str("-∞"),
            Int64::Num(n) => n.fmt(f),
            Int64::PlusInf => f.write_str("+∞"),
        }
    }
}

impl AsciiDisplay for Int64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int64::NaN => f.write_str("NaN"),
            Int64::MinusInf => f.write_str("-Inf"),
            Int64::Num(n) => n.fmt(f),
            Int64::PlusInf => f.write_str("+Inf"),
        }
    }
}

impl Add for Int64 {
    type Output = Int64;

    #[inline]
    fn add(self, rhs: Self) -> Int64 {
        use Int64::*;
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

impl Sub for Int64 {
    type Output = Int64;

    #[inline]
    fn sub(self, rhs: Self) -> Int64 {
        use Int64::*;
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

impl Mul for Int64 {
    type Output = Int64;

    #[inline]
    fn mul(self, rhs: Self) -> Int64 {
        use Int64::*;
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

impl Div for Int64 {
    type Output = Int64;

    #[inline]
    fn div(self, rhs: Self) -> Int64 {
        use Int64::*;
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

super::impl_ref_op!(Int64, Add, add);
super::impl_ref_op!(Int64, Sub, sub);
super::impl_ref_op!(Int64, Mul, mul);
super::impl_ref_op!(Int64, Div, div);
