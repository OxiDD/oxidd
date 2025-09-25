mod f64;
mod i64;
pub use f64::F64;
pub use i64::I64;

#[deprecated(since = "0.11.0", note = "Renamed to I64")]
pub type Int64 = I64;

/// Implement e.g. `Add<&T> for T`, `Add<T> for &T`, and `Add<&T> for &T` given
/// that `T: Add<T, Output = T>`
macro_rules! impl_ref_op {
    ($t:ty, $trait:ident, $method:ident) => {
        impl<'a> $trait<&'a $t> for $t {
            type Output = $t;
            #[inline]
            fn $method(self, rhs: &'a $t) -> Self::Output {
                $trait::$method(self, *rhs)
            }
        }

        impl<'a> $trait<$t> for &'a $t {
            type Output = $t;
            #[inline]
            fn $method(self, rhs: $t) -> Self::Output {
                $trait::$method(*self, rhs)
            }
        }

        impl<'a> $trait<&'a $t> for &'a $t {
            type Output = $t;
            #[inline]
            fn $method(self, rhs: &'a $t) -> Self::Output {
                $trait::$method(*self, *rhs)
            }
        }
    };
}
pub(crate) use impl_ref_op;
