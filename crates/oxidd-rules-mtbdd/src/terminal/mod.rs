mod int64;
pub use int64::Int64;

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
