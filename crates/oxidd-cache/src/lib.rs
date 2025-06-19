//! Apply cache implementations
//!
//! # Feature flags
#![doc = document_features::document_features!()]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "direct")]
pub mod direct;

/// Trait for generating and printing statistics
///
/// Implementors of this trait in this crate do not print anything unless the
/// `statistics` feature is enabled.
pub trait StatisticsGenerator {
    /// Print statistics to stdout
    fn print_stats(&self);
}

mod util;
