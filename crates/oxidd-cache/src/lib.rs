//! Apply cache implementations
//!
//! # Feature flags
#![doc = document_features::document_features!()]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
// We use const assertions for checking configurations and reporting errors in
// case of obscure targets. To achieve this, we use assertions that evaluate to
// `true` on usual targets as well as unit let bindings.
#![allow(clippy::let_unit_value)]

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
