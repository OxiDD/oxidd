//! Im- and export of decision diagrams
//!
//! # Feature flags
#![doc = document_features::document_features!()]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::type_complexity)]
// We use const assertions for reporting errors in case of obscure targets. To
// achieve this, we use assertions that evaluate to `true` on usual targets.
#![allow(clippy::assertions_on_constants)]

use std::fmt;

/// Like [`std::fmt::Display`], but the format should use ASCII characters only
pub trait AsciiDisplay {
    /// Format the value with the given formatter
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error>;
}

/// Parse a value from a string, along with a tag
pub trait ParseTagged<Tag>: Sized {
    /// Parse the string `s`
    fn parse(s: &str) -> Option<(Self, Tag)>;
}

#[cfg(feature = "dddmp")]
pub mod dddmp;

// No feature gate here to always have the traits
pub mod dot;

#[cfg(feature = "visualize")]
mod visualize;
#[cfg(feature = "visualize")]
pub use visualize::{VisualizationListener, Visualizer};
