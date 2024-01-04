//! Im- and export of decision diagrams
//!
//! # Feature flags
#![doc = document_features::document_features!()]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "dddmp")]
pub mod dddmp;

#[cfg(feature = "dot")]
pub mod dot;
