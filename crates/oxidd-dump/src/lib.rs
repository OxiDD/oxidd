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

#[cfg(feature = "dddmp")]
pub mod dddmp;

#[cfg(feature = "dot")]
pub mod dot;
