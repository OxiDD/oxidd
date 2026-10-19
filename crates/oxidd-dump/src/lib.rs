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

use oxidd_core::function::{Function, INodeValOfFunc, OwnEdgeOfFunc};
use oxidd_core::util::AllocResult;
use oxidd_core::{LevelNo, Manager};

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

/// [`Function`] that can be deserialized from a dump
pub trait DumpDeserialize: Function {
    /// Try to create a node at `level` with `children` and `value`, applying
    /// reduction rules
    ///
    /// The default implementation will just ensure node uniqueness.
    fn reduce<'id>(
        manager: &Self::Manager<'id>,
        level: LevelNo,
        children: impl IntoIterator<Item = OwnEdgeOfFunc<'id, Self>>,
        value: INodeValOfFunc<'id, Self>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>> {
        manager.get_or_insert(level, children, value)
    }

    /// Negate `edge`
    ///
    /// This method provides the semantics for negative node references in the
    /// DDDMP dump.
    fn neg<'id>(
        manager: &Self::Manager<'id>,
        edge: OwnEdgeOfFunc<'id, Self>,
    ) -> AllocResult<OwnEdgeOfFunc<'id, Self>>;
}

#[cfg(feature = "dddmp")]
pub mod dddmp;

// No feature gate here to always have the traits
pub mod dot;

#[cfg(feature = "visualize")]
mod visualize;
#[cfg(feature = "visualize")]
pub use visualize::{VisualizationListener, Visualizer};
