//! Visual representation using [dot]
//!
//! [dot]: https://graphviz.org/docs/layouts/dot/

use std::fmt;

use oxidd_core::Tag;

/// Edge styles
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeStyle {
    #[allow(missing_docs)]
    Solid,
    #[allow(missing_docs)]
    Dashed,
    #[allow(missing_docs)]
    Dotted,
}

impl fmt::Display for EdgeStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EdgeStyle::Solid => "solid",
            EdgeStyle::Dashed => "dashed",
            EdgeStyle::Dotted => "dotted",
        })
    }
}

/// RGB colors
///
/// `Color(0, 0, 0)` is black, `Color(255, 255, 255)` is white.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    #[allow(missing_docs)]
    pub const BLACK: Self = Color(0, 0, 0);
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.0, self.1, self.2)
    }
}

/// Styling attributes defined by the node type
pub trait DotStyle<T: Tag> {
    /// Get the style for the `n`-th outgoing edge, tagged with `tag`
    ///
    /// Returns the edge style (solid/dashed/dotted), whether it is bold, and
    /// the color.
    ///
    /// The default implementation distinguishes three kinds of edges (all
    /// non-bold and black):
    ///
    /// 1. If the edge is tagged (i.e. `tag` is not the default value) then the
    ///    edge is dotted.
    /// 2. If the edge is untagged and the second edge (`no == 1`), then it is
    ///    dashed
    /// 3. Otherwise the edge is solid
    fn edge_style(no: usize, tag: T) -> (EdgeStyle, bool, Color) {
        (
            if tag != Default::default() {
                EdgeStyle::Dotted
            } else if no == 1 {
                EdgeStyle::Dashed
            } else {
                EdgeStyle::Solid
            },
            false,
            Color::BLACK,
        )
    }
}

#[cfg(feature = "dot")]
mod dot_impl;
#[cfg(feature = "dot")]
pub use dot_impl::dump_all;
