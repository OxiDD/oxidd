//! Visual representation using [dot]
//!
//! [dot]: https://graphviz.org/docs/layouts/dot/

use std::fmt;
use std::io;

use oxidd_core::function::Function;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::Edge;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::LevelView;
use oxidd_core::Manager;
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
        let str = match self {
            EdgeStyle::Solid => "solid",
            EdgeStyle::Dashed => "dashed",
            EdgeStyle::Dotted => "dotted",
        };
        write!(f, "{str}")
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

/// Dump the entire decision diagram represented by `manager` as Graphviz DOT
/// code to `file`
///
/// `variables` contains pairs of edges representing the variable and names
/// (`VD`, implementing [`std::fmt::Display`]). `functions` contains pairs of
/// edges representing the function and names (`FD`, implementing
/// [`std::fmt::Display`]). In both cases, the order of elements does not
/// matter.
pub fn dump_all<'a, 'id, F, VD, FD>(
    mut file: impl io::Write,
    manager: &F::Manager<'id>,
    variables: impl IntoIterator<Item = (&'a F, VD)>,
    functions: impl IntoIterator<Item = (&'a F, FD)>,
) -> io::Result<()>
where
    F: 'a + Function + DotStyle<<<F::Manager<'id> as Manager>::Edge as Edge>::Tag>,
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    <<F::Manager<'id> as Manager>::Edge as Edge>::Tag: fmt::Debug,
    <F::Manager<'id> as Manager>::Terminal: fmt::Display,
    VD: fmt::Display + Clone,
    FD: fmt::Display,
{
    writeln!(file, "digraph DD {{")?;

    let mut last_level = LevelNo::MAX;
    let mut levels = vec![None; manager.num_levels() as usize];

    for (f, label) in variables {
        let node = manager
            .get_node(f.as_edge(manager))
            .expect_inner("variables must not be const terminals");
        levels[node.level() as usize] = Some(label);
    }

    let mut edges = Vec::new();

    for level in manager.levels() {
        for edge in level.iter() {
            let id = edge.node_id();
            let node = manager
                .get_node(edge)
                .expect_inner("unique tables should not include terminal nodes");
            let rc = node.ref_count();
            let level = node.level();

            // note outgoing edges
            // TODO: maybe use a second pass instead?
            for (idx, child) in node.children().enumerate() {
                edges.push(('x', id, child.node_id(), idx, child.tag()));
            }

            if level != last_level {
                if last_level != LevelNo::MAX {
                    // No closing braces before the first level
                    writeln!(file, "  }};")?;
                }

                if let Some(level_name) = &levels[level as usize] {
                    // TODO: escaping for level_name
                    writeln!(
                        file,
                        "  {{ rank = same; l{level:x} [label=\"{level_name}\", shape=none, tooltip=\"level {level}\"];"
                    )?;
                } else {
                    writeln!(
                        file,
                        "  {{ rank = same; l{level:x} [label=\"{level}\", color=\"#AAAAAA\", shape=none, tooltip=\"level {level}\"];"
                    )?;
                }
                last_level = level;
            }

            // Unreferenced nodes are gray
            let color = if rc == 0 { ", color=\"#AAAAAA\"" } else { "" };
            writeln!(
                file,
                "    x{id:x} [label=\"\"{color}, tooltip=\"id: {id:#x}, rc: {rc}\"];"
            )?;
        }
    }
    writeln!(file, "  }};")?;

    const TERMINAL_LEVEL: LevelNo = LevelNo::MAX;
    writeln!(file, "  {{ rank = same; l{TERMINAL_LEVEL:x} [label=\"-\", shape=none, tooltip=\"level {TERMINAL_LEVEL} (terminals)\"];")?;
    for edge in manager.terminals() {
        let edge = EdgeDropGuard::new(manager, edge);
        let id = edge.node_id();
        let node = manager.get_node(&*edge);
        let terminal = node.unwrap_terminal();
        writeln!(
            file,
            "    x{id:x} [label=\"{terminal}\", tooltip=\"terminal, id: 0x{id:x}\"];"
        )?;
    }
    writeln!(file, "  }};")?;

    if !levels.is_empty() {
        // If `levels` is empty, there is no point in writing the order (even
        // if there are terminals).
        write!(file, "  ")?;
        for level in 0..levels.len() {
            write!(file, "l{level:x} -> ")?;
        }
        // spell-checker:ignore invis
        writeln!(file, "l{TERMINAL_LEVEL:x} [style=invis];")?;
    }

    for (i, (func, label)) in functions.into_iter().enumerate() {
        // TODO: escape label string
        writeln!(file, "  f{i:x} [label=\"{label}\", shape=box];")?;
        let edge = func.as_edge(manager);
        edges.push(('f', i, edge.node_id(), 0, edge.tag()));
    }

    for (parent_type, parent, child, idx, tag) in edges {
        let (style, bold, color) = F::edge_style(idx, tag);
        let bold = if bold { ",bold" } else { "" };
        writeln!(
            file,
            "  {parent_type}{parent:x} -> x{child:x} [style=\"{style}{bold}\", color=\"{color}\", tooltip=\"child {idx}, tag: {tag:?}\"];"
        )?;
    }

    writeln!(file, "}}")?;
    Ok(())
}
