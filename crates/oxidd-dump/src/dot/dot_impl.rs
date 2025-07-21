use std::fmt;
use std::io;

use oxidd_core::function::{ETagOfFunc, Function, INodeOfFunc, TermOfFunc};
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::{Edge, HasLevel, InnerNode, LevelNo, LevelView, Manager};

/// Dump the entire decision diagram represented by `manager` as Graphviz DOT
/// code to `file`
///
/// To label functions in the decision diagram, you can pass pairs of function
/// and name (type `D`, implementing [`std::fmt::Display`]).
pub fn dump_all<'a, 'id, F, D>(
    mut file: impl io::Write,
    manager: &F::Manager<'id>,
    functions: impl IntoIterator<Item = (&'a F, D)>,
) -> io::Result<()>
where
    F: 'a + Function + super::DotStyle<ETagOfFunc<'id, F>>,
    INodeOfFunc<'id, F>: HasLevel,
    ETagOfFunc<'id, F>: fmt::Debug,
    TermOfFunc<'id, F>: fmt::Display,
    D: fmt::Display,
{
    writeln!(file, "digraph DD {{")?;

    let mut last_level = LevelNo::MAX;

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

                let level_name = manager.var_name(manager.level_to_var(level));
                if level_name.is_empty() {
                    writeln!(
                        file,
                        "  {{ rank = same; l{level:x} [label=\"{level}\", color=\"#AAAAAA\", shape=none, tooltip=\"level {level}\"];"
                    )?;
                } else {
                    // TODO: escaping for level_name
                    writeln!(
                        file,
                        "  {{ rank = same; l{level:x} [label=\"{level_name}\", shape=none, tooltip=\"level {level}\"];"
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

    let num_levels = manager.num_levels();
    if num_levels != 0 {
        // If `levels` is empty, there is no point in writing the order (even
        // if there are terminals).
        write!(file, "  ")?;
        for level in 0..num_levels {
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
