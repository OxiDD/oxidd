//! Negation normal form parser for an extended version of [c2d's][c2d] d-DNNF
//! output format
//!
//! The format extensions subsume [Bella's][bella] wDNNF format.
//!
//! [c2d]: http://reasoning.cs.ucla.edu/c2d/
//! [bella]: https://github.com/Illner/BellaCompiler

// spell-checker:ignore multispace

use nom::bytes::complete::tag;
use nom::character::complete::{char, i64, line_ending, multispace0, space0, space1, u64};
use nom::combinator::{consumed, cut, eof, value};
use nom::error::{context, ContextError, FromExternalError, ParseError};
use nom::multi::many0_count;
use nom::sequence::{preceded, terminated};
use nom::{IResult, Offset};
use rustc_hash::FxHashSet;

use crate::util::{
    self, context_loc, eol, fail, fail_with_contexts, line_span, usize, word_span, MAX_CAPACITY,
};
use crate::{Circuit, GateKind, Literal, ParseOptions, Problem, Tree, Var, VarSet};

/// Parses a problem line, i.e., `nnf <#nodes> <#edges> <#inputs>`
///
/// Returns the three numbers along with their spans.
fn problem_line<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
    input: &'a [u8],
) -> IResult<&'a [u8], [(&'a [u8], usize); 3], E> {
    let inner = |input| {
        let (input, _) = context(
            "all lines in the preamble must begin with 'c' or 'nnf'",
            cut(tag("nnf")),
        )(input)?;
        let (input, _) = space1(input)?;
        let (input, num_nodes) = consumed(usize)(input)?;
        let (input, _) = space1(input)?;
        let (input, num_edges) = consumed(usize)(input)?;
        let (input, _) = space1(input)?;
        let (input, num_inputs) = consumed(usize)(input)?;
        value([num_nodes, num_edges, num_inputs], line_ending)(input)
    };

    context_loc(
        || line_span(input),
        "problem line must have format 'nnf <#nodes> <#edges> <#inputs>'",
        cut(inner),
    )(input)
}

/// Parses the preamble, i.e., all `c` and `nnf` lines at the beginning of the
/// file
///
/// Note: `parse_orders` only instructs the parser to treat comment lines as
/// order lines. In case a var order is given, this function ensures that they
/// are valid. However, if no var order is given, then the returned var order is
/// empty.
fn preamble<'a, E>(
    parse_var_order: bool,
) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], (VarSet, [(&'a [u8], usize); 3]), E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    move |mut input| {
        if parse_var_order {
            let mut vars = VarSet {
                len: 0,
                order: Vec::new(),
                order_tree: None,
                names: Vec::new(),
            };

            let mut max_var_span = [].as_slice(); // in the name mapping / linear order
            let mut tree_max_var = ([].as_slice(), 0); // dummy value
            let mut name_set: FxHashSet<&str> = Default::default();

            loop {
                let next_input = match preceded(char::<_, E>('c'), space1)(input) {
                    Ok((i, _)) => i,
                    Err(_) => break,
                };
                if let Ok((next_input, _)) = preceded(tag("vo"), space1::<_, E>)(next_input) {
                    // variable order tree
                    if vars.order_tree.is_some() {
                        let msg = "variable order tree may only be given once";
                        return fail(line_span(input), msg);
                    }
                    let t: Tree<Var>;
                    (input, (t, tree_max_var)) =
                        terminated(util::tree(true, true), eol)(next_input)?;

                    // The variable order tree takes precedence (and determines the linear order)
                    vars.order.clear();
                    vars.order.reserve(tree_max_var.1 + 1);
                    t.flatten_into(&mut vars.order);
                    vars.order_tree = Some(t);
                } else if let Ok((next_input, ((var_span, var), name))) =
                    util::var_order_record::<E>(next_input)
                {
                    // var order line
                    input = next_input;
                    if var == 0 {
                        return fail(var_span, "variable number must be greater than 0");
                    }
                    if var > MAX_CAPACITY {
                        return fail(var_span, "variable number too large");
                    }

                    let num_vars = var as usize;
                    let var = num_vars - 1;

                    if num_vars > vars.names.len() {
                        vars.names.resize(num_vars, None);
                        vars.order.reserve(num_vars - vars.names.len());
                        max_var_span = var_span;
                    } else if vars.names[var].is_some() {
                        return fail(var_span, "second occurrence of variable in order");
                    }
                    // always write Some to mark the variable as present
                    vars.names[var] = Some(if let Some(name) = name {
                        let Ok(name) = std::str::from_utf8(name) else {
                            return fail(name, "invalid UTF-8");
                        };
                        if !name_set.insert(name) {
                            return fail(name.as_bytes(), "second occurrence of variable name");
                        }
                        name.to_owned()
                    } else {
                        String::new()
                    });
                    if vars.order_tree.is_none() {
                        vars.order.push(var);
                    }
                } else {
                    return fail(
                        line_span(input),
                        "expected a variable order record ('c <var> [<name>]') or a variable order tree ('c vo <tree>')",
                    );
                }
            }

            if vars.order_tree.is_none() && vars.names.len() != vars.order.len() {
                return fail_with_contexts([
                    (input, "expected another variable order line"),
                    (max_var_span, "note: maximal variable number given here"),
                ]);
            }

            let (next_input, sizes) = problem_line(input)?;
            let [_, _, num_vars] = sizes;
            if vars.order_tree.is_none() {
                if !vars.order.is_empty() && num_vars.1 != vars.order.len() {
                    return fail_with_contexts([
                        (num_vars.0, "number of variables does not match"),
                        (max_var_span, "note: maximal variable number given here"),
                    ]);
                }
            } else {
                if num_vars.1 != tree_max_var.1 + 1 {
                    return fail_with_contexts([
                        (num_vars.0, "number of variables does not match"),
                        (tree_max_var.0, "note: maximal variable number given here"),
                    ]);
                }
                if vars.names.len() > num_vars.1 {
                    return fail_with_contexts([
                        (max_var_span, "name assigned to non-existing variable"),
                        (num_vars.0, "note: number of variables given here"),
                    ]);
                }
            }

            // cleanup: we used `Some(String::new())` to mark unnamed variables as present
            while let Some(name) = vars.names.last() {
                if !name.as_ref().is_some_and(String::is_empty) {
                    break;
                }
                vars.names.pop();
            }
            for name in &mut vars.names {
                if name.as_ref().is_some_and(String::is_empty) {
                    *name = None;
                }
            }

            vars.len = num_vars.1;
            #[cfg(debug_assertions)]
            vars.check_valid();
            Ok((next_input, (vars, sizes)))
        } else {
            let (input, sizes) = preceded(many0_count(util::comment), problem_line)(input)?;
            let [_, _, (_, num_vars)] = sizes;
            Ok((input, (VarSet::new(num_vars), sizes)))
        }
    }
}

/// Parse a NNF file
pub fn parse<'a, E>(options: &ParseOptions) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], Problem, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    let parse_var_orders = options.var_order;
    let check_acyclic = options.check_acyclic;

    move |input| {
        let (mut input, (vars, [num_nodes, num_edges, num_inputs])) =
            preamble(parse_var_orders)(input)?;

        if num_nodes.1 == 0 {
            return fail(num_nodes.0, "NNF must have at least one node");
        }

        let mut circuit = Circuit::new(vars);
        circuit.reserve_gates(num_nodes.1);
        circuit.reserve_gate_inputs(num_edges.1);

        let mut nodes = Vec::with_capacity(num_nodes.1);
        let mut gate_spans = Vec::with_capacity(num_nodes.1);

        for _ in 0..num_nodes.1 {
            let (inp, l) = match input {
                [kind @ (b'A' | b'a' | b'B' | b'b' | b'X' | b'x'), inp @ ..] => {
                    let kind = if let b'X' | b'x' = kind {
                        GateKind::Xor
                    } else {
                        GateKind::And
                    };
                    let (mut inp, children) = preceded(space1, u64)(inp)?;
                    if children == 0 {
                        (inp, kind.empty_gate())
                    } else {
                        let l = circuit.push_gate(kind);
                        for _ in 0..children {
                            let (i, child) = preceded(space1, consumed(u64))(inp)?;
                            inp = i;

                            if child.1 >= num_nodes.1 as u64 {
                                return fail_with_contexts([
                                    (child.0, "invalid node number"),
                                    (num_nodes.0, "number of nodes given here"),
                                ]);
                            }
                            // In contrast to the original c2d format, we do not enforce a
                            // topological order on the input. We just collect the node IDs now and
                            // map them to `Literal`s later.
                            circuit.push_gate_input(Literal(child.1 as usize));
                        }
                        gate_spans.push(&input[..input.offset(inp)]);
                        (inp, l)
                    }
                }
                [b'O' | b'o', inp @ ..] => {
                    let (inp, conflict) = preceded(space1, consumed(u64))(inp)?;
                    if conflict.1 > num_inputs.1 as u64 {
                        return fail_with_contexts([
                            (conflict.0, "invalid variable"),
                            (num_inputs.0, "number of input variables given here"),
                        ]);
                    }

                    let (mut inp, children) = preceded(space1, consumed(u64))(inp)?;

                    if conflict.1 != 0 && children.1 != 2 {
                        return fail_with_contexts([
                            (children.0, "expected 2 children, since a conflict variable is given"),
                            (conflict.0, "using 0 in place of the conflict variable here allows arbitrary arity"),
                        ]);
                    }

                    if children.1 == 0 {
                        (inp, Literal::FALSE)
                    } else {
                        let l = circuit.push_gate(GateKind::Or);
                        for _ in 0..children.1 {
                            let (i, child) = preceded(space1, consumed(u64))(inp)?;
                            inp = i;

                            if child.1 >= num_nodes.1 as u64 {
                                return fail_with_contexts([
                                    (child.0, "invalid node number"),
                                    (num_nodes.0, "number of nodes given here"),
                                ]);
                            }
                            circuit.push_gate_input(Literal(child.1 as usize));
                        }
                        gate_spans.push(&input[..input.offset(inp)]);
                        (inp, l)
                    }
                }
                [b'L' | b'l', inp @ ..] => {
                    let (inp, lit) = preceded(space1, consumed(i64))(inp)?;
                    let var = lit.1.unsigned_abs();
                    if var == 0 || var > num_inputs.1 as u64 {
                        return fail_with_contexts([
                            (lit.0, "invalid literal"),
                            (num_inputs.0, "number of input variables given here"),
                        ]);
                    }
                    (inp, Literal::from_input(lit.1 < 0, (var - 1) as usize))
                }
                inp => {
                    return fail(
                        word_span(inp),
                        "expected a node ('A', 'B', 'O', 'X', or 'L')",
                    )
                }
            };
            nodes.push(l);
            input = preceded(space0, line_ending)(inp)?.0;
        }

        let (input, _) = preceded(multispace0, eof)(input)?;

        for l in circuit.gates.all_elements_mut() {
            *l = nodes[l.0];
        }
        if check_acyclic {
            if let Some(l) = circuit.find_cycle() {
                return fail(
                    gate_spans[l.get_gate_no().unwrap()],
                    "node depends on itself",
                );
            }
        }

        let problem = Problem {
            circuit,
            details: crate::ProblemDetails::Root(*nodes.last().unwrap()),
        };
        Ok((input, problem))
    }
}

#[cfg(test)]
mod tests {
    use nom::Finish;

    use crate::{util::test::*, Gate};

    use super::*;

    /// NNF taken from the c2d manual
    #[test]
    fn c2d_example() {
        let input = b"nnf 15 17 4\n\
            L -3\n\
            L -2\n\
            L 1\n\
            A 3 2 1 0\n\
            L 3\n\
            O 3 2 4 3\n\
            L -4\n\
            A 2 6 5\n\
            L 4\n\
            A 2 2 8\n\
            A 2 1 4\n\
            L 2\n\
            O 2 2 11 10\n\
            A 2 12 9\n\
            O 4 2 13 7\n";

        let (input, problem) = parse::<()>(&OPTS_NO_ORDER)(input).finish().unwrap();
        assert!(input.is_empty());

        let (circuit, root) = unwrap_problem(problem);
        let inputs = circuit.inputs();
        assert_eq!(inputs.len(), 4);
        assert!(inputs.order().is_none());

        let nodes = &[
            !v(2), // L -3
            !v(1), // L -2
            v(0),  // L 1
            g(0),  // A 3 2 1 0
            v(2),  // L 3
            g(1),  // O 3 2 4 3
            !v(3), // L -4
            g(2),  // A 2 6 5
            v(3),  // L 4
            g(3),  // A 2 2 8
            g(4),  // A 2 1 4
            v(1),  // L 2
            g(5),  // O 2 2 11 10
            g(6),  // A 2 12 9
            g(7),  // O 4 2 13 7
        ];
        assert_eq!(root, *nodes.last().unwrap());

        for (i, &gate) in [
            Gate::and(&[nodes[2], nodes[1], nodes[0]]), // 0: A 3 2 1 0
            Gate::or(&[nodes[4], nodes[3]]),            // 1: O 3 2 4 3
            Gate::and(&[nodes[6], nodes[5]]),           // 2: A 2 6 5
            Gate::and(&[nodes[2], nodes[8]]),           // 3: A 2 2 8
            Gate::and(&[nodes[1], nodes[4]]),           // 4: A 2 1 4
            Gate::or(&[nodes[11], nodes[10]]),          // 5: O 2 2 11 10
            Gate::and(&[nodes[12], nodes[9]]),          // 6: A 2 12 9
            Gate::or(&[nodes[13], nodes[7]]),           // 7: O 4 2 13 7
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(circuit.gate(g(i)), Some(gate), "mismatch for gate {i}");
        }
    }
}
