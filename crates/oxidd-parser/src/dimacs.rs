//! DIMACS CNF/SAT parser based on the paper
//! "[Satisfiability Suggested Format][spec]"
//!
//! The parsers perform some trivial simplifications on the CNF or the SAT
//! formula: CNFs with empty clauses become
//! [`Literal::FALSE`][crate::Literal::FALSE], empty CNFs become
//! [`Literal::TRUE`][crate::Literal::TRUE]. Likewise, empty disjunctions,
//! conjunctions, etc., in SAT formulas are replaced by the respective constant.
//! Equivalence operators are replaced by XOR and negation (since
//! [`Circuit`s][crate::Circuit] do not support equivalence). Conjunction,
//! disjunction, and XOR operators with just a single operand are discarded.
//!
//! In addition to regular OR clauses, the CNF parser also supports [XOR
//! clauses][xcnf] (XCNF).
//!
//! [spec]: https://www21.in.tum.de/~lammich/2015_SS_Seminar_SAT/resources/dimacs-cnf.pdf
//! [xcnf]: https://www.msoos.org/xor-clauses/

// spell-checker:ignore multispace

use rustc_hash::FxHashSet;

use nom::bytes::complete::tag;
use nom::character::complete::{char, line_ending, multispace0, space0, space1, u64};
use nom::combinator::{consumed, cut, eof, value};
use nom::error::{context, ContextError, ErrorKind, FromExternalError, ParseError};
use nom::multi::many0_count;
use nom::sequence::{preceded, terminated};
use nom::{Err, IResult};

use crate::util::{
    self, context_loc, eol, fail, fail_with_contexts, line_span, word, word_span, MAX_CAPACITY,
};
use crate::{ParseOptions, Problem, Tree, Var, VarSet};

// spell-checker:ignore SATX,SATEX

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    CNF,
    SAT { xor: bool, eq: bool },
}
#[rustfmt::skip]
const SAT: Format = Format::SAT { xor: false, eq: false };
#[rustfmt::skip]
const SATX: Format = Format::SAT { xor: true, eq: false };
#[rustfmt::skip]
const SATE: Format = Format::SAT { xor: false, eq: false };
#[rustfmt::skip]
const SATEX: Format = Format::SAT { xor: true, eq: true };

fn format<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
    input: &'a [u8],
) -> IResult<&'a [u8], Format, E> {
    let inner = |input: &'a [u8]| match input {
        [b'c', b'n', b'f', r @ ..] => Ok((r, Format::CNF)),
        [b's', b'a', b't', b'e', b'x', r @ ..] => Ok((r, SATEX)),
        [b's', b'a', b't', b'e', r @ ..] => Ok((r, SATE)),
        [b's', b'a', b't', b'x', r @ ..] => Ok((r, SATX)),
        [b's', b'a', b't', r @ ..] => Ok((r, SAT)),
        _ => Err(Err::Error(E::from_error_kind(input, ErrorKind::Alt))),
    };

    context_loc(
        || word_span(input),
        "format must be one of 'cnf', 'sat', 'satx', 'sate', or 'satex'",
        word(inner),
    )(input)
}

/// Parses a problem line, i.e., `p cnf <#vars> <#clauses>` or
/// `p sat[e][x] <#vars>`
///
/// Returns the format, number of vars and in case of `cnf` format the number of
/// clauses together with their spans.
fn problem_line<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
    input: &'a [u8],
) -> IResult<&'a [u8], ((&'a [u8], Format), (&'a [u8], usize), (&'a [u8], usize)), E> {
    let inner = |input| {
        let (input, _) = context(
            "all lines in the preamble must begin with 'c' or 'p'",
            cut(char('p')),
        )(input)?;
        let (input, _) = space1(input)?;
        let (input, fmt) = consumed(format)(input)?;
        let (input, _) = space1(input)?;
        let (input, num_vars) = consumed(u64)(input)?;
        if num_vars.1 > MAX_CAPACITY {
            return fail(num_vars.0, "too many variables");
        }
        let num_vars = (num_vars.0, num_vars.1 as usize);
        if fmt.1 == Format::CNF {
            let msg = "expected the number of clauses (CNF format)";
            let (input, _) = context(msg, space1)(input)?;
            let (input, num_clauses) = context_loc(|| word_span(input), msg, consumed(u64))(input)?;
            if num_clauses.1 > MAX_CAPACITY {
                return fail(num_clauses.0, "too many clauses");
            }
            let num_clauses = (num_clauses.0, num_clauses.1 as usize);
            let (input, _) = space0(input)?;
            value((fmt, num_vars, num_clauses), line_ending)(input)
        } else {
            let (input, _) = space0(input)?;
            context(
                "expected a line break (SAT formats do not take a number of clauses)",
                value((fmt, num_vars, ([].as_slice(), 0)), line_ending),
            )(input)
        }
    };

    context_loc(
        || line_span(input),
        "problem line must have format 'p <format> <#vars> [<#clauses>]'",
        cut(inner),
    )(input)
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Preamble {
    format: Format,
    vars: VarSet,
    /// Number of clauses (CNF format, otherwise 0)
    num_clauses: usize,
    clause_tree: Option<Tree<usize>>,
}

/// Parses the preamble, i.e., all `c` and `p` lines at the beginning of the
/// file
///
/// Note: `parse_orders` only instructs the parser to treat comment lines as
/// order lines. In case a var or a clause order is given, this function ensures
/// that they are valid. However, if no var order is given, then the returned
/// var order is empty, and if no clause order is given, then the returned
/// clause order is empty.
fn preamble<'a, E>(
    parse_var_order: bool,
    parse_clause_tree: bool,
) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], Preamble, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    move |mut input| {
        // TODO: `parse_clause_tree` currently requires a valid (or empty)
        // variable order. Is this what we want?
        if parse_var_order || parse_clause_tree {
            let mut vars = VarSet {
                len: 0,
                order: Vec::new(),
                order_tree: None,
                names: Vec::new(),
            };

            let mut max_var_span = [].as_slice(); // in the name mapping / linear order
            let mut tree_max_var = ([].as_slice(), 0); // dummy value
            let mut name_set: FxHashSet<&str> = Default::default();

            // clause order
            let mut clause_tree = None;
            let mut clause_order_span = [].as_slice();
            let mut max_clause = ([].as_slice(), 0); // dummy value

            loop {
                let next_input = match preceded(char::<_, E>('c'), space1)(input) {
                    Ok((i, _)) => i,
                    Err(_) => break,
                };
                if let Ok((next_input, _)) = preceded(tag("co"), space1::<_, E>)(next_input) {
                    if parse_clause_tree {
                        if clause_tree.is_some() {
                            return fail(line_span(input), "clause order may only be given once");
                        }
                        let t: Tree<usize>;
                        (input, (clause_order_span, (t, max_clause))) =
                            terminated(consumed(util::tree(false, false)), eol)(next_input)?;
                        clause_tree = Some(t);
                    } else {
                        input = match memchr::memchr(b'\n', input) {
                            Some(i) => &input[i + 1..],
                            None => &input[input.len()..],
                        };
                    }
                } else if let Ok((next_input, _)) = preceded(tag("vo"), space1::<_, E>)(next_input)
                {
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
                    let Ok(name) = std::str::from_utf8(name) else {
                        return fail(name, "invalid UTF-8");
                    };
                    if !name_set.insert(name) {
                        return fail(name.as_bytes(), "second occurrence of variable name");
                    }
                    vars.names[var] = Some(name.to_owned());
                    if vars.order_tree.is_none() {
                        vars.order.push(var);
                    }
                } else {
                    return fail(
                        line_span(input),
                        "expected a variable order record ('c <var> <name>'), a variable order tree ('c vo <tree>'), or a clause order tree ('c co <tree>')",
                    );
                }
            }

            if vars.order_tree.is_none() && vars.names.len() != vars.order.len() {
                return fail_with_contexts([
                    (input, "expected another variable order line"),
                    (max_var_span, "note: maximal variable number given here"),
                ]);
            }

            let (next_input, (format, num_vars, num_clauses)) = problem_line(input)?;
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

            if clause_tree.is_some() {
                if format.1 != Format::CNF {
                    let msg0 = "clause tree only supported for 'cnf' format";
                    return fail_with_contexts([
                        (clause_order_span, msg0),
                        (format.0, "note: format given here"),
                    ]);
                }
                if max_clause.1 != num_clauses.1 - 1 {
                    return fail_with_contexts([
                        (num_clauses.0, "number of clauses does not match"),
                        (max_clause.0, "note: maximal clause number given here"),
                    ]);
                }
            }

            vars.len = num_vars.1;
            #[cfg(debug_assertions)]
            vars.check_valid();
            let preamble = Preamble {
                format: format.1,
                vars,
                num_clauses: num_clauses.1,
                clause_tree,
            };
            Ok((next_input, preamble))
        } else {
            let (input, (format, num_vars, num_clauses)) =
                preceded(many0_count(util::comment), problem_line)(input)?;
            let preamble = Preamble {
                format: format.1,
                vars: VarSet::new(num_vars.1),
                num_clauses: num_clauses.1,
                clause_tree: None,
            };
            Ok((input, preamble))
        }
    }
}

mod cnf {
    use nom::branch::alt;
    use nom::character::complete::{char, multispace0, one_of, u64};
    use nom::combinator::{consumed, eof, iterator, map, recognize};
    use nom::error::{context, ContextError, ErrorKind, FromExternalError, ParseError};
    use nom::sequence::preceded;
    use nom::{Err, IResult};

    use crate::util::fail;
    use crate::{Circuit, GateKind, Literal, Problem, Tree};

    use super::Preamble;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum CNFTokenKind {
        Int(u64),
        Neg,
        Xor,
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct CNFToken<'a> {
        span: &'a [u8],
        kind: CNFTokenKind,
    }

    fn lex<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        input: &'a [u8],
    ) -> IResult<&'a [u8], CNFToken<'a>, E> {
        let tok = alt((
            map(consumed(u64), |(span, n)| CNFToken {
                span,
                kind: CNFTokenKind::Int(n),
            }),
            map(recognize(char('-')), |span| CNFToken {
                span,
                kind: CNFTokenKind::Neg,
            }),
            map(recognize(one_of("xX")), |span| CNFToken {
                span,
                kind: CNFTokenKind::Xor,
            }),
        ));

        preceded(multispace0, tok)(input)
    }

    fn make_conj_tree(
        circuit: &mut Circuit,
        conjuncts: &[Literal],
        tree: Tree<usize>,
        stack: &mut Vec<Literal>,
    ) -> Literal {
        match tree {
            Tree::Inner(children) => {
                let saved_stack_len = stack.len();
                for child in children {
                    let l = make_conj_tree(circuit, conjuncts, child, stack);
                    stack.push(l);
                }
                let root = circuit.push_gate(GateKind::And);
                circuit.push_gate_inputs(stack[saved_stack_len..].iter().copied());
                stack.truncate(saved_stack_len);
                root
            }
            Tree::Leaf(i) => conjuncts[i],
        }
    }

    pub fn parse<'a, E>(
        preamble: Preamble,
    ) -> impl FnOnce(&'a [u8]) -> IResult<&'a [u8], Problem, E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
    {
        move |input| {
            let Preamble {
                vars,
                num_clauses,
                clause_tree: clause_order_tree,
                ..
            } = preamble;
            let num_vars = vars.len();
            let mut circuit = Circuit::new(vars);
            circuit.push_gate(GateKind::Or);

            let mut neg = false;

            let mut it = iterator(input, lex::<E>);
            for token in &mut it {
                match token.kind {
                    CNFTokenKind::Int(0) => {
                        circuit.push_gate(GateKind::Or);
                    }
                    CNFTokenKind::Int(n) => {
                        if n > num_vars as u64 {
                            return fail(token.span, "variables must be in range [1, #vars]");
                        }
                        circuit.push_gate_input(Literal::from_input(neg, (n - 1) as usize));
                        neg = false;
                    }
                    CNFTokenKind::Neg if !neg => neg = true,
                    CNFTokenKind::Neg => return fail(token.span, "expected a variable"),
                    CNFTokenKind::Xor => {
                        if let Some(gate) = circuit.last_gate() {
                            if !gate.inputs.is_empty() {
                                return fail(token.span, "XOR clauses must be marked as such at the beginning of the clause");
                            }
                            circuit.set_last_gate_kind(GateKind::Xor);
                        }
                    }
                }
            }

            let (input, ()) = it.finish()?;
            let (input, _) = multispace0(input)?;
            let (input, _) = context("expected a literal or '0'", eof)(input)?;

            let num_gates = circuit.num_gates();
            if num_gates != num_clauses {
                // The last clause may or may not be terminated by 0. In case it is, we called
                // `push_clause()` once too often.
                if num_gates == num_clauses + 1 && circuit.last_gate().unwrap().inputs.is_empty() {
                    circuit.pop_gate();
                } else {
                    return Err(Err::Failure(E::from_external_error(
                        input,
                        ErrorKind::Fail,
                        format!("expected {num_clauses} clauses, got {num_gates}"),
                    )));
                }
            }
            let num_gates = circuit.num_gates(); // may have been decremented above

            let root = if num_gates == 0 {
                Literal::TRUE
            } else {
                let mut is_false = false;
                let mut conj = Vec::with_capacity(num_gates);
                let mut gate = 0;
                circuit.retain_gates(|inputs| {
                    if is_false {
                        return false;
                    }
                    match inputs {
                        [] => {
                            is_false = true;
                            false
                        }
                        [l] => {
                            conj.push(*l);
                            false
                        }
                        _ => {
                            conj.push(Literal::from_gate(false, gate));
                            gate += 1;
                            true
                        }
                    }
                });

                if is_false {
                    circuit.clear_gates();
                    Literal::FALSE
                } else if let Some(tree) = clause_order_tree {
                    let mut stack = Vec::with_capacity(num_clauses);
                    make_conj_tree(&mut circuit, &conj, tree, &mut stack)
                } else {
                    let root = circuit.push_gate(GateKind::And);
                    circuit.push_gate_inputs(conj);
                    root
                }
            };

            Ok((
                input,
                Problem {
                    circuit,
                    details: crate::ProblemDetails::Root(root),
                },
            ))
        }
    }
}

mod sat {
    use nom::branch::alt;
    use nom::bytes::complete::tag;
    use nom::character::complete::{char, multispace0, u64};
    use nom::combinator::{consumed, map, recognize, value};
    use nom::error::{ContextError, ErrorKind, ParseError};
    use nom::Err;
    use nom::IResult;

    use crate::util::{fail, map_res_fail, word};
    use crate::{Circuit, GateKind, Literal, Problem, ProblemDetails, Var, VarSet};

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum TokenKind {
        Var(Var),
        Lpar,
        Rpar,
        Neg,
        And,
        Or,
        Xor,
        Eq,
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct Token<'a> {
        span: &'a [u8],
        kind: TokenKind,
    }

    macro_rules! match_tok {
        ($matcher:expr, $tok:ident) => {
            map(recognize($matcher), |span| {
                Some(Token {
                    span,
                    kind: TokenKind::$tok,
                })
            })
        };
    }

    fn lex<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        num_vars: usize,
    ) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], Option<Token<'a>>, E> {
        move |input| {
            let (input, _) = multispace0(input)?; // should never fail
            if input.is_empty() {
                return Ok((input, None));
            }
            alt((
                map_res_fail(consumed(u64), |(span, n)| {
                    if n == 0 || n > num_vars as u64 {
                        Err((span, "variables must be in range [1, #vars]"))
                    } else {
                        Ok(Some(Token {
                            span,
                            kind: TokenKind::Var(n as usize),
                        }))
                    }
                }),
                match_tok!(char('('), Lpar),
                match_tok!(char(')'), Rpar),
                match_tok!(char('-'), Neg),
                match_tok!(char('*'), And),
                match_tok!(char('+'), Or),
                match_tok!(word(tag("xor")), Xor),
                match_tok!(char('='), Eq),
            ))(input)
        }
    }

    #[derive(Debug)]
    enum SatParserErr<'a, E> {
        E(E),
        Rpar { input: &'a [u8], span: &'a [u8] },
    }
    impl<'a, E: ParseError<&'a [u8]>> ParseError<&'a [u8]> for SatParserErr<'a, E> {
        fn from_error_kind(input: &'a [u8], kind: ErrorKind) -> Self {
            Self::E(E::from_error_kind(input, kind))
        }

        fn append(input: &'a [u8], kind: ErrorKind, other: Self) -> Self {
            match other {
                Self::E(other) => Self::E(E::append(input, kind, other)),
                Self::Rpar { .. } => unreachable!(),
            }
        }
    }
    impl<'a, E: ContextError<&'a [u8]>> ContextError<&'a [u8]> for SatParserErr<'a, E> {
        fn add_context(input: &'a [u8], ctx: &'static str, other: Self) -> Self {
            match other {
                Self::E(other) => Self::E(E::add_context(input, ctx, other)),
                Self::Rpar { .. } => other,
            }
        }
    }

    fn expect<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        kind: TokenKind,
        err: &'static str,
        num_vars: usize,
    ) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], (), E> {
        move |input| {
            let (input, tok) = lex(num_vars)(input)?;
            match tok {
                None => fail(input, err),
                Some(tok) if tok.kind != kind => fail(tok.span, err),
                _ => Ok((input, ())),
            }
        }
    }

    fn formula<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        allow_xor: bool,
        allow_eq: bool,
        circuit: &mut Circuit,
        stack: &mut Vec<Literal>,
        input: &'a [u8],
    ) -> IResult<&'a [u8], Literal, SatParserErr<'a, E>> {
        let num_vars = circuit.inputs().len();
        let (input, tok) = lex(num_vars)(input)?;
        let tok = match tok {
            Some(tok) => tok,
            None => return fail(input, "expected a formula"),
        };

        match tok.kind {
            TokenKind::Var(n) => Ok((input, Literal::from_input(false, n - 1))),
            TokenKind::Lpar => {
                let (input, l) = formula(allow_xor, allow_eq, circuit, stack, input)?;
                value(l, expect(TokenKind::Rpar, "expected ')'", num_vars))(input)
            }
            TokenKind::Rpar => Err(Err::Error(SatParserErr::Rpar {
                input,
                span: tok.span,
            })),
            TokenKind::Neg => {
                let (input, tok) = lex(num_vars)(input)?;
                let tok = match tok {
                    Some(t) => t,
                    None => return fail(input, "expected a variable or '('"),
                };
                match tok.kind {
                    TokenKind::Var(n) => Ok((input, Literal::from_input(true, n - 1))),
                    TokenKind::Lpar => {
                        let (input, l) = formula(allow_xor, allow_eq, circuit, stack, input)?;
                        value(!l, expect(TokenKind::Rpar, "expected ')'", num_vars))(input)
                    }
                    _ => fail(tok.span, "expected a variable or '('"),
                }
            }
            TokenKind::Xor if !allow_xor => fail(
                tok.span,
                "'xor' is only allowed in formats 'satx' and 'satex'",
            ),
            TokenKind::Eq if !allow_eq => fail(
                tok.span,
                "'=' is only allowed in formats 'sate' and 'satex'",
            ),
            _ => {
                let (mut input, ()) = expect(TokenKind::Lpar, "expected '('", num_vars)(input)?;

                let saved_stack_len = stack.len();
                let input = loop {
                    match formula(allow_xor, allow_eq, circuit, stack, input) {
                        Ok((i, sub)) => {
                            input = i;
                            stack.push(sub)
                        }
                        Err(Err::Error(SatParserErr::Rpar { input, .. })) => break input,
                        Err(f) => {
                            stack.truncate(saved_stack_len);
                            return Err(f);
                        }
                    }
                };

                let children = &stack[saved_stack_len..];
                let literal = match children {
                    [] => match tok.kind {
                        TokenKind::And | TokenKind::Eq => Literal::TRUE,
                        TokenKind::Or | TokenKind::Xor => Literal::FALSE,
                        _ => unreachable!(),
                    },
                    [l] => *l,
                    _ => {
                        let l = circuit.push_gate(match tok.kind {
                            TokenKind::And => GateKind::And,
                            TokenKind::Or => GateKind::Or,
                            _ => GateKind::Xor,
                        });

                        circuit.push_gate_inputs(children.iter().copied());

                        if tok.kind == TokenKind::Eq && children.len() % 2 == 0 {
                            !l
                        } else {
                            l
                        }
                    }
                };
                stack.truncate(saved_stack_len);

                Ok((input, literal))
            }
        }
    }

    pub fn parse<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        vars: VarSet,
        allow_xor: bool,
        allow_eq: bool,
    ) -> impl FnOnce(&'a [u8]) -> IResult<&'a [u8], Problem, E> {
        let num_vars = vars.len();
        let mut circuit = Circuit::new(vars);
        let mut stack = Vec::with_capacity(2 * num_vars);
        move |input| match formula(allow_xor, allow_eq, &mut circuit, &mut stack, input) {
            Ok((input, root)) => Ok((
                input,
                Problem {
                    circuit,
                    details: ProblemDetails::Root(root),
                },
            )),
            Err(e) => Err(e.map(|e| match e {
                SatParserErr::E(e) => e,
                SatParserErr::Rpar { input, span } => E::add_context(
                    span,
                    "expected a formula",
                    E::from_error_kind(input, ErrorKind::Fail),
                ),
            })),
        }
    }
}

/// Parse a DIMACS CNF/SAT file
pub fn parse<'a, E>(options: &ParseOptions) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], Problem, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    let parse_var_order = options.var_order;
    let parse_clause_tree = options.clause_tree;
    move |input| {
        let (input, preamble) = preamble(parse_var_order, parse_clause_tree)(input)?;
        match preamble.format {
            Format::CNF => cnf::parse(preamble)(input),
            Format::SAT { xor, eq } => {
                let (input, res) = sat::parse(preamble.vars, xor, eq)(input)?;
                let (input, _) = context(
                    "expected end of file (SAT files may only contain a single formula)",
                    preceded(multispace0, eof),
                )(input)?;
                Ok((input, res))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nom::Finish;

    use crate::util::test::*;
    use crate::{Gate, Literal};

    use super::*;

    #[test]
    fn example_cnf() {
        let input = "c Example CNF format file
c
p cnf 4 3
1 3 -4 0
4 0 2
-3";
        let (input, problem) = parse::<()>(&OPTS_NO_ORDER)(input.as_bytes())
            .finish()
            .unwrap();
        assert!(input.is_empty());

        let (circuit, root) = unwrap_problem(problem);
        let inputs = circuit.inputs();
        assert_eq!(inputs.len(), 4);
        assert!(inputs.order().is_none());

        assert_eq!(root, g(2));
        assert_eq!(circuit.gate(root), Some(Gate::and(&[g(0), v(3), g(1)])));
        assert_eq!(circuit.gate(g(0)), Some(Gate::or(&[v(0), v(2), !v(3)])));
        assert_eq!(circuit.gate(g(1)), Some(Gate::or(&[v(1), !v(2)])));
    }

    #[test]
    fn example_cnf_0term() {
        let input = "c Example CNF format file
c
p cnf 4 3
1 3 -4 0
4 0 2
-3 0";
        let (input, problem) = parse::<()>(&OPTS_NO_ORDER)(input.as_bytes())
            .finish()
            .unwrap();
        assert!(input.is_empty());

        let (circuit, root) = unwrap_problem(problem);
        let inputs = circuit.inputs();
        assert_eq!(inputs.len(), 4);
        assert!(inputs.order().is_none());

        assert_eq!(root, g(2));
        assert_eq!(circuit.gate(root), Some(Gate::and(&[g(0), v(3), g(1)])));
        assert_eq!(circuit.gate(g(0)), Some(Gate::or(&[v(0), v(2), !v(3)])));
        assert_eq!(circuit.gate(g(1)), Some(Gate::or(&[v(1), !v(2)])));
    }

    #[test]
    fn empty_cnf() {
        let input = "p cnf 0 0\n";
        let (input, problem) = parse::<()>(&OPTS_NO_ORDER)(input.as_bytes())
            .finish()
            .unwrap();
        assert!(input.is_empty());

        let (circuit, root) = unwrap_problem(problem);
        let inputs = circuit.inputs();
        assert_eq!(inputs.len(), 0);

        assert_eq!(root, Literal::TRUE);
    }

    #[test]
    fn example_sat() {
        let input = "c Sample SAT format
c
p sat 4
(*(+(1 3 -4)
    +(4)
    +(2 3)))";
        let (input, problem) = parse::<()>(&OPTS_NO_ORDER)(input.as_bytes())
            .finish()
            .unwrap();
        assert!(input.is_empty());

        let (circuit, root) = unwrap_problem(problem);
        let inputs = circuit.inputs();
        assert_eq!(inputs.len(), 4);
        assert!(inputs.order().is_none());

        assert_eq!(root, g(2));
        assert_eq!(circuit.gate(root), Some(Gate::and(&[g(0), v(3), g(1)])));
        assert_eq!(circuit.gate(g(0)), Some(Gate::or(&[v(0), v(2), !v(3)])));
        assert_eq!(circuit.gate(g(1)), Some(Gate::or(&[v(1), v(2)])));
    }

    #[test]
    fn preamble_satx() {
        let (input, preamble) = preamble::<()>(false, false)(b"p satx 1337 \n").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            preamble,
            Preamble {
                format: SATX,
                vars: VarSet::new(1337),
                num_clauses: 0,
                clause_tree: None
            }
        );
    }

    #[test]
    fn preamble_sate() {
        let (input, preamble) = preamble::<()>(false, false)(b"p sate 1\n").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            preamble,
            Preamble {
                format: SATE,
                vars: VarSet::new(1),
                num_clauses: 0,
                clause_tree: None
            }
        );
    }

    #[test]
    fn preamble_satex() {
        let (input, preamble) = preamble::<()>(false, false)(b"p satex 42 \n").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            preamble,
            Preamble {
                format: SATEX,
                vars: VarSet::new(42),
                num_clauses: 0,
                clause_tree: None
            }
        );
    }
}
