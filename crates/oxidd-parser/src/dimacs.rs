//! DIMACS CNF/SAT parser based on the paper
//! "[Satisfiability Suggested Format][spec]"
//!
//! [spec]: https://www21.in.tum.de/~lammich/2015_SS_Seminar_SAT/resources/dimacs-cnf.pdf

// spell-checker:ignore multispace

use bitvec::vec::BitVec;
use rustc_hash::FxHashSet;

use nom::bytes::complete::{tag, take, take_till};
use nom::character::complete::{
    char, line_ending, multispace0, not_line_ending, space0, space1, u64,
};
use nom::combinator::{consumed, cut, eof, value};
use nom::error::{context, ContextError, ErrorKind, FromExternalError, ParseError};
use nom::multi::many0_count;
use nom::sequence::preceded;
use nom::{Err, IResult};

use crate::util::{
    context_loc, fail, fail_with_contexts, line_span, trim_end, word, word_span, MAX_CAPACITY,
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

fn comment<'a, E: ParseError<&'a [u8]>>(input: &'a [u8]) -> IResult<&'a [u8], (), E> {
    let (input, _) = char('c')(input)?;
    let (input, _) = take_till(|c| c == b'\n')(input)?;
    let (input, _) = take(1usize)(input)?;
    Ok((input, ()))
}

/// Parse a tree with unique `usize` leaves, e.g.: `[0, [2, 3, 1]]`
///
/// Returns the tree, as well as the maximal number and its span.
fn tree<'a, E>(
    one_based: bool,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (Tree<usize>, (&'a [u8], usize)), E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    fn rec<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        input: &'a [u8],
        inserted: &mut BitVec,
        buffer: &mut Vec<Tree<usize>>,
        one_based: bool,
    ) -> IResult<&'a [u8], (Tree<usize>, (&'a [u8], usize)), E> {
        debug_assert!(space0::<_, E>(input).is_err());

        if let Ok((input, (span, n))) = consumed(u64::<_, E>)(input) {
            let (input, _) = space0(input)?;

            if n > MAX_CAPACITY {
                return fail(span, "number too large");
            }
            let mut n = n as usize;
            if one_based {
                if n == 0 {
                    return fail(span, "numbers must be greater than 0");
                }
                n -= 1;
            }
            if inserted.len() <= n {
                inserted.resize(n + 1, false);
            } else if inserted[n] {
                return fail(span, "second occurrence in tree");
            }
            inserted.set(n, true);
            return Ok((input, (Tree::Leaf(n), (span, n))));
        }

        if let Ok((mut input, _)) = char::<_, E>('[')(input) {
            (input, _) = space0(input)?;
            let buffer_pos = buffer.len();
            let mut max_span = [].as_slice();
            let mut max = 0;

            let input = loop {
                (input, _) = space0(input)?;
                if let [b']', r @ ..] = input {
                    break r;
                }
                let (i, (sub, (sub_max_span, sub_max))) = rec(input, inserted, buffer, one_based)?;
                buffer.push(sub);
                if sub_max >= max {
                    max = sub_max;
                    max_span = sub_max_span;
                }

                (input, _) = space0(i)?;
                match input {
                    [b']', r @ ..] => break r,
                    [b',', r @ ..] => input = r,
                    _ => return fail(input, "expected ',' or ']'"),
                }
            };

            let t = if buffer.len() == buffer_pos + 1 {
                // flatten `[42]` into `42`
                buffer.pop().unwrap()
            } else {
                Tree::Inner(buffer.split_off(buffer_pos).into_boxed_slice())
            };
            return Ok((input, (t, (max_span, max))));
        }

        fail(word_span(input), "expected '[' or a number")
    }

    move |input| {
        let mut inserted = BitVec::new();
        let (input, _) = space0(input)?;
        let (input, (span, res)) = cut(consumed(|input| {
            rec(
                input,
                &mut inserted,
                &mut Vec::with_capacity(65536),
                one_based,
            )
        }))(input)?;
        if let Some(n) = inserted.first_zero() {
            return Err(Err::Failure(E::from_external_error(
                span,
                ErrorKind::Fail,
                format!("number {} missing in tree", n),
            )));
        }
        let (input, _) = cut(preceded(space0, line_ending))(input)?;
        Ok((input, res))
    }
}

fn var_order_record<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
    input: &'a [u8],
) -> IResult<&'a [u8], ((&'a [u8], u64), &'a [u8]), E> {
    let (input, (var_span, var)) = consumed(u64)(input)?;
    let (input, _) = space1(input)?;
    let (input, name) = not_line_ending(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, ((var_span, var), trim_end(name))))
}

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
/// `p sat(e|x|ex) <#vars>`
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
    clause_order_tree: Option<Tree<usize>>,
}

/// Parses the preamble, i.e., all `c` and `p` lines at the beginning of the
/// file
///
/// Note: `parse_orders` only instructs the parser to treat comment lines as
/// order lines. In case a var or a clause order is given, this function ensures
/// that they are valid. However, if no var order is given, then the returned
/// var order is empty, and if no clause order is given, then the returned
/// clause order is empty.
fn preamble<'a, E>(parse_orders: bool) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], Preamble, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    move |mut input| {
        if parse_orders {
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
            let mut clause_order = None;
            let mut clause_order_span = [].as_slice();
            let mut max_clause = ([].as_slice(), 0); // dummy value

            loop {
                let next_input = match preceded(char::<_, E>('c'), space1)(input) {
                    Ok((i, _)) => i,
                    Err(_) => break,
                };
                if let Ok((next_input, _)) = preceded(tag("co"), space1::<_, E>)(next_input) {
                    // clause order tree
                    if clause_order.is_some() {
                        return fail(line_span(input), "clause order may only be given once");
                    }
                    let t: Tree<usize>;
                    (input, (clause_order_span, (t, max_clause))) =
                        consumed(tree(false))(next_input)?;
                    clause_order = Some(t);
                } else if let Ok((next_input, _)) = preceded(tag("vo"), space1::<_, E>)(next_input)
                {
                    // variable order tree
                    if vars.order_tree.is_some() {
                        let msg = "variable order tree may only be given once";
                        return fail(line_span(input), msg);
                    }
                    let t: Tree<Var>;
                    (input, (t, tree_max_var)) = tree(true)(next_input)?;

                    // The variable order tree takes precedence (and determines the linear order)
                    vars.order.clear();
                    vars.order.reserve(tree_max_var.1 + 1);
                    t.flatten_into(&mut vars.order);
                    vars.order_tree = Some(t);
                } else if let Ok((next_input, ((var_span, var), name))) =
                    var_order_record::<E>(next_input)
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

            if clause_order.is_some() {
                if format.1 != Format::CNF {
                    let msg0 = "clause order only supported for 'cnf' format";
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
                clause_order_tree: clause_order,
            };
            Ok((next_input, preamble))
        } else {
            let (input, (format, num_vars, num_clauses)) =
                preceded(many0_count(comment), problem_line)(input)?;
            let preamble = Preamble {
                format: format.1,
                vars: VarSet::simple(num_vars.1),
                num_clauses: num_clauses.1,
                clause_order_tree: None,
            };
            Ok((input, preamble))
        }
    }
}

mod cnf {
    use nom::branch::alt;
    use nom::character::complete::{char, multispace0, u64};
    use nom::combinator::{consumed, eof, iterator, map, recognize};
    use nom::error::{context, ContextError, ErrorKind, FromExternalError, ParseError};
    use nom::sequence::preceded;
    use nom::{Err, IResult};

    use crate::util::fail;
    use crate::{CNFProblem, Literal, Problem, Vec2d};

    use super::Preamble;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum CNFTokenKind {
        Int(u64),
        Neg,
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct CNFToken<'a> {
        span: &'a [u8],
        kind: CNFTokenKind,
    }

    fn lex<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        input: &'a [u8],
    ) -> IResult<&'a [u8], CNFToken, E> {
        let tok = alt((
            map(consumed(u64), |(span, n)| CNFToken {
                span,
                kind: CNFTokenKind::Int(n),
            }),
            map(recognize(char('-')), |span| CNFToken {
                span,
                kind: CNFTokenKind::Neg,
            }),
        ));

        preceded(multispace0, tok)(input)
    }

    pub fn parse<'a, E>(
        preamble: Preamble,
    ) -> impl FnOnce(&'a [u8]) -> IResult<&'a [u8], Problem, E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
    {
        move |input| {
            let num_vars = preamble.vars.len();
            let num_clauses = preamble.num_clauses;

            let mut clauses = Vec2d::with_capacity(num_clauses, num_clauses * 4);
            clauses.push_vec();
            let mut neg = false;

            let mut it = iterator(input, lex::<E>);
            for token in &mut it {
                match token.kind {
                    CNFTokenKind::Int(0) => clauses.push_vec(),
                    CNFTokenKind::Int(n) => {
                        if n > num_vars as u64 {
                            return fail(token.span, "variables must be in range [1, #vars]");
                        }
                        clauses.push_element(Literal::new(neg, (n - 1) as usize));
                        neg = false;
                    }
                    CNFTokenKind::Neg if !neg => neg = true,
                    CNFTokenKind::Neg => return fail(token.span, "expected a variable"),
                }
            }

            let (input, ()) = it.finish()?;
            let (input, _) = multispace0(input)?;
            let (input, _) = context("expected a literal or '0'", eof)(input)?;

            if clauses.len() != num_clauses {
                // The last clause may or may not be terminated by 0. In case it is, we called
                // `push_clause()` once too often.
                if clauses.len() == num_clauses + 1 && clauses.last().unwrap().is_empty() {
                    clauses.pop_vec();
                } else {
                    return Err(Err::Failure(E::from_external_error(
                        input,
                        ErrorKind::Fail,
                        format!("expected {num_clauses} clauses, got {}", clauses.len()),
                    )));
                }
            }

            Ok((
                input,
                Problem::CNF(Box::new(CNFProblem {
                    vars: preamble.vars,
                    clauses,
                    clause_order_tree: preamble.clause_order_tree,
                })),
            ))
        }
    }
}

mod sat {
    use nom::branch::alt;
    use nom::bytes::complete::tag;
    use nom::character::complete::{char, multispace0, u64};
    use nom::combinator::{consumed, cut, map, recognize};
    use nom::error::{ContextError, ErrorKind, ParseError};
    use nom::sequence::terminated;
    use nom::Err;
    use nom::IResult;

    use crate::util::{fail, map_res_fail, word};
    use crate::{Literal, Problem, Prop, PropProblem, Var, VarSet};

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
        ($matcher:expr , $tok:ident) => {
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
        num_vars: usize,
        allow_xor: bool,
        allow_eq: bool,
    ) -> impl Fn(&'a [u8]) -> IResult<&'a [u8], Prop, SatParserErr<E>> {
        move |input| {
            let (input, tok) = lex(num_vars)(input)?;
            let tok = match tok {
                Some(tok) => tok,
                None => return fail(input, "expected a formula"),
            };

            match tok.kind {
                TokenKind::Var(n) => Ok((input, Prop::Lit(Literal::new(false, n - 1)))),
                TokenKind::Lpar => terminated(
                    cut(formula(num_vars, allow_xor, allow_eq)),
                    expect(TokenKind::Rpar, "expected ')'", num_vars),
                )(input),
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
                        TokenKind::Var(n) => Ok((input, Prop::Lit(Literal::new(true, n - 1)))),
                        TokenKind::Lpar => map(
                            terminated(
                                cut(formula(num_vars, allow_xor, allow_eq)),
                                expect(TokenKind::Rpar, "expected ')'", num_vars),
                            ),
                            |ast| Prop::Neg(Box::new(ast)),
                        )(input),
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
                    let mut children = Vec::new();
                    let input = loop {
                        match formula(num_vars, allow_xor, allow_eq)(input) {
                            Ok((i, sub)) => {
                                input = i;
                                children.push(sub)
                            }
                            Err(Err::Error(SatParserErr::Rpar { input, .. })) => break input,
                            Err(f) => return Err(f),
                        }
                    };
                    Ok((
                        input,
                        match tok.kind {
                            TokenKind::And => Prop::And(children),
                            TokenKind::Or => Prop::Or(children),
                            TokenKind::Xor => Prop::Xor(children),
                            TokenKind::Eq => Prop::Eq(children),
                            _ => unreachable!(),
                        },
                    ))
                }
            }
        }
    }

    pub fn parse<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        vars: VarSet,
        allow_xor: bool,
        allow_eq: bool,
    ) -> impl FnOnce(&'a [u8]) -> IResult<&'a [u8], Problem, E> {
        move |input| match formula(vars.len(), allow_xor, allow_eq)(input) {
            Ok((input, ast)) => Ok((
                input,
                Problem::Prop(Box::new(PropProblem {
                    vars,
                    xor: allow_xor,
                    eq: allow_eq,
                    ast,
                })),
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
    let parse_var_order = options.orders;
    move |input| {
        let (input, preamble) = preamble(parse_var_order)(input)?;
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

    use crate::Prop::{And, Or};
    use crate::{Literal, Prop, Vec2d};

    use super::*;

    // CNF var
    fn cv(v: isize) -> Literal {
        Literal::new(v < 0, v.unsigned_abs() - 1)
    }
    // SAT var
    fn sv(v: isize) -> Prop {
        Prop::Lit(Literal::new(v < 0, v.unsigned_abs() - 1))
    }

    const OPTS_NO_ORDER: ParseOptions = ParseOptions { orders: false };

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
        if let Problem::CNF(cnf) = problem {
            assert_eq!(cnf.vars().len(), 4);
            assert!(cnf.vars.order().is_none());
            assert_eq!(
                cnf.clauses(),
                &Vec2d::from_iter([&[cv(1), cv(3), cv(-4)] as &[_], &[cv(4)], &[cv(2), cv(-3)]])
            );
            assert!(cnf.clause_order().is_none());
        } else {
            panic!()
        }
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
        if let Problem::CNF(cnf) = problem {
            assert_eq!(cnf.vars().len(), 4);
            assert!(cnf.vars().order().is_none());
            assert_eq!(
                cnf.clauses(),
                &Vec2d::from_iter([&[cv(1), cv(3), cv(-4)] as &[_], &[cv(4)], &[cv(2), cv(-3)]])
            );
            assert!(cnf.clause_order().is_none());
        } else {
            panic!()
        }
    }

    #[test]
    fn empty_cnf() {
        let input = "p cnf 0 0\n";
        let (input, problem) = parse::<()>(&OPTS_NO_ORDER)(input.as_bytes())
            .finish()
            .unwrap();
        assert!(input.is_empty());
        if let Problem::CNF(cnf) = problem {
            assert_eq!(cnf.vars().len(), 0);
            assert!(cnf.clauses().is_empty());
            assert!(cnf.clause_order().is_none());
        } else {
            panic!()
        }
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
        if let Problem::Prop(prop) = problem {
            assert_eq!(prop.vars().len(), 4);
            assert!(prop.vars().order().is_none());
            assert!(!prop.xor_allowed());
            assert!(!prop.eq_allowed());
            assert_eq!(
                prop.formula(),
                &And(vec![
                    Or(vec![sv(1), sv(3), sv(-4)]),
                    Or(vec![sv(4)]),
                    Or(vec![sv(2), sv(3)])
                ])
            );
        } else {
            panic!();
        }
    }

    #[test]
    fn preamble_satx() {
        let (input, preamble) = preamble::<()>(false)(b"p satx 1337 \n").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            preamble,
            Preamble {
                format: SATX,
                vars: VarSet::simple(1337),
                num_clauses: 0,
                clause_order_tree: None
            }
        );
    }

    #[test]
    fn preamble_sate() {
        let (input, preamble) = preamble::<()>(false)(b"p sate 1\n").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            preamble,
            Preamble {
                format: SATE,
                vars: VarSet::simple(1),
                num_clauses: 0,
                clause_order_tree: None
            }
        );
    }

    #[test]
    fn preamble_satex() {
        let (input, preamble) = preamble::<()>(false)(b"p satex 42 \n").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            preamble,
            Preamble {
                format: SATEX,
                vars: VarSet::simple(42),
                num_clauses: 0,
                clause_order_tree: None
            }
        );
    }
}
