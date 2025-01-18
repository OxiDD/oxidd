//! Parsing helpers

use std::mem::size_of;
use std::ops::{Range, RangeFrom, RangeTo};

use bitvec::vec::BitVec;
use nom::branch::alt;
use nom::character::complete::{char, line_ending, not_line_ending, space0, space1, u64};
use nom::combinator::{consumed, cut, eof, value};
use nom::error::{ContextError, ErrorKind, FromExternalError, ParseError};
use nom::sequence::preceded;
use nom::{
    AsBytes, AsChar, Compare, Err, IResult, InputIter, InputLength, InputTakeAtPosition, Parser,
    Slice,
};

use crate::Tree;

pub const MAX_CAPACITY: u64 = (usize::MAX / 2 / size_of::<usize>()) as u64;

pub fn usize<I, E>(input: I) -> IResult<I, usize, E>
where
    I: InputIter + Slice<RangeFrom<usize>> + Slice<RangeTo<usize>> + InputLength + AsBytes + Clone,
    I::Item: AsChar,
    E: ParseError<I> + ContextError<I>,
{
    let (remaining, res) = u64(input.clone())?;
    if res > MAX_CAPACITY {
        let span = word_span(input);
        fail(span, "number too large")
    } else {
        Ok((remaining, res as usize))
    }
}

pub fn collect<'a, I, O, E, F>(
    n: usize,
    to: &'a mut Vec<O>,
    mut parser: F,
) -> impl 'a + FnMut(I) -> IResult<I, (), E>
where
    F: 'a + Parser<I, O, E>,
{
    move |mut input| {
        for _ in 0..n {
            let (inp, parsed) = parser.parse(input)?;
            to.push(parsed);
            input = inp;
        }
        Ok((input, ()))
    }
}

pub fn collect_pair<'a, I, O1, O2, E, F>(
    n: usize,
    to1: &'a mut Vec<O1>,
    to2: &'a mut Vec<O2>,
    mut parser: F,
) -> impl 'a + FnMut(I) -> IResult<I, (), E>
where
    F: 'a + Parser<I, (O1, O2), E>,
{
    move |mut input| {
        for _ in 0..n {
            let (inp, (o1, o2)) = parser.parse(input)?;
            to1.push(o1);
            to2.push(o2);
            input = inp;
        }
        Ok((input, ()))
    }
}

/// Optionally space-preceded end of line
pub fn eol<I, E>(input: I) -> IResult<I, (), E>
where
    I: Slice<Range<usize>>
        + Slice<RangeFrom<usize>>
        + Slice<RangeTo<usize>>
        + InputIter
        + InputLength
        + InputTakeAtPosition
        + Compare<&'static str>,
    <I as InputTakeAtPosition>::Item: AsChar + Clone,
    E: ParseError<I>,
{
    preceded(space0, value((), line_ending))(input)
}

/// Optionally space-preceded end of line or file
pub fn eol_or_eof<I, E>(input: I) -> IResult<I, (), E>
where
    I: Slice<Range<usize>>
        + Slice<RangeFrom<usize>>
        + Slice<RangeTo<usize>>
        + InputIter
        + InputLength
        + InputTakeAtPosition
        + Compare<&'static str>
        + Clone,
    <I as InputTakeAtPosition>::Item: AsChar + Clone,
    E: ParseError<I>,
{
    preceded(space0, value((), alt((line_ending, eof))))(input)
}

pub fn word<I, O, E, F>(mut parser: F) -> impl FnMut(I) -> IResult<I, O, E>
where
    I: Clone + Slice<RangeTo<usize>> + AsBytes,
    E: ParseError<I>,
    F: Parser<I, O, E>,
{
    move |i1| {
        let (i2, o) = parser.parse(i1.clone())?;
        match i2.as_bytes().first() {
            Some(c) if !c.is_ascii_alphanumeric() => Ok((i2, o)),
            None => Ok((i2, o)),
            _ => Err(Err::Error(E::from_error_kind(i1, ErrorKind::Fail))),
        }
    }
}

pub fn word_span<I: Slice<RangeTo<usize>> + AsBytes>(input: I) -> I {
    let bytes = input.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b' ' | b'\t' | b'\n' | b'\r' => return input.slice(..i),
            _ => {}
        }
    }
    input
}

#[inline]
pub fn line_span<I: Slice<RangeTo<usize>> + AsBytes>(input: I) -> I {
    match memchr::memchr2(b'\n', b'\r', input.as_bytes()) {
        Some(i) => input.slice(..i),
        None => input,
    }
}

pub fn context_loc<I, E, F, O>(
    span: impl Fn() -> I,
    msg: &'static str,
    mut f: F,
) -> impl FnMut(I) -> IResult<I, O, E>
where
    E: ContextError<I>,
    F: Parser<I, O, E>,
{
    move |input| match f.parse(input) {
        Ok(o) => Ok(o),
        Err(Err::Incomplete(i)) => Err(Err::Incomplete(i)),
        Err(Err::Error(e)) => Err(Err::Error(E::add_context(span(), msg, e))),
        Err(Err::Failure(e)) => Err(Err::Failure(E::add_context(span(), msg, e))),
    }
}

#[inline]
pub fn fail<I: Clone, O, E: ParseError<I> + ContextError<I>>(
    span: I,
    msg: &'static str,
) -> IResult<I, O, E> {
    Err(Err::Failure(E::add_context(
        span.clone(),
        msg,
        E::from_error_kind(span, ErrorKind::Fail),
    )))
}

pub fn fail_with_contexts<I, O, E, It>(ctxs: It) -> IResult<I, O, E>
where
    I: Clone,
    E: ParseError<I> + ContextError<I>,
    It: IntoIterator<Item = (I, &'static str)>,
{
    let mut iter = ctxs.into_iter();
    let (span, msg) = iter.next().expect("At least one context required");
    let mut err = E::add_context(span.clone(), msg, E::from_error_kind(span, ErrorKind::Fail));
    for (span, msg) in iter {
        err = E::add_context(span, msg, err);
    }
    Err(Err::Failure(err))
}

pub fn map_res_fail<I: Clone, O1, O2, E: ParseError<I> + ContextError<I>, F, G>(
    mut parser: F,
    mut f: G,
) -> impl FnMut(I) -> IResult<I, O2, E>
where
    F: Parser<I, O1, E>,
    G: FnMut(O1) -> Result<O2, (I, &'static str)>,
{
    move |input: I| {
        let i = input.clone();
        let (input, o1) = parser.parse(input)?;
        match f(o1) {
            Ok(o2) => Ok((input, o2)),
            Err((span, msg)) => Err(Err::Failure(E::add_context(
                span,
                msg,
                E::from_error_kind(i, ErrorKind::MapOpt),
            ))),
        }
    }
}

/// Remove leading spaces and tabs
pub const fn trim_start(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., b' ' | b'\t'] = s {
        s = rest;
    }
    s
}

/// Remove trailing spaces and tabs
pub const fn trim_end(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., b' ' | b'\t'] = s {
        s = rest;
    }
    s
}

/// Remove leading and trailing spaces and tabs
pub const fn trim(s: &[u8]) -> &[u8] {
    trim_end(trim_start(s))
}

// --- Higher-level parsers used for multiple formats --------------------------

// A comment line starting with 'c'. Consumes everything until the next '\n'
// (inclusive).
pub fn comment<'a, E: ParseError<&'a [u8]>>(input: &'a [u8]) -> IResult<&'a [u8], (), E> {
    let input = char('c')(input)?.0;
    let input = match memchr::memchr(b'\n', input) {
        Some(i) => &input[i + 1..],
        None => &input[input.len()..],
    };
    Ok((input, ()))
}

pub fn var_order_record<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
    input: &'a [u8],
) -> IResult<&'a [u8], ((&'a [u8], u64), Option<&'a [u8]>), E> {
    let (input_before_name, (var_span, var)) = consumed(u64)(input)?;
    let (input, name) = not_line_ending(input_before_name)?;
    let (input, _) = line_ending(input)?;
    let trimmed_name = trim(name);
    let name = if trimmed_name.is_empty() {
        None
    } else if let [b' ' | b'\t', ..] = name {
        Some(trimmed_name)
    } else {
        return Err(Err::Error(E::from_error_kind(
            input_before_name,
            ErrorKind::Space,
        )));
    };
    Ok((input, ((var_span, var), name)))
}

/// Parse a tree with unique `usize` leaves, e.g.: `[0, [2, 3, 1]]`
///
/// All numbers in range `0..=max_number` (or `1..=max_number` if `one_based`
/// is set) are required to occur as leaves. If `unique_leaves` is true, each
/// number must only occur once.
///
/// Returns the tree, as well as the maximal number with its span. If
/// `one_based` is true, all numbers are one less compared to the input.
pub fn tree<'a, E>(
    one_based: bool,
    unique_leaves: bool,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (Tree<usize>, (&'a [u8], usize)), E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    fn rec<'a, E: ParseError<&'a [u8]> + ContextError<&'a [u8]>>(
        input: &'a [u8],
        inserted: &mut BitVec,
        buffer: &mut Vec<Tree<usize>>,
        one_based: bool,
        unique_leaves: bool,
    ) -> IResult<&'a [u8], (Tree<usize>, (&'a [u8], usize)), E> {
        debug_assert!(space1::<_, E>(input).is_err());

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
            } else if unique_leaves && inserted[n] {
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
                let (i, (sub, (sub_max_span, sub_max))) =
                    rec(input, inserted, buffer, one_based, unique_leaves)?;
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
                unique_leaves,
            )
        }))(input)?;
        if let Some(n) = inserted.first_zero() {
            return Err(Err::Failure(E::from_external_error(
                span,
                ErrorKind::Fail,
                format!("number {n} missing in tree"),
            )));
        }
        Ok((input, res))
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{Circuit, Literal, ParseOptions, Problem, ProblemDetails};

    use super::*;

    pub fn unwrap_problem(problem: Problem) -> (Circuit, Literal) {
        match problem {
            Problem {
                circuit,
                details: ProblemDetails::Root(root),
            } => (circuit, root),
            _ => panic!(),
        }
    }

    // Shorthand for an input variable
    pub const fn v(v: usize) -> Literal {
        Literal::from_input(false, v)
    }

    // Shorthand for a positive (i.e., non-negated) gate literal
    pub const fn g(g: usize) -> Literal {
        Literal::from_gate(false, g)
    }

    pub const OPTS_NO_ORDER: ParseOptions = ParseOptions {
        var_order: false,
        clause_tree: false,
        check_acyclic: true,
    };

    #[test]
    fn tree_simple() {
        use Tree::*;
        let (input, (t, max)) = tree::<()>(false, true)(b"[0, [2, 3, 1]]").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            t,
            Inner(Box::from([
                Leaf(0),
                Inner(Box::from([Leaf(2), Leaf(3), Leaf(1)]))
            ]))
        );
        assert_eq!(max.1, 3);
    }

    #[test]
    fn tree_one_based() {
        use Tree::*;
        let (input, (t, max)) = tree::<()>(true, true)(b"[[2, 3, 4], [[1], 5]]").unwrap();
        assert!(input.is_empty());
        assert_eq!(
            t,
            Inner(Box::from([
                Inner(Box::from([Leaf(1), Leaf(2), Leaf(3)])),
                Inner(Box::from([Leaf(0), Leaf(4)]))
            ]))
        );
        assert_eq!(max.1, 4);
    }

    #[test]
    fn tree_err() {
        // violates leaf uniqueness
        assert!(tree::<()>(false, true)(b"[0, 0]").is_err());
        assert!(tree::<()>(false, false)(b"[0, 0]").is_ok());
        // violates "no holes"
        assert!(tree::<()>(false, false)(b"[1]").is_err());
        // 0 in 1-based
        assert!(tree::<()>(true, false)(b"[0]").is_err());
    }
}
