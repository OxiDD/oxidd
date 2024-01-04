use std::ops::RangeTo;

use memchr::memchr2;
use nom::error::{ContextError, ErrorKind, ParseError};
use nom::{AsBytes, Err, IResult, Parser, Slice};

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

pub fn line_span<I: Slice<RangeTo<usize>> + AsBytes>(input: I) -> I {
    match memchr2(b'\n', b'\t', input.as_bytes()) {
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

/// Remove trailing spaces and tabs
pub const fn trim_end(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., b' ' | b'\t'] = s {
        s = rest;
    }
    s
}
