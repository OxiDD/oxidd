//! AIGER parser based on
//! "[The AIGER And-Inverter Graph (AIG) Format Version 20071012][spec1]" and
//! "[AIGER 1.9 And Beyond][spec2]"
//!
//! [spec1]: https://github.com/arminbiere/aiger/blob/master/FORMAT
//! [spec2]: https://fmv.jku.at/papers/BiereHeljankoWieringa-FMV-TR-11-2.pdf

use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::space1;
use nom::combinator::{consumed, eof, map, rest, value};
use nom::error::{ContextError, FromExternalError, ParseError};
use nom::sequence::{preceded, terminated};
use nom::IResult;

use crate::tv_bitvec::TVBitVec;
use crate::util::{
    collect, collect_pair, context_loc, eol, fail, fail_with_contexts, line_span, usize, word,
    word_span,
};
use crate::{Literal, ParseOptions, Var, Vec2d, AIG};

// spell-checker:ignore toposort

#[derive(Clone, Copy)]
struct Header<'a> {
    binary: (&'a [u8], bool),
    vars: (&'a [u8], usize),
    inputs: (&'a [u8], usize),
    latches: (&'a [u8], usize),
    out: (&'a [u8], usize),
    and: (&'a [u8], usize),
    bad: (&'a [u8], usize),
    inv: (&'a [u8], usize),
    just: (&'a [u8], usize),
    fair: (&'a [u8], usize),
}

/// Returns `Ok(.., false)` if `aag` (ASCII), `Ok(.., true)` if `aig` (binary)
fn format<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], bool, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    context_loc(
        || word_span(input),
        "expected 'aag' (ASCII) or 'aig' (binary)",
        word(alt((value(false, tag("aag")), value(true, tag("aig"))))),
    )(input)
}

fn header<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], Header, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    let inner = |input| {
        let (mut input, binary) = consumed(format)(input)?;
        let mandatory = 5; // M I L O A
        let mut numbers: [(&[u8], usize); 9] = [(&[], 0); 9]; // M I L O A B C J F
        for (parsed, num) in numbers.iter_mut().enumerate() {
            (input, *num) = match preceded(space1, consumed(usize))(input) {
                Ok(p) => p,
                Err(e) if parsed < mandatory => return Err(e),
                Err(_) => break,
            };
        }
        let (input, _) = eol(input)?;
        let h = Header {
            binary,
            vars: numbers[0],
            inputs: numbers[1],
            latches: numbers[2],
            out: numbers[3],
            and: numbers[4],
            bad: numbers[5],
            inv: numbers[6],
            just: numbers[7],
            fair: numbers[8],
        };
        let min_vars = h.inputs.1 + h.latches.1 + h.and.1;
        if h.binary.1 {
            if h.vars.1 != min_vars {
                return fail(
                    h.vars.0,
                    "#vars must be equal to #inputs + #latches + #outputs",
                );
            }
        } else if h.vars.1 < min_vars {
            return fail(
                h.vars.0,
                "#vars must be at least #inputs + #latches + #outputs",
            );
        }
        Ok((input, h))
    };

    context_loc(
        || line_span(input),
        "header line must have format 'aag|aig <#vars> <#inputs> <#latches> <#outputs> <#AND gates> [<#bad> [<#invariant constraints> [<#justice> [<#fairness>]]]]'",
        inner,
    )(input)
}

mod ascii {
    use nom::branch::alt;
    use nom::character::complete::{line_ending, not_line_ending, space1, u64};
    use nom::combinator::{consumed, eof, opt};
    use nom::error::{ContextError, ParseError};
    use nom::sequence::preceded;
    use nom::IResult;

    use crate::util::{eol, fail, fail_with_contexts, trim_end};
    use crate::{Literal, AIG};

    use super::Header;

    pub fn literal<'a, E>(
        vars: (&'a [u8], usize),
    ) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (&'a [u8], Literal), E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]>,
    {
        const MSG0: &str = "variable too large. The literal number given here divided by 2 must not be larger than the maximal variable number";
        const MSG1: &str = "note: maximal variable number given here";

        move |input| {
            let (input, (span, lit)) = consumed(u64)(input)?;
            if lit / 2 > vars.1 as u64 {
                return fail_with_contexts([(span, MSG0), (vars.0, MSG1)]);
            }
            Ok((input, (span, Literal(lit as usize))))
        }
    }

    pub fn input_line<'a, E>(
        vars: (&'a [u8], usize),
    ) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (&'a [u8], Literal), E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]>,
    {
        move |input| {
            let (input, lit) = literal(vars)(input)?;
            if lit.1.negative() {
                return fail(
                    lit.0,
                    "inputs mut not be negated (i.e., the number must be even)",
                );
            }
            let (input, _) = eol(input)?;
            Ok((input, lit))
        }
    }

    /// Optional init value, preceded by at least one space if present
    pub fn latch_init_ext<'a, E>(
        latch: Literal,
    ) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], Option<bool>, E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]>,
    {
        const MSG: &str = "initial value must be 0, 1, or the latch literal itself";

        move |input| {
            let (input, init) = opt(preceded(space1, consumed(u64)))(input)?;
            let init = match init {
                None | Some((_, 0)) => Some(false),
                Some((_, 1)) => Some(true),
                Some((_, v)) if v == latch.0 as u64 => None,
                Some((span, _)) => return fail(span, MSG),
            };
            Ok((input, init))
        }
    }

    pub fn latch_line<'a, E>(
        vars: (&'a [u8], usize),
    ) -> impl FnMut(
        &'a [u8],
    )
        -> IResult<&'a [u8], ((&'a [u8], Literal), (&'a [u8], Literal), Option<bool>), E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]>,
    {
        move |input| {
            let (input, lit) = literal(vars)(input)?;
            let (input, _) = space1(input)?;
            let (input, inp) = literal(vars)(input)?;

            if lit.1.negative() {
                return fail(
                    lit.0,
                    "latch literals mut not be negated (i.e., the number must be even)",
                );
            }

            let (input, init) = latch_init_ext(lit.1)(input)?;
            let (input, _) = eol(input)?;
            Ok((input, (lit, inp, init)))
        }
    }

    pub fn symbol_table<'a, 'b, E>(
        header: &'b Header<'a>,
        aig: &'b mut AIG,
    ) -> impl 'b + FnMut(&'a [u8]) -> IResult<&'a [u8], (), E>
    where
        E: ParseError<&'a [u8]> + ContextError<&'a [u8]>,
    {
        const MSGS: [(&str, &str); 7] = [
            ("input not defined", "number of inputs given here"),
            ("latch not defined", "number of latches given here"),
            ("output not defined", "number of outputs given here"),
            (
                "bad state literal not defined",
                "number of bad state literals given here",
            ),
            (
                "invariant constraint not defined",
                "number of invariant constraints given here",
            ),
            (
                "justice property not defined",
                "number of justice properties given here",
            ),
            (
                "fairness constraint not defined",
                "number of fairness constraints given here",
            ),
        ];
        move |mut input| {
            let counts = [
                header.inputs,
                header.latches,
                header.out,
                header.bad,
                header.inv,
                header.just,
                header.fair,
            ];
            const EMPTY_VEC: Vec<Option<String>> = Vec::new();
            let mut symbols = [EMPTY_VEC; 7];
            loop {
                let (inp, kind) = match input {
                    [b'i', inp @ ..] => (inp, 0),
                    [b'l', inp @ ..] => (inp, 1),
                    [b'o', inp @ ..] => (inp, 2),
                    [b'b', inp @ ..] => (inp, 3),
                    [b'c', inp @ ..] => match inp {
                        [b'0'..=b'9', ..] => (inp, 4),
                        _ => break, // c is also used to begin the comment section
                    },
                    [b'j', inp @ ..] => (inp, 5),
                    [b'f', inp @ ..] => (inp, 6),
                    _ => break,
                };

                let (inp, (span, i)) = consumed(u64)(inp)?;
                let (count_span, count) = counts[kind];
                if i >= count as u64 {
                    return fail_with_contexts([(span, MSGS[kind].0), (count_span, MSGS[kind].1)]);
                }

                let (inp, _) = space1(inp)?;
                let (inp, name) = not_line_ending(inp)?;
                (input, _) = alt((line_ending, eof))(inp)?;

                let symbol_list = &mut symbols[kind];
                if symbol_list.is_empty() {
                    symbol_list.resize(count, None);
                }
                let symbol = &mut symbol_list[i as usize];
                let name = String::from_utf8_lossy(trim_end(name));
                if let Some(symbol) = symbol {
                    symbol.reserve(1 + name.len());
                    symbol.push(' ');
                    symbol.push_str(&name);
                } else {
                    *symbol = Some(name.to_string());
                }
            }

            [
                aig.input_names,
                aig.latch_names,
                aig.output_names,
                aig.bad_names,
                aig.invariant_names,
                aig.justice_names,
                aig.fairness_names,
            ] = symbols;

            Ok((input, ()))
        }
    }
}

fn usize_7bit(input: &[u8]) -> Result<(&[u8], usize), ()> {
    let mut input = input;
    let mut val = 0;
    let mut shift = 0u32;
    loop {
        let &[b, ref rem @ ..] = input else {
            return Err(());
        };
        input = rem;

        val |= ((b & ((1 << 7) - 1)) as usize).wrapping_shl(shift);
        if b & (1 << 7) == 0 {
            return Ok((input, val));
        }
        shift += 7;
    }
}

/// Parse a (binary or ASCII) AIGER file
pub fn parse<'a, E>(
    options: &ParseOptions,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], Box<AIG>, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    let _ = options; // may be used later
    move |input| {
        let (mut input, h) = header(input)?;

        let mut aig = Box::new(AIG {
            inputs: h.inputs.1,
            latches: Vec::with_capacity(h.latches.1),
            latch_init_values: TVBitVec::with_capacity(h.latches.1),
            and_gates: Vec::with_capacity(h.and.1),
            outputs: Vec::with_capacity(h.out.1),
            bad: Vec::with_capacity(h.bad.1),
            invariants: Vec::with_capacity(h.inv.1),
            justice: Vec2d::with_capacity(h.just.1, 0),
            fairness: Vec::with_capacity(h.fair.1),

            input_names: Vec::new(),
            latch_names: Vec::new(),
            output_names: Vec::new(),
            bad_names: Vec::new(),
            invariant_names: Vec::new(),
            justice_names: Vec::new(),
            fairness_names: Vec::new(),
        });

        let mut justice_len = Vec::<usize>::with_capacity(h.just.1);

        let first_input = 1;
        let first_latch = first_input + h.inputs.1;
        let first_and_gate = first_latch + h.latches.1;
        let var_count = first_and_gate + h.and.1; // reduced, including false

        if h.binary.1 {
            // latches
            for i in 0..h.latches.1 {
                let (inp, (_, lit)) = ascii::literal(h.vars)(input)?;
                let (inp, init) = ascii::latch_init_ext(Literal(i + first_latch * 2))(inp)?;
                input = eol(inp)?.0;
                aig.latches.push(lit);
                aig.latch_init_values.push(init);
            }

            // outputs etc.
            let mut literal = map(terminated(ascii::literal(h.vars), eol), |(_, l)| l);
            input = collect(h.out.1, &mut aig.outputs, &mut literal)(input)?.0;
            input = collect(h.bad.1, &mut aig.bad, &mut literal)(input)?.0;
            input = collect(h.inv.1, &mut aig.invariants, &mut literal)(input)?.0;
            input = collect(h.just.1, &mut justice_len, terminated(usize, eol))(input)?.0;
            aig.justice.reserve_elements(justice_len.iter().sum());
            for &n in &justice_len {
                aig.justice.push_vec();
                for _ in 0..n {
                    let (inp, (_, lit)) = ascii::literal(h.vars)(input)?;
                    (input, _) = eol(inp)?;
                    aig.justice.push_element(lit);
                }
            }
            input = collect(h.fair.1, &mut aig.fairness, &mut literal)(input)?.0;

            // and gates
            let and_gate_eof_err = fail(
                h.binary.0,
                "invalid binary: reached end of file while parsing an and gate",
            );
            for i in first_and_gate..var_count {
                let Ok((inp, d1)) = usize_7bit(input) else {
                    return and_gate_eof_err;
                };
                let Ok((inp, d2)) = usize_7bit(inp) else {
                    return and_gate_eof_err;
                };
                input = inp;
                let lhs = i * 2;
                let in1 = lhs.wrapping_sub(d1);
                if d1 > lhs || d1 == 0 || d2 > in1 {
                    return fail(h.binary.0, "invalid binary: invalid and gate inputs");
                }
                let in2 = in1 - d2;
                aig.and_gates.push((Literal(in1), Literal(in2)));
            }
        } else {
            const UNDEF: Var = usize::MAX;
            let mut var_map = vec![UNDEF; h.vars.1 + 1];
            var_map[0] = 0;
            let mut latch_input_spans = Vec::with_capacity(h.latches.1);
            let mut out_spans = Vec::with_capacity(h.out.1);
            let mut bad_spans = Vec::with_capacity(h.bad.1);
            let mut inv_spans = Vec::with_capacity(h.inv.1);
            let mut just_spans = Vec::new();
            let mut fair_spans = Vec::with_capacity(h.fair.1);

            let mut and_gate_spans = Vec::with_capacity(h.and.1);

            // inputs
            const SECOND_DEF_MSG: &str = "second variable definition";
            for i in first_input..first_latch {
                let (inp, (span, lit)) = ascii::input_line(h.vars)(input)?;
                let var = lit.variable();
                if var_map[var] != UNDEF {
                    return fail(span, SECOND_DEF_MSG);
                }
                var_map[var] = i;
                input = inp;
            }

            // latches
            for i in 0..h.latches.1 {
                let tmp = ascii::latch_line(h.vars)(input)?;
                input = tmp.0;
                let ((lit_span, lit), (inp_span, inp), init) = tmp.1;
                let var = lit.variable();
                if var_map[var] != UNDEF {
                    return fail(lit_span, SECOND_DEF_MSG);
                }
                var_map[var] = i + first_latch;
                aig.latches.push(inp);
                latch_input_spans.push(inp_span);
                aig.latch_init_values.push(init);
            }

            // outputs etc.
            let mut literal = terminated(ascii::literal(h.vars), eol);
            input = collect_pair(h.out.1, &mut out_spans, &mut aig.outputs, &mut literal)(input)?.0;
            input = collect_pair(h.bad.1, &mut bad_spans, &mut aig.bad, &mut literal)(input)?.0;
            input =
                collect_pair(h.inv.1, &mut inv_spans, &mut aig.invariants, &mut literal)(input)?.0;
            input = collect(h.just.1, &mut justice_len, terminated(usize, eol))(input)?.0;

            // justice
            let justice_elements = justice_len.iter().sum();
            aig.justice.reserve_elements(justice_elements);
            just_spans.reserve(justice_elements);
            for &n in &justice_len {
                aig.justice.push_vec();
                for _ in 0..n {
                    let (inp, (span, lit)) = ascii::literal(h.vars)(input)?;
                    (input, _) = eol(inp)?;
                    just_spans.push(span);
                    aig.justice.push_element(lit);
                }
            }

            // fairness
            input =
                collect_pair(h.fair.1, &mut fair_spans, &mut aig.fairness, &mut literal)(input)?.0;

            // and gates
            for i in first_and_gate..var_count {
                let (inp, (lit_span, lit)) = ascii::literal(h.vars)(input)?;
                let (inp, _) = space1(inp)?;
                let (inp, (in1_span, in1)) = ascii::literal(h.vars)(inp)?;
                let (inp, _) = space1(inp)?;
                let (inp, (in2_span, in2)) = ascii::literal(h.vars)(inp)?;
                input = eol(inp)?.0;

                if lit.negative() {
                    return fail(lit_span, "gate literal must not be negative");
                }
                let var = lit.variable();
                if var_map[var] != UNDEF {
                    return fail(lit_span, SECOND_DEF_MSG);
                }
                var_map[var] = i;
                aig.and_gates.push((in1, in2));
                and_gate_spans.push((lit_span, in1_span, in2_span));
            }

            // map literals
            // `map` does not capture `var_map` but takes it as an argument such that we can
            // reuse `map` without into lifetime issues.
            let map = move |var_map: &[Var], l: Literal| Literal((var_map[l.0 / 2] * 2) | l.0 & 1);
            let map_slice = move |var_map: &[Var],
                                  slice: &mut [Literal],
                                  spans: &[&'a [u8]],
                                  undef: &mut Vec<&'a [u8]>| {
                for (i, inp) in slice.iter_mut().enumerate() {
                    *inp = map(var_map, *inp);
                    if inp.0 == usize::MAX {
                        undef.push(spans[i]);
                    }
                }
            };
            let mut undef = Vec::new();
            map_slice(&var_map, &mut aig.latches, &latch_input_spans, &mut undef);
            map_slice(&var_map, &mut aig.outputs, &out_spans, &mut undef);
            map_slice(&var_map, &mut aig.bad, &bad_spans, &mut undef);
            map_slice(&var_map, &mut aig.invariants, &inv_spans, &mut undef);
            let justice_elements = aig.justice.all_elements_mut();
            map_slice(&var_map, justice_elements, &just_spans, &mut undef);
            map_slice(&var_map, &mut aig.fairness, &fair_spans, &mut undef);
            let mut sorted = true;
            for (i, (in1, in2)) in aig.and_gates.iter_mut().enumerate() {
                *in1 = map(&var_map, *in1);
                if in1.0 == usize::MAX {
                    undef.push(and_gate_spans[i].1);
                }
                *in2 = map(&var_map, *in2);
                if in2.0 == usize::MAX {
                    undef.push(and_gate_spans[i].2);
                }
                if in1 < in2 {
                    std::mem::swap(in1, in2);
                }
                sorted &= (i + first_and_gate) * 2 > in1.0;
            }
            if !undef.is_empty() {
                return fail_with_contexts(undef.iter().map(|&span| (span, "undefined literal")));
            }

            // check for cycles
            if !sorted {
                let mut target = first_and_gate;
                let mut visited = BitVec::repeat(false, aig.and_gates.len() * 2);
                for (i, (span, _, _)) in and_gate_spans.iter().enumerate() {
                    if !toposort(
                        i,
                        &aig.and_gates,
                        first_and_gate,
                        &mut visited,
                        &mut var_map[first_and_gate..var_count],
                        &mut target,
                    ) {
                        return fail(*span, "and gate depends on itself");
                    }
                }

                // normalize the graph
                let map_slice = move |var_map: &[Var], slice: &mut [Literal]| {
                    for i in slice {
                        if i.variable() >= first_and_gate {
                            debug_assert!(i.variable() < var_count);
                            *i = map(var_map, *i);
                        }
                    }
                };
                map_slice(&var_map, &mut aig.latches);
                map_slice(&var_map, &mut aig.outputs);
                map_slice(&var_map, &mut aig.bad);
                map_slice(&var_map, &mut aig.invariants);
                map_slice(&var_map, aig.justice.all_elements_mut());
                map_slice(&var_map, &mut aig.fairness);

                let mut mapped_and_gates = vec![(Literal(0), Literal(0)); aig.and_gates.len()];
                for (&i, &(mut in1, mut in2)) in var_map[first_and_gate..var_count]
                    .iter()
                    .zip(&aig.and_gates)
                {
                    if in1.variable() >= first_and_gate {
                        debug_assert!(in1.variable() < var_count);
                        in1 = map(&var_map, in1);
                    }
                    if in2.variable() >= first_and_gate {
                        debug_assert!(in2.variable() < var_count);
                        in2 = map(&var_map, in2);
                    }
                    mapped_and_gates[i - first_and_gate] =
                        if in1 >= in2 { (in1, in2) } else { (in2, in1) };
                }
                drop(std::mem::replace(&mut aig.and_gates, mapped_and_gates));
            }
        }

        // symbol table
        let (input, ()) = ascii::symbol_table(&h, &mut aig)(input)?;

        // optional comment section
        let (input, _) = alt((preceded(tag("c"), rest), eof))(input)?;

        Ok((input, aig))
    }
}

/// `current` is the and gate number
fn toposort(
    current: usize,
    and_gates: &[(Literal, Literal)],
    first_and_gate: Var,
    visited: &mut BitSlice,
    map: &mut [Var],
    target: &mut Var,
) -> bool {
    if visited[current * 2 + 1] {
        return true; // finished
    }
    if visited[current * 2] {
        return false; // discovered -> cycle
    }
    visited.set(current * 2, true); // discovered

    let (l1, l2) = and_gates[current];
    let (v1, v2) = (l1.variable(), l2.variable());
    let first = first_and_gate; // just a shorthand
    if v2 >= first && !toposort(v2 - first, and_gates, first_and_gate, visited, map, target) {
        return false;
    }
    if v1 >= first && !toposort(v1 - first, and_gates, first_and_gate, visited, map, target) {
        return false;
    }

    map[current] = *target;
    *target += 1;
    visited.set(current * 2 + 1, true); // finished
    true
}

#[cfg(test)]
mod tests {
    use nom::error::VerboseError;

    use crate::tv_bitvec::TVBitVec;
    use crate::{Literal, ParseOptionsBuilder, Vec2d};

    use super::{usize_7bit, AIG};

    impl AIG {
        fn empty() -> Self {
            Self {
                inputs: 0,
                latches: Vec::new(),
                latch_init_values: TVBitVec::new(),
                and_gates: Vec::new(),
                outputs: Vec::new(),
                bad: Vec::new(),
                invariants: Vec::new(),
                justice: Vec2d::new(),
                fairness: Vec::new(),

                input_names: Vec::new(),
                latch_names: Vec::new(),
                output_names: Vec::new(),
                bad_names: Vec::new(),
                invariant_names: Vec::new(),
                justice_names: Vec::new(),
                fairness_names: Vec::new(),
            }
        }

        fn with_inputs(mut self, inputs: usize) -> Self {
            self.inputs = inputs;
            self
        }

        fn with_named_inputs<'a>(mut self, inputs: impl IntoIterator<Item = &'a str>) -> Self {
            self.input_names = inputs.into_iter().map(|s| Some(s.to_string())).collect();
            self.inputs = self.input_names.len();
            self
        }

        fn with_outputs(mut self, outputs: impl IntoIterator<Item = Literal>) -> Self {
            self.outputs = outputs.into_iter().collect();
            self
        }

        fn with_named_outputs<'a>(
            mut self,
            outputs: impl IntoIterator<Item = (Literal, &'a str)>,
        ) -> Self {
            let it = outputs.into_iter();
            self.outputs = Vec::with_capacity(it.size_hint().0);
            self.output_names = Vec::with_capacity(it.size_hint().0);
            for (literal, name) in it {
                self.outputs.push(literal);
                self.output_names.push(Some(name.to_string()));
            }
            self
        }

        fn with_and_gates(
            mut self,
            and_gates: impl IntoIterator<Item = (Literal, Literal)>,
        ) -> Self {
            self.and_gates = and_gates.into_iter().collect();
            self
        }

        fn with_latches(mut self, latches: impl IntoIterator<Item = Literal>) -> Self {
            self.latches = latches.into_iter().collect();
            self.latch_init_values = TVBitVec::with_capacity(2 * self.latches.len());
            for _ in 0..self.latches.len() {
                self.latch_init_values.push(Some(false));
            }
            self
        }
    }

    #[test]
    fn decode_7bit() {
        // cases taken from https://github.com/arminbiere/aiger/blob/master/FORMAT
        let cases: &[(usize, &[u8])] = &[
            (0, b"\x00"),
            (1, b"\x01"),
            ((1 << 7) - 1, b"\x7f"),
            ((1 << 7), b"\x80\x01"),
            ((1 << 8) + 2, b"\x82\x02"),
            ((1 << 14) - 1, b"\xff\x7f"),
            ((1 << 14) + 3, b"\x83\x80\x01"),
            ((1 << 28) - 1, b"\xff\xff\xff\x7f"),
            ((1 << 28) + 7, b"\x87\x80\x80\x80\x01"),
        ];

        for &(expected, input) in cases {
            let (remaining, val) = usize_7bit(input).unwrap();
            assert!(remaining.is_empty());
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn aag() {
        // cases taken from https://github.com/arminbiere/aiger/blob/master/FORMAT
        let cases: &[(&[u8], &[u8], AIG)] = &[
            (b"aag 0 0 0 0 0\n", b"aig 0 0 0 0 0\n", AIG::empty()),
            (
                b"aag 0 0 0 1 0\n0\n",
                b"aig 0 0 0 1 0\n0\n",
                AIG::empty().with_outputs([Literal(0)]),
            ),
            (
                b"aag 0 0 0 1 0\n1\n",
                b"aig 0 0 0 1 0\n1\n",
                AIG::empty().with_outputs([Literal(1)]),
            ),
            (
                b"aag 1 1 0 1 0\n2\n2\n",
                b"aig 1 1 0 1 0\n2\n",
                AIG::empty().with_inputs(1).with_outputs([Literal(2)]),
            ),
            (
                b"aag 1 1 0 1 0\n2\n3\n",
                b"aig 1 1 0 1 0\n3\n",
                AIG::empty().with_inputs(1).with_outputs([Literal(3)]),
            ),
            (
                b"aag 3 2 0 1 1\n2\n4\n6\n6 2 4\n",
                b"aig 3 2 0 1 1\n6\n\x02\x02",
                AIG::empty()
                    .with_inputs(2)
                    .with_outputs([Literal(6)])
                    .with_and_gates([(Literal(4), Literal(2))]),
            ),
            (
                b"aag 3 2 0 1 1\n2\n4\n7\n6 3 5\n",
                b"aig 3 2 0 1 1\n7\n\x01\x02",
                AIG::empty()
                    .with_inputs(2)
                    .with_outputs([Literal(7)])
                    .with_and_gates([(Literal(5), Literal(3))]),
            ),
            (
                b"aag 7 2 0 2 3\n\
                2\n\
                4\n\
                6\n\
                12\n\
                6 13 15\n\
                12 2 4\n\
                14 3 5\n\
                i0 x\n\
                i1 y\n\
                o0 s\n\
                o1 c\n\
                c\nhalf adder\n",
                b"aig 5 2 0 2 3\n\
                10\n\
                6\n\
                \x02\x02\
                \x03\x02\
                \x01\x02\
                i0 x\n\
                i1 y\n\
                o0 s\n\
                o1 c\n\
                c\nhalf adder\n",
                AIG::empty()
                    .with_named_inputs(["x", "y"])
                    .with_named_outputs([(Literal(10), "s"), (Literal(6), "c")])
                    .with_and_gates([
                        (Literal(4), Literal(2)), // could be swapped with the line below
                        (Literal(5), Literal(3)),
                        (Literal(9), Literal(7)),
                    ]),
            ),
            (
                b"aag 1 0 1 2 0\n2 3\n2\n3\n",
                b"aig 1 0 1 2 0\n3\n2\n3\n",
                AIG::empty()
                    .with_latches([Literal(3)])
                    .with_outputs([Literal(2), Literal(3)]),
            ),
            (
                b"aag 7 2 1 2 4\n\
                2\n\
                4\n\
                6 8\n\
                6\n\
                7\n\
                8 4 10\n\
                10 13 15\n\
                12 2 6\n\
                14 3 7\n",
                b"aig 7 2 1 2 4\n\
                14\n\
                6\n\
                7\n\
                \x02\x04\
                \x03\x04\
                \x01\x02\
                \x02\x08",
                AIG::empty()
                    .with_inputs(2)
                    .with_latches([Literal(14)])
                    .with_outputs([Literal(6), Literal(7)])
                    .with_and_gates([
                        (Literal(6), Literal(2)),
                        (Literal(7), Literal(3)),
                        (Literal(11), Literal(9)),
                        (Literal(12), Literal(4)),
                    ]),
            ),
        ];

        let options = ParseOptionsBuilder::default().build().unwrap();

        for &(ascii, bin, ref expected) in cases {
            let (_, aig) = super::parse::<()>(&options)(ascii).unwrap();
            assert_eq!(&*aig, expected);
            let (_, aig) = super::parse::<VerboseError<&[u8]>>(&options)(bin).unwrap();
            assert_eq!(&*aig, expected);
        }
    }
}
