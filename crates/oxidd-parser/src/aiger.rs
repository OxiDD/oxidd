//! AIGER parser based on
//! "[The AIGER And-Inverter Graph (AIG) Format Version 20071012][spec1]" and
//! "[AIGER 1.9 And Beyond][spec2]"
//!
//! [spec1]: https://github.com/arminbiere/aiger/blob/master/FORMAT
//! [spec2]: https://fmv.jku.at/papers/BiereHeljankoWieringa-FMV-TR-11-2.pdf

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::space1;
use nom::combinator::{consumed, eof, map, rest, value};
use nom::error::{ContextError, FromExternalError, ParseError};
use nom::sequence::{preceded, terminated};
use nom::IResult;

use crate::tv_bitvec::TVBitVec;
use crate::util::{
    collect, collect_pair, context_loc, eol_or_eof, fail, fail_with_contexts, line_span, usize,
    word, word_span,
};
use crate::{
    AIGERDetails, Circuit, GateKind, Literal, ParseOptions, Problem, ProblemDetails, Var, VarSet,
    Vec2d,
};

// spell-checker:ignore toposort

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AIGLiteral(usize);

impl AIGLiteral {
    #[inline(always)]
    fn negative(self) -> bool {
        self.0 & 1 != 0
    }

    #[inline(always)]
    fn variable(self) -> Var {
        self.0 >> 1
    }
}

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

fn header<'a, E>(input: &'a [u8]) -> IResult<&'a [u8], Header<'a>, E>
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
        let (input, _) = eol_or_eof(input)?;
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

    use crate::util::{eol_or_eof, fail, fail_with_contexts, trim_end};
    use crate::{AIGERDetails, VarSet};

    use super::{AIGLiteral, Header};

    pub fn literal<'a, E>(
        vars: (&'a [u8], usize),
    ) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (&'a [u8], AIGLiteral), E>
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
            Ok((input, (span, AIGLiteral(lit as usize))))
        }
    }

    pub fn input_line<'a, E>(
        vars: (&'a [u8], usize),
    ) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (&'a [u8], AIGLiteral), E>
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
            let (input, _) = eol_or_eof(input)?;
            Ok((input, lit))
        }
    }

    /// Optional init value, preceded by at least one space if present
    pub fn latch_init_ext<'a, E>(
        latch: AIGLiteral,
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
    ) -> IResult<
        &'a [u8],
        ((&'a [u8], AIGLiteral), (&'a [u8], AIGLiteral), Option<bool>),
        E,
    >
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
            let (input, _) = eol_or_eof(input)?;
            Ok((input, (lit, inp, init)))
        }
    }

    pub fn symbol_table<'a, 'b, E>(
        header: &'b Header<'a>,
        inputs: &'b mut VarSet,
        aig: &'b mut AIGERDetails,
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
                header.out,
                header.bad,
                header.inv,
                header.just,
                header.fair,
                header.latches,
            ];
            let mut symbols = [const { Vec::<Option<String>>::new() }; 6];
            loop {
                let (inp, mut kind) = match input {
                    [b'i', inp @ ..] => (inp, 0),
                    [b'o', inp @ ..] => (inp, 1),
                    [b'b', inp @ ..] => (inp, 2),
                    [b'c', inp @ ..] => match inp {
                        [b'0'..=b'9', ..] => (inp, 3),
                        _ => break, // c is also used to begin the comment section
                    },
                    [b'j', inp @ ..] => (inp, 4),
                    [b'f', inp @ ..] => (inp, 5),
                    [b'l', inp @ ..] => (inp, 6),
                    _ => break,
                };

                let (inp, (span, i)) = consumed(u64)(inp)?;
                let (count_span, mut count) = counts[kind];
                if i >= count as u64 {
                    return fail_with_contexts([(span, MSGS[kind].0), (count_span, MSGS[kind].1)]);
                }
                let mut i = i as usize;

                if kind == 6 {
                    kind = 0;
                    i += header.inputs.1;
                    count += header.inputs.1;
                } else if kind == 0 {
                    count += header.latches.1;
                }

                let (inp, _) = space1(inp)?;
                let (inp, name) = not_line_ending(inp)?;
                (input, _) = alt((line_ending, eof))(inp)?;

                let symbol_list = &mut symbols[kind];
                if symbol_list.is_empty() {
                    symbol_list.resize(count, None);
                }
                let symbol = &mut symbol_list[i];
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
                inputs.names,
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
pub fn parse<'a, E>(options: &ParseOptions) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], Problem, E>
where
    E: ParseError<&'a [u8]> + ContextError<&'a [u8]> + FromExternalError<&'a [u8], String>,
{
    let check_acyclic = options.check_acyclic;
    move |input| {
        let (mut input, h) = header(input)?;

        let mut circuit = Circuit::new(VarSet::new(h.inputs.1 + h.latches.1));
        circuit.reserve_gates(h.and.1);
        circuit.reserve_gate_inputs(h.and.1 * 2);
        let mut aig = Box::new(AIGERDetails {
            inputs: h.inputs.1,
            latches: Vec::with_capacity(h.latches.1),
            latch_init_values: TVBitVec::with_capacity(h.latches.1),
            outputs: Vec::with_capacity(h.out.1),
            bad: Vec::with_capacity(h.bad.1),
            invariants: Vec::with_capacity(h.inv.1),
            justice: Vec2d::with_capacity(h.just.1, 0),
            fairness: Vec::with_capacity(h.fair.1),

            map: Vec::new(),

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
            let make_literal = move |aig_literal: AIGLiteral| {
                let var = aig_literal.variable();
                let negative = aig_literal.negative();
                if var >= first_and_gate {
                    Literal::from_gate(negative, var - first_and_gate)
                } else {
                    Literal::from_input_or_false(negative, var)
                }
            };

            // latches
            for i in 0..h.latches.1 {
                let (inp, (_, lit)) = ascii::literal(h.vars)(input)?;
                let (inp, init) = ascii::latch_init_ext(AIGLiteral(i + first_latch * 2))(inp)?;
                input = eol_or_eof(inp)?.0;
                aig.latches.push(make_literal(lit));
                aig.latch_init_values.push(init);
            }

            // outputs etc.
            let mut literal = map(
                terminated(ascii::literal(h.vars), eol_or_eof),
                move |(_, l)| make_literal(l),
            );
            input = collect(h.out.1, &mut aig.outputs, &mut literal)(input)?.0;
            input = collect(h.bad.1, &mut aig.bad, &mut literal)(input)?.0;
            input = collect(h.inv.1, &mut aig.invariants, &mut literal)(input)?.0;
            input = collect(h.just.1, &mut justice_len, terminated(usize, eol_or_eof))(input)?.0;
            aig.justice.reserve_elements(justice_len.iter().sum());
            for &n in &justice_len {
                aig.justice.push_vec();
                for _ in 0..n {
                    let (inp, (_, lit)) = ascii::literal(h.vars)(input)?;
                    (input, _) = eol_or_eof(inp)?;
                    aig.justice.push_element(make_literal(lit));
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
                circuit.push_gate(GateKind::And);
                circuit.push_gate_inputs([
                    make_literal(AIGLiteral(in1)),
                    make_literal(AIGLiteral(in2)),
                ]);
            }
        } else {
            // There may be undefined literals in the ASCII format (see the tests below).
            // Since the numeric values of the `Literal` type do not match the `AIGLiteral`
            // values anyway, we just map the literals into a contiguous space. `aig.map`
            // maps variables to non-negated literals.
            // Due to possible forward-references, we can only map the `AIGLiteral`s to
            // `Literal`s once all definitions are read. Hence, we write intermediate values
            // (`Literal(_.0)`) to the respective fields of `aig`.
            aig.map = vec![Literal::UNDEF; h.vars.1 + 1];
            aig.map[0] = Literal::FALSE;

            // inputs
            const SECOND_DEF_MSG: &str = "second variable definition";
            for i in first_input..first_latch {
                let (inp, (span, lit)) = ascii::input_line(h.vars)(input)?;
                let var = lit.variable();
                if aig.map[var] != Literal::UNDEF {
                    return fail(span, SECOND_DEF_MSG);
                }
                aig.map[var] = Literal::from_input_or_false(false, i);
                input = inp;
            }

            // latches
            let mut latch_input_spans = Vec::with_capacity(h.latches.1);
            for i in first_latch..first_and_gate {
                let tmp = ascii::latch_line(h.vars)(input)?;
                input = tmp.0;
                let ((lit_span, lit), (inp_span, inp), init) = tmp.1;
                let var = lit.variable();
                if aig.map[var] != Literal::UNDEF {
                    return fail(lit_span, SECOND_DEF_MSG);
                }
                aig.map[var] = Literal::from_input_or_false(false, i);
                aig.latches.push(Literal(inp.0));
                latch_input_spans.push(inp_span);
                aig.latch_init_values.push(init);
            }

            // outputs etc.
            let mut out_spans = Vec::with_capacity(h.out.1);
            let mut bad_spans = Vec::with_capacity(h.bad.1);
            let mut inv_spans = Vec::with_capacity(h.inv.1);
            let mut literal = map(
                terminated(ascii::literal(h.vars), eol_or_eof),
                move |(s, l)| (s, Literal(l.0)),
            );
            input = collect_pair(h.out.1, &mut out_spans, &mut aig.outputs, &mut literal)(input)?.0;
            input = collect_pair(h.bad.1, &mut bad_spans, &mut aig.bad, &mut literal)(input)?.0;
            input =
                collect_pair(h.inv.1, &mut inv_spans, &mut aig.invariants, &mut literal)(input)?.0;
            input = collect(h.just.1, &mut justice_len, terminated(usize, eol_or_eof))(input)?.0;

            // justice
            let justice_elements = justice_len.iter().sum();
            aig.justice.reserve_elements(justice_elements);
            let mut just_spans = Vec::with_capacity(justice_elements);
            for &n in &justice_len {
                aig.justice.push_vec();
                for _ in 0..n {
                    let (inp, (span, lit)) = ascii::literal(h.vars)(input)?;
                    (input, _) = eol_or_eof(inp)?;
                    just_spans.push(span);
                    aig.justice.push_element(Literal(lit.0));
                }
            }

            // fairness
            let mut fair_spans = Vec::with_capacity(h.fair.1);
            input =
                collect_pair(h.fair.1, &mut fair_spans, &mut aig.fairness, &mut literal)(input)?.0;

            // and gates
            let mut and_gate_spans = Vec::with_capacity(h.and.1);
            for i in 0..h.and.1 {
                let (inp, (lit_span, lit)) = ascii::literal(h.vars)(input)?;
                let (inp, _) = space1(inp)?;
                let (inp, (in1_span, in1)) = ascii::literal(h.vars)(inp)?;
                let (inp, _) = space1(inp)?;
                let (inp, (in2_span, in2)) = ascii::literal(h.vars)(inp)?;
                input = eol_or_eof(inp)?.0;

                if lit.negative() {
                    return fail(lit_span, "gate literal must not be negative");
                }
                let var = lit.variable();
                if aig.map[var] != Literal::UNDEF {
                    return fail(lit_span, SECOND_DEF_MSG);
                }
                aig.map[var] = Literal::from_gate(false, i);
                circuit.push_gate(GateKind::And);
                circuit.push_gate_inputs([Literal(in1.0), Literal(in2.0)]);
                and_gate_spans.push((lit_span, in1_span, in2_span));
            }

            // map literals
            let mut undef = Vec::new();
            let mut map = |l: Literal, span: &'a [u8]| {
                let mapped_literal = aig.map[l.0 >> 1];
                if mapped_literal == Literal::UNDEF {
                    undef.push(span);
                }
                Literal(mapped_literal.0 | ((l.0 & 1) << Literal::POLARITY_BIT))
            };
            let mut map_slice = |slice: &mut [Literal], spans: &[&'a [u8]]| {
                for (inp, &span) in slice.iter_mut().zip(spans.iter()) {
                    *inp = map(*inp, span);
                }
            };
            map_slice(&mut aig.latches, &latch_input_spans);
            map_slice(&mut aig.outputs, &out_spans);
            map_slice(&mut aig.bad, &bad_spans);
            map_slice(&mut aig.invariants, &inv_spans);
            let justice_elements = aig.justice.all_elements_mut();
            map_slice(justice_elements, &just_spans);
            map_slice(&mut aig.fairness, &fair_spans);
            #[allow(clippy::needless_range_loop)]
            // there is no iter_mut on Circuit (probably, we cannot implement that in Safe Rust)
            for i in 0..circuit.num_gates() {
                if let Some([in1, in2]) = circuit.gate_inputs_mut_for_no(i) {
                    *in1 = map(*in1, and_gate_spans[i].1);
                    *in2 = map(*in2, and_gate_spans[i].2);
                } else {
                    debug_assert!(false);
                }
            }
            if !undef.is_empty() {
                return fail_with_contexts(undef.iter().map(|&span| (span, "undefined literal")));
            }

            // check for cycles
            if check_acyclic {
                if let Some(l) = circuit.find_cycle() {
                    return fail(
                        and_gate_spans[l.get_gate_no().unwrap()].0,
                        "and gate depends on itself",
                    );
                }
            }
        }

        // symbol table
        let (input, ()) = ascii::symbol_table(&h, &mut circuit.inputs, &mut aig)(input)?;

        // optional comment section
        let (input, _) = alt((preceded(tag("c"), rest), eof))(input)?;

        if h.binary.1 {
            // collect he map for binary mode at the very end since it cannot fail
            debug_assert!(aig.map.is_empty());
            aig.map.reserve(var_count);
            aig.map
                .extend((0..first_and_gate).map(|i| Literal::from_input_or_false(false, i)));
            aig.map
                .extend((0..h.and.1).map(|i| Literal::from_gate(false, i)));
        }

        let problem = Problem {
            circuit,
            details: ProblemDetails::AIGER(aig),
        };

        Ok((input, problem))
    }
}

#[cfg(test)]
mod tests {
    use crate::tv_bitvec::TVBitVec;
    use crate::util::test::OPTS_NO_ORDER;
    use crate::{Circuit, Literal, Problem, ProblemDetails, VarSet, Vec2d};

    use super::{usize_7bit, AIGERDetails};

    impl Problem {
        fn new_aig(vars: VarSet) -> Self {
            let details = Box::new(AIGERDetails {
                inputs: vars.len(),
                latches: Vec::new(),
                latch_init_values: TVBitVec::new(),
                outputs: Vec::new(),
                bad: Vec::new(),
                invariants: Vec::new(),
                justice: Vec2d::new(),
                fairness: Vec::new(),

                map: Vec::new(),

                output_names: Vec::new(),
                bad_names: Vec::new(),
                invariant_names: Vec::new(),
                justice_names: Vec::new(),
                fairness_names: Vec::new(),
            });
            Self {
                circuit: Circuit::new(vars),
                details: ProblemDetails::AIGER(details),
            }
        }

        fn with_outputs(mut self, outputs: impl IntoIterator<Item = Literal>) -> Self {
            if let ProblemDetails::AIGER(aig) = &mut self.details {
                aig.outputs = outputs.into_iter().collect()
            }
            self
        }

        fn with_named_outputs<'a>(
            mut self,
            outputs: impl IntoIterator<Item = (Literal, &'a str)>,
        ) -> Self {
            if let ProblemDetails::AIGER(aig) = &mut self.details {
                (aig.outputs, aig.output_names) = outputs
                    .into_iter()
                    .map(|(l, s)| (l, Some(s.to_string())))
                    .unzip();
            }
            self
        }

        fn with_and_gates(
            mut self,
            and_gates: impl IntoIterator<Item = (Literal, Literal)>,
        ) -> Self {
            let it = and_gates.into_iter();
            let n = it.size_hint().0;
            self.circuit.reserve_gates(n);
            self.circuit.reserve_gate_inputs(2 * n);

            for (l1, l2) in it {
                self.circuit.push_gate(crate::GateKind::And);
                self.circuit.push_gate_inputs([l1, l2]);
            }

            self
        }

        /// Add latches as well as the respective number of inputs to the
        /// combinational circuit
        fn with_latches(mut self, latches: impl IntoIterator<Item = Literal>) -> Self {
            let ProblemDetails::AIGER(aig) = &mut self.details else {
                panic!();
            };
            aig.latches = latches.into_iter().collect();
            self.circuit.inputs.len += aig.latches.len();
            aig.latch_init_values = TVBitVec::with_capacity(aig.latches.len());
            for _ in 0..aig.latches.len() {
                aig.latch_init_values.push(Some(false));
            }

            self
        }

        fn with_default_map(mut self) -> Self {
            let inputs = self.circuit.inputs.len();
            let gates = self.circuit.num_gates();

            let ProblemDetails::AIGER(aig) = &mut self.details else {
                panic!();
            };
            aig.map.clear();
            aig.map.reserve(1 + inputs + gates);
            aig.map
                .extend((0..inputs + 1).map(|i| Literal::from_input_or_false(false, i)));
            aig.map
                .extend((0..gates).map(|i| Literal::from_gate(false, i)));

            self
        }

        fn with_map(mut self, map: Vec<Literal>) -> Self {
            if let ProblemDetails::AIGER(aig) = &mut self.details {
                aig.map = map;
            }
            self
        }

        fn with_bad(mut self, bad: impl IntoIterator<Item = Literal>) -> Self {
            let ProblemDetails::AIGER(aig) = &mut self.details else {
                panic!();
            };
            aig.bad = bad.into_iter().collect();
            self
        }

        fn with_invariants(mut self, invariants: impl IntoIterator<Item = Literal>) -> Self {
            let ProblemDetails::AIGER(aig) = &mut self.details else {
                panic!();
            };
            aig.invariants = invariants.into_iter().collect();
            self
        }

        fn with_justice<'a>(mut self, justice: impl IntoIterator<Item = &'a [Literal]>) -> Self {
            let ProblemDetails::AIGER(aig) = &mut self.details else {
                panic!();
            };
            aig.justice = Vec2d::from_iter(justice);
            self
        }

        fn with_fairness(mut self, fairness: impl IntoIterator<Item = Literal>) -> Self {
            let ProblemDetails::AIGER(aig) = &mut self.details else {
                panic!();
            };
            aig.fairness = fairness.into_iter().collect();
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

    fn test_aag_aig_match(aag: &[u8], aig: &[u8], expected: Problem) {
        let (_, problem) = super::parse::<()>(&OPTS_NO_ORDER)(aag).unwrap();
        assert_eq!(problem, expected, "aag does not match expected");
        let (_, problem) = super::parse::<()>(&OPTS_NO_ORDER)(aig).unwrap();
        assert_eq!(problem, expected, "aig does not match expected");
    }

    // cases (mostly) taken from https://github.com/arminbiere/aiger/blob/master/FORMAT

    #[test]
    fn aag_aig_match_empty() {
        test_aag_aig_match(
            b"aag 0 0 0 0 0\n",
            b"aig 0 0 0 0 0\n",
            Problem::new_aig(VarSet::new(0)).with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_false_out() {
        test_aag_aig_match(
            b"aag 0 0 0 1 0\n0\n",
            b"aig 0 0 0 1 0\n0\n",
            Problem::new_aig(VarSet::new(0))
                .with_outputs([Literal::FALSE])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_true_out() {
        test_aag_aig_match(
            b"aag 0 0 0 1 0\n1\n",
            b"aig 0 0 0 1 0\n1\n",
            Problem::new_aig(VarSet::new(0))
                .with_outputs([Literal::TRUE])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_in_out() {
        test_aag_aig_match(
            b"aag 1 1 0 1 0\n2\n2\n",
            b"aig 1 1 0 1 0\n2\n",
            Problem::new_aig(VarSet::new(1))
                .with_outputs([Literal::from_input(false, 0)])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_neg() {
        test_aag_aig_match(
            b"aag 1 1 0 1 0\n2\n3\n",
            b"aig 1 1 0 1 0\n3\n",
            Problem::new_aig(VarSet::new(1))
                .with_outputs([Literal::from_input(true, 0)])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_and() {
        test_aag_aig_match(
            b"aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n",
            b"aig 3 2 0 1 1\n6\n\x02\x02",
            Problem::new_aig(VarSet::new(2))
                .with_outputs([Literal::from_gate(false, 0)])
                .with_and_gates([(Literal::from_input(false, 1), Literal::from_input(false, 0))])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_or() {
        test_aag_aig_match(
            b"aag 3 2 0 1 1\n2\n4\n7\n6 5 3\n",
            b"aig 3 2 0 1 1\n7\n\x01\x02",
            Problem::new_aig(VarSet::new(2))
                .with_outputs([Literal::from_gate(true, 0)])
                .with_and_gates([(Literal::from_input(true, 1), Literal::from_input(true, 0))])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_half_adder() {
        // Since we do not normalize the parsed AIG and also include a literal map, it
        // is difficult to compare the ASCII and binary versions of this example
        // directly. Hence, we have two test cases here.
        let aag = b"aag 7 2 0 2 3\n\
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
            c\nhalf adder\n";
        let (_, problem) = super::parse::<()>(&OPTS_NO_ORDER)(aag).unwrap();

        let map = vec![
            Literal::FALSE,
            Literal::from_input(false, 0),
            Literal::from_input(false, 1),
            Literal::from_gate(false, 0),
            Literal::UNDEF,
            Literal::UNDEF,
            Literal::from_gate(false, 1),
            Literal::from_gate(false, 2),
        ];

        let expected =
            Problem::new_aig(VarSet::with_names(vec![Some("x".into()), Some("y".into())]))
                .with_named_outputs([(map[6 >> 1], "s"), (map[12 >> 1], "c")])
                .with_and_gates([
                    (!map[13 >> 1], !map[15 >> 1]),
                    (map[2 >> 1], map[4 >> 1]),
                    (!map[3 >> 1], !map[5 >> 1]),
                ])
                .with_map(map);
        assert_eq!(problem, expected);
    }

    #[test]
    fn aig_half_adder() {
        let aig = b"aig 5 2 0 2 3\n\
            10\n\
            6\n\
            \x02\x02\
            \x03\x02\
            \x01\x02\
            i0 x\n\
            i1 y\n\
            o0 s\n\
            o1 c\n\
            c\nhalf adder\n";
        let (_, problem) = super::parse::<()>(&OPTS_NO_ORDER)(aig).unwrap();

        let expected =
            Problem::new_aig(VarSet::with_names(vec![Some("x".into()), Some("y".into())]))
                .with_named_outputs([
                    (Literal::from_gate(false, 2), "s"),
                    (Literal::from_gate(false, 0), "c"),
                ])
                .with_and_gates([
                    (Literal::from_input(false, 1), Literal::from_input(false, 0)),
                    (Literal::from_input(true, 1), Literal::from_input(true, 0)),
                    (Literal::from_gate(true, 1), Literal::from_gate(true, 0)),
                ])
                .with_default_map();

        assert_eq!(problem, expected);
    }

    #[test]
    fn aag_aig_match_toggle() {
        test_aag_aig_match(
            b"aag 1 0 1 2 0\n2 3\n2\n3\n",
            b"aig 1 0 1 2 0\n3\n2\n3\n",
            Problem::new_aig(VarSet::new(0))
                .with_latches([Literal::from_input(true, 0)])
                .with_outputs([Literal::from_input(false, 0), Literal::from_input(true, 0)])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_toggle_with_reset() {
        let aag = b"aag 7 2 1 2 4\n\
            2\n\
            4\n\
            6 8\n\
            6\n\
            7\n\
            8 4 10\n\
            10 13 15\n\
            12 2 6\n\
            14 3 7\n\
            i0 toggle\n\
            i1 ~reset\n\
            o0 q\n\
            o1 ~q\n\
            l0 q\n\
            c foobar\n";
        let (_, problem) = super::parse::<()>(&OPTS_NO_ORDER)(aag).unwrap();

        let mut expected = Problem::new_aig(VarSet::with_names(vec![
            Some("toggle".into()),
            Some("~reset".into()),
        ]))
        .with_latches([Literal::from_gate(false, 0)])
        .with_named_outputs([
            (Literal::from_input(false, 2), "q"),
            (Literal::from_input(true, 2), "~q"),
        ])
        .with_and_gates([
            (Literal::from_input(false, 1), Literal::from_gate(false, 1)),
            (Literal::from_gate(true, 2), Literal::from_gate(true, 3)),
            (Literal::from_input(false, 0), Literal::from_input(false, 2)),
            (Literal::from_input(true, 0), Literal::from_input(true, 2)),
        ])
        .with_default_map();

        expected.circuit.inputs.set_name(2, "q");

        assert_eq!(problem, expected);
    }

    #[test]
    fn aig_toggle_with_reset() {
        let aig = b"aig 7 2 1 2 4\n\
            14\n\
            6\n\
            7\n\
            \x02\x04\
            \x03\x04\
            \x01\x02\
            \x02\x08";
        let (_, problem) = super::parse::<()>(&OPTS_NO_ORDER)(aig).unwrap();

        let expected = Problem::new_aig(VarSet::new(2))
            .with_latches([Literal::from_gate(false, 3)])
            .with_outputs([Literal::from_input(false, 2), Literal::from_input(true, 2)])
            .with_and_gates([
                (Literal::from_input(false, 2), Literal::from_input(false, 0)),
                (Literal::from_input(true, 2), Literal::from_input(true, 0)),
                (Literal::from_gate(true, 1), Literal::from_gate(true, 0)),
                (Literal::from_gate(false, 2), Literal::from_input(false, 1)),
            ])
            .with_default_map();
        assert_eq!(problem, expected);
    }

    // next case taken from "AIGER 1.9 And Beyond"

    #[test]
    fn aag_aig_match_bad_and_invariant() {
        test_aag_aig_match(
            b"aag 5 1 1 0 3 1 1\n\
            2\n\
            4 10 0\n\
            4\n\
            3\n\
            6 5 3\n\
            8 4 2\n\
            10 9 7\n",
            b"aig 5 1 1 0 3 1 1\n\
            10 0\n\
            4\n\
            3\n\
            \x01\x02\x04\x02\x01\x02",
            Problem::new_aig(VarSet::new(1))
                .with_latches([Literal::from_gate(false, 2)])
                .with_and_gates([
                    (Literal::from_input(true, 1), Literal::from_input(true, 0)),
                    (Literal::from_input(false, 1), Literal::from_input(false, 0)),
                    (Literal::from_gate(true, 1), Literal::from_gate(true, 0)),
                ])
                .with_bad([Literal::from_input(false, 1)])
                .with_invariants([Literal::from_input(true, 0)])
                .with_default_map(),
        );
    }

    #[test]
    fn aag_aig_match_extra_properties() {
        // M I L O A B C J F
        test_aag_aig_match(
            b"aag 3 2 0 1 1 1 1 2 1\n2\n4\n6\n2\n3\n1\n2\n1\n4\n5\n6\n6 4 2\n",
            b"aig 3 2 0 1 1 1 1 2 1\n6\n2\n3\n1\n2\n1\n4\n5\n6\n\x02\x02",
            Problem::new_aig(VarSet::new(2))
                .with_outputs([Literal::from_gate(false, 0)])
                .with_and_gates([(Literal::from_input(false, 1), Literal::from_input(false, 0))])
                .with_bad([Literal::from_input(false, 0)])
                .with_invariants([Literal::from_input(true, 0)])
                .with_justice([
                    [Literal::TRUE].as_slice(),
                    &[Literal::from_input(false, 1), Literal::from_input(true, 1)],
                ])
                .with_fairness([Literal::from_gate(false, 0)])
                .with_default_map(),
        );
    }
}
