use std::fmt;
use std::fmt::Write;

use oxidd::util::AllocResult;
use oxidd::BooleanFunction;
use oxidd::BooleanFunctionQuant;
use oxidd::ManagerRef;

// spell-checker:ignore nvars

pub type Var = u32;
pub type VarSet = u32;

/// Propositional logic formula
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Prop {
    False,
    True,
    Var(Var),
    Not(Box<Prop>),
    And(Box<Prop>, Box<Prop>),
    Or(Box<Prop>, Box<Prop>),
    Xor(Box<Prop>, Box<Prop>),
    Equiv(Box<Prop>, Box<Prop>),
    Nand(Box<Prop>, Box<Prop>),
    Nor(Box<Prop>, Box<Prop>),
    Imp(Box<Prop>, Box<Prop>),
    ImpStrict(Box<Prop>, Box<Prop>),
    Ite(Box<Prop>, Box<Prop>, Box<Prop>),
    /// Restrict(positive, negative, f)
    Restrict(VarSet, VarSet, Box<Prop>),
    Exists(VarSet, Box<Prop>),
    Forall(VarSet, Box<Prop>),
    Unique(VarSet, Box<Prop>),
}

impl From<bool> for Prop {
    fn from(value: bool) -> Self {
        if value {
            Prop::True
        } else {
            Prop::False
        }
    }
}

impl fmt::Display for Prop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Prop::*;
        match self {
            False => f.write_str("⊥"),
            True => f.write_str("⊤"),
            Var(i) => write!(f, "x{i}"),
            Not(p) => write!(f, "(¬{p})"),
            And(p, q) => write!(f, "({p} ∧ {q})"),
            Or(p, q) => write!(f, "({p} ∨ {q})"),
            Xor(p, q) => write!(f, "({p} ⊕ {q})"),
            Equiv(p, q) => write!(f, "({p} ↔ {q})"),
            Nand(p, q) => write!(f, "({p} ⊼ {q})"),
            Nor(p, q) => write!(f, "({p} ⊽ {q})"),
            Imp(p, q) => write!(f, "({p} → {q})"),
            ImpStrict(p, q) => write!(f, "({p} < {q})"),
            Ite(p, q, r) => write!(f, "ite({p}, {q}, {r})"),
            Restrict(positive, negative, p) => {
                write!(f, "({p}|{positive:b},{negative:b})")
            }
            Exists(vars, p) => write!(f, "∃ {vars:b}. {p}"),
            Forall(vars, p) => write!(f, "∀ {vars:b}. {p}"),
            Unique(vars, p) => write!(f, "∃! {vars:b}. {p}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Diff {
    expected: bool,
    actual: bool,
    args: u32,
}

#[derive(Clone)]
pub struct Error<F: fmt::Display> {
    formula: F,
    nvars: u32,
    diffs: Vec<Diff>,
}

impl<F: fmt::Display> fmt::Debug for Error<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "differences for formula {}", self.formula)?;
        for diff in &self.diffs {
            f.write_str("   ")?;
            for var in 0..self.nvars {
                if var % 4 == 0 {
                    f.write_char(' ')?;
                }
                let c = if diff.args & (1 << var) != 0 {
                    '1'
                } else {
                    '0'
                };
                f.write_char(c)?;
            }
            writeln!(f, "  expected: {}, got: {}", diff.expected, diff.actual)?;
        }
        Ok(())
    }
}

impl Prop {
    fn eval(&self, args: u32) -> bool {
        use Prop::*;

        /// Q: 0 = ∃, 1 = ∀, 2 = ∃!
        fn quant_inner<const Q: u8>(vars: u32, args: u32, p: &Prop) -> bool {
            if vars == 0 {
                p.eval(args)
            } else {
                let trailing = vars.trailing_zeros();
                let v = 1 << trailing;
                let vars = vars & !v;
                match Q {
                    0 => {
                        quant_inner::<Q>(vars, args | v, p) || quant_inner::<Q>(vars, args & !v, p)
                    }
                    1 => {
                        quant_inner::<Q>(vars, args | v, p) && quant_inner::<Q>(vars, args & !v, p)
                    }
                    2 => quant_inner::<Q>(vars, args | v, p) ^ quant_inner::<Q>(vars, args & !v, p),
                    _ => unreachable!(),
                }
            }
        }

        match self {
            False => false,
            True => true,
            Var(i) => args & (1 << *i) != 0,
            Not(p) => !p.eval(args),
            And(p, q) => p.eval(args) && q.eval(args),
            Or(p, q) => p.eval(args) || q.eval(args),
            Xor(p, q) => p.eval(args) ^ q.eval(args),
            Equiv(p, q) => p.eval(args) == q.eval(args),
            Nand(p, q) => !(p.eval(args) && q.eval(args)),
            Nor(p, q) => !(p.eval(args) || q.eval(args)),
            Imp(p, q) => !p.eval(args) || q.eval(args),
            ImpStrict(p, q) => !p.eval(args) && q.eval(args),
            Ite(i, t, e) => {
                if i.eval(args) {
                    t.eval(args)
                } else {
                    e.eval(args)
                }
            }
            Restrict(positive, negative, p) => p.eval((args | positive) & !negative),
            Exists(vars, p) => quant_inner::<0>(*vars, args, p),
            Forall(vars, p) => quant_inner::<1>(*vars, args, p),
            Unique(vars, p) => quant_inner::<2>(*vars, args, p),
        }
    }

    fn cons<B: BooleanFunction>(&self, manager: &B::ManagerRef, vars: &[B]) -> AllocResult<B> {
        use Prop::*;
        match self {
            False => Ok(manager.with_manager_shared(|m| B::f(m))),
            True => Ok(manager.with_manager_shared(|m| B::t(m))),
            Var(i) => Ok(vars[*i as usize].clone()),
            Not(p) => p.cons(manager, vars)?.not(),
            And(p, q) => p.cons(manager, vars)?.and(&q.cons(manager, vars)?),
            Or(p, q) => p.cons(manager, vars)?.or(&q.cons(manager, vars)?),
            Xor(p, q) => p.cons(manager, vars)?.xor(&q.cons(manager, vars)?),
            Equiv(p, q) => p.cons(manager, vars)?.equiv(&q.cons(manager, vars)?),
            Nand(p, q) => p.cons(manager, vars)?.nand(&q.cons(manager, vars)?),
            Nor(p, q) => p.cons(manager, vars)?.nor(&q.cons(manager, vars)?),
            Imp(p, q) => p.cons(manager, vars)?.imp(&q.cons(manager, vars)?),
            ImpStrict(p, q) => p.cons(manager, vars)?.imp_strict(&q.cons(manager, vars)?),
            Ite(i, t, e) => i
                .cons(manager, vars)?
                .ite(&t.cons(manager, vars)?, &e.cons(manager, vars)?),
            Restrict(..) | Exists(..) | Forall(..) | Unique(..) => {
                panic!("cons cannot construct quantification formulas, use `cons_q()` instead")
            }
        }
    }

    fn cons_q<B: BooleanFunctionQuant>(
        &self,
        manager: &B::ManagerRef,
        vars: &[B],
    ) -> AllocResult<B> {
        use Prop::*;
        match self {
            False => Ok(manager.with_manager_shared(|m| B::f(m))),
            True => Ok(manager.with_manager_shared(|m| B::t(m))),
            Var(i) => Ok(vars[*i as usize].clone()),
            Not(p) => p.cons_q(manager, vars)?.not(),
            And(p, q) => p.cons_q(manager, vars)?.and(&q.cons_q(manager, vars)?),
            Or(p, q) => p.cons_q(manager, vars)?.or(&q.cons_q(manager, vars)?),
            Xor(p, q) => p.cons_q(manager, vars)?.xor(&q.cons_q(manager, vars)?),
            Equiv(p, q) => p.cons_q(manager, vars)?.equiv(&q.cons_q(manager, vars)?),
            Nand(p, q) => p.cons_q(manager, vars)?.nand(&q.cons_q(manager, vars)?),
            Nor(p, q) => p.cons_q(manager, vars)?.nor(&q.cons_q(manager, vars)?),
            Imp(p, q) => p.cons_q(manager, vars)?.imp(&q.cons_q(manager, vars)?),
            ImpStrict(p, q) => p
                .cons_q(manager, vars)?
                .imp_strict(&q.cons_q(manager, vars)?),
            Ite(i, t, e) => i
                .cons_q(manager, vars)?
                .ite(&t.cons_q(manager, vars)?, &e.cons_q(manager, vars)?),
            Restrict(mut pos, mut neg, p) => {
                debug_assert_eq!(pos & neg, 0);
                let prop = p.cons_q(manager, vars)?;
                // Convert the positive (`pos`) and negative (`neg`) literals
                // into a decision diagram (`v_dd`)
                let mut v_dd = manager.with_manager_shared(|m| B::t(m));
                let mut i = 0;
                while pos | neg != 0 {
                    if pos & 1 != 0 {
                        v_dd = v_dd.and(&vars[i])?;
                    } else if neg & 1 != 0 {
                        v_dd = v_dd.and(&vars[i].not()?)?;
                    }
                    i += 1;
                    pos >>= 1;
                    neg >>= 1;
                }
                prop.restrict(&v_dd)
            }
            Exists(mut v, p) | Forall(mut v, p) | Unique(mut v, p) => {
                let prop = p.cons_q(manager, vars)?;
                // Construct `v` as decision diagram (`v_dd`)
                let mut v_dd = manager.with_manager_shared(|m| B::t(m));
                let mut i = 0;
                while v != 0 {
                    if v & 1 != 0 {
                        v_dd = v_dd.and(&vars[i])?;
                    }
                    i += 1;
                    v >>= 1;
                }
                // Perform the quantification
                match self {
                    Exists(..) => prop.exist(&v_dd),
                    Forall(..) => prop.forall(&v_dd),
                    Unique(..) => prop.unique(&v_dd),
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn build_and_check<B: BooleanFunction>(
        &self,
        manager: &B::ManagerRef,
        vars: &[B],
        var_handles: &[B],
    ) -> Result<(), Error<&Self>> {
        assert!(vars.len() == var_handles.len());
        assert!(
            vars.len() < 31,
            "Too many variables for exhaustive checking"
        );
        let end = 1u32 << vars.len();
        let f = self.cons(manager, vars).expect("out of memory");
        let mut diffs = Vec::new();
        for args in 0..end {
            let expected = self.eval(args);
            let actual =
                f.eval((0..var_handles.len()).map(|i| (&var_handles[i], args & (1 << i) != 0)));
            if expected != actual {
                diffs.push(Diff {
                    expected,
                    actual,
                    args,
                })
            }
        }
        if diffs.is_empty() {
            Ok(())
        } else {
            Err(Error {
                formula: self,
                nvars: vars.len() as u32,
                diffs,
            })
        }
    }

    pub fn build_and_check_quant<B: BooleanFunctionQuant>(
        &self,
        manager: &B::ManagerRef,
        vars: &[B],
        var_handles: &[B],
    ) -> Result<(), Error<&Self>> {
        assert!(vars.len() == var_handles.len());
        assert!(
            vars.len() < 31,
            "Too many variables for exhaustive checking"
        );
        let end = 1u32 << vars.len();
        let f = self.cons_q(manager, vars).expect("out of memory");
        let mut diffs = Vec::new();
        for args in 0..end {
            let expected = self.eval(args);
            let actual =
                f.eval((0..var_handles.len()).map(|i| (&var_handles[i], args & (1 << i) != 0)));
            if expected != actual {
                diffs.push(Diff {
                    expected,
                    actual,
                    args,
                })
            }
        }
        if diffs.is_empty() {
            Ok(())
        } else {
            Err(Error {
                formula: self,
                nvars: vars.len() as u32,
                diffs,
            })
        }
    }

    /// This method is used to enumerate all propositional formulas with depth
    /// `<= depth` and up to `nvars` variables (`depth` is counted in terms of
    /// edges, i.e., `False` has depth 0). If you start with `Prop::False`, and
    /// repeatedly call this method, you will enumerate all such formulas.
    /// Returns `false` iff this is the last formula. In this case the formula
    /// is left unchanged.
    ///
    /// If `QUANT` is set to `true`, this will also enumerate operators defined
    /// in the [`BooleanFunctionQuant`] trait.
    pub fn next<const QUANT: bool>(&mut self, depth: u32, nvars: u32) -> bool {
        use Prop::*;
        match self {
            False => *self = True,
            True => {
                if nvars > 0 {
                    *self = Var(0);
                } else if depth > 0 {
                    *self = Not(Box::new(False));
                } else {
                    return false;
                }
            }
            Var(i) => {
                if *i + 1 < nvars {
                    *self = Var(*i + 1);
                } else if depth > 0 {
                    *self = Not(Box::new(False));
                } else {
                    return false;
                }
            }
            Not(p) => {
                if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = And(Box::new(False), Box::new(False));
                }
            }
            And(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Or(Box::new(False), Box::new(False));
                    }
                }
            }
            Or(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Xor(Box::new(False), Box::new(False));
                    }
                }
            }
            Xor(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Equiv(Box::new(False), Box::new(False));
                    }
                }
            }
            Equiv(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Nand(Box::new(False), Box::new(False));
                    }
                }
            }
            Nand(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Nor(Box::new(False), Box::new(False));
                    }
                }
            }
            Nor(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Imp(Box::new(False), Box::new(False));
                    }
                }
            }
            Imp(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = ImpStrict(Box::new(False), Box::new(False));
                    }
                }
            }
            ImpStrict(p, q) => {
                if !q.next::<QUANT>(depth - 1, nvars) {
                    **q = False;
                    if !p.next::<QUANT>(depth - 1, nvars) {
                        *self = Ite(Box::new(False), Box::new(False), Box::new(False));
                    }
                }
            }
            Ite(i, t, e) => {
                if !e.next::<QUANT>(depth - 1, nvars) {
                    **e = False;
                    if !t.next::<QUANT>(depth - 1, nvars) {
                        **t = False;
                        if !i.next::<QUANT>(depth - 1, nvars) {
                            if !QUANT {
                                return false;
                            }
                            *self = Restrict(0, 0, Box::new(False));
                        }
                    }
                }
            }
            Restrict(positive, negative, p) => {
                if *positive < (1 << nvars) - 1 {
                    // Given `negative`, compute the next `positive` such that
                    // both don't share any bits, i.e., we don't assign both
                    // true and false to a variable.
                    *positive += 1;
                    loop {
                        let both = *positive & *negative;
                        if both == 0 {
                            return true;
                        }
                        let trailing = both.trailing_zeros();
                        if (1 << trailing) > (1 << nvars) - 1 - *positive {
                            break; // *positive + (1 << trailing) is too large
                        }
                        *positive += 1 << trailing;
                    }
                }
                *positive = 0;
                if *negative < (1 << nvars) - 1 {
                    *negative += 1;
                    return true;
                }
                *negative = 0;
                if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = Exists(0, Box::new(False))
                }
            }
            Exists(v, p) => {
                if *v < (1 << nvars) - 1 {
                    *v += 1;
                    return true;
                }
                *v = 0;
                if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = Forall(0, Box::new(False))
                }
            }
            Forall(v, p) => {
                if *v < (1 << nvars) - 1 {
                    *v += 1;
                    return true;
                }
                *v = 0;
                if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = Unique(0, Box::new(False))
                }
            }
            Unique(v, p) => {
                if *v < (1 << nvars) - 1 {
                    *v += 1;
                    return true;
                }
                *v = 0;
                if !p.next::<QUANT>(depth - 1, nvars) {
                    return false;
                }
            }
        }
        true
    }
}
