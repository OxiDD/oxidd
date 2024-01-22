//! Tests for BooleanFunction implementations

#![cfg_attr(miri, allow(unused))]

use std::fmt;
use std::fmt::Write;

use oxidd::bdd::BDDFunction;
use oxidd::cbdd::CBDDFunction;
use oxidd::zbdd::ZBDDSet;
use oxidd::AllocResult;
use oxidd::BooleanFunction;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::ManagerRef;

/// Propositional logic formula
#[derive(Clone, PartialEq, Eq, Debug)]
enum Prop {
    False,
    True,
    Var(u32),
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
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Diff {
    expected: bool,
    actual: bool,
    env: u32,
}

#[derive(Clone)]
struct Error<F: fmt::Display> {
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
                let c = if diff.env & (1 << var) != 0 { '1' } else { '0' };
                f.write_char(c)?;
            }
            writeln!(f, "  expected: {}, got: {}", diff.expected, diff.actual)?;
        }
        Ok(())
    }
}

impl Prop {
    fn eval(&self, env: u32) -> bool {
        use Prop::*;
        match self {
            False => false,
            True => true,
            Var(i) => env & (1 << *i) != 0,
            Not(p) => !p.eval(env),
            And(p, q) => p.eval(env) && q.eval(env),
            Or(p, q) => p.eval(env) || q.eval(env),
            Xor(p, q) => p.eval(env) ^ q.eval(env),
            Equiv(p, q) => p.eval(env) == q.eval(env),
            Nand(p, q) => !(p.eval(env) && q.eval(env)),
            Nor(p, q) => !(p.eval(env) || q.eval(env)),
            Imp(p, q) => !p.eval(env) || q.eval(env),
            ImpStrict(p, q) => !p.eval(env) && q.eval(env),
            Ite(i, t, e) => {
                if i.eval(env) {
                    t.eval(env)
                } else {
                    e.eval(env)
                }
            }
        }
    }

    fn cons<B: BooleanFunction>(&self, manager: &B::ManagerRef, vars: &[B]) -> AllocResult<B> {
        match self {
            Prop::False => Ok(manager.with_manager_shared(|m| B::f(m))),
            Prop::True => Ok(manager.with_manager_shared(|m| B::t(m))),
            Prop::Var(i) => Ok(vars[*i as usize].clone()),
            Prop::Not(p) => p.cons(manager, vars)?.not(),
            Prop::And(p, q) => p.cons(manager, vars)?.and(&q.cons(manager, vars)?),
            Prop::Or(p, q) => p.cons(manager, vars)?.or(&q.cons(manager, vars)?),
            Prop::Xor(p, q) => p.cons(manager, vars)?.xor(&q.cons(manager, vars)?),
            Prop::Equiv(p, q) => p.cons(manager, vars)?.equiv(&q.cons(manager, vars)?),
            Prop::Nand(p, q) => p.cons(manager, vars)?.nand(&q.cons(manager, vars)?),
            Prop::Nor(p, q) => p.cons(manager, vars)?.nor(&q.cons(manager, vars)?),
            Prop::Imp(p, q) => p.cons(manager, vars)?.imp(&q.cons(manager, vars)?),
            Prop::ImpStrict(p, q) => p.cons(manager, vars)?.imp_strict(&q.cons(manager, vars)?),
            Prop::Ite(i, t, e) => i
                .cons(manager, vars)?
                .ite(&t.cons(manager, vars)?, &e.cons(manager, vars)?),
        }
    }

    fn build_and_check<B: BooleanFunction>(
        &self,
        manager: &B::ManagerRef,
        vars: &[B],
        var_roots: &[B],
    ) -> Result<(), Error<&Self>> {
        assert!(vars.len() == var_roots.len());
        assert!(
            vars.len() < 31,
            "Too many variables for exhaustive checking"
        );
        let end = 1u32 << vars.len();
        let f = self.cons(manager, vars).expect("out of memory");
        let mut diffs = Vec::new();
        for env in 0..end {
            let expected = self.eval(env);
            let actual = f.eval((0..var_roots.len()).map(|i| (&var_roots[i], env & (1 << i) != 0)));
            if expected != actual {
                diffs.push(Diff {
                    expected,
                    actual,
                    env,
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
    /// `<= depth` and up to `nvars` variables. If you start with `Prop::False`,
    /// and repeatedly call this method, you will enumerate all such formulas.
    /// Returns `false` iff this if the last formula. In this case the formula
    /// is left unchanged.
    fn next(&mut self, depth: u32, nvars: u32) -> bool {
        use Prop::*;
        match self {
            False => {
                *self = True;
                true
            }
            True => {
                if nvars > 0 {
                    *self = Var(0);
                    true
                } else if depth > 0 {
                    *self = Not(Box::new(False));
                    true
                } else {
                    false
                }
            }
            Var(i) => {
                if *i + 1 < nvars {
                    *self = Var(*i + 1);
                    true
                } else if depth > 0 {
                    *self = Not(Box::new(False));
                    true
                } else {
                    false
                }
            }

            Not(p) => {
                if !p.next(depth - 1, nvars) {
                    *self = And(Box::new(False), Box::new(False));
                }
                true
            }
            And(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Or(Box::new(False), Box::new(False));
                }
                true
            }
            Or(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Xor(Box::new(False), Box::new(False));
                }
                true
            }
            Xor(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Equiv(Box::new(False), Box::new(False));
                }
                true
            }
            Equiv(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Nand(Box::new(False), Box::new(False));
                }
                true
            }
            Nand(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Nor(Box::new(False), Box::new(False));
                }
                true
            }
            Nor(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Imp(Box::new(False), Box::new(False));
                }
                true
            }
            Imp(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = ImpStrict(Box::new(False), Box::new(False));
                }
                true
            }
            ImpStrict(p, q) => {
                if !p.next(depth - 1, nvars) && !q.next(depth - 1, nvars) {
                    *self = Ite(Box::new(False), Box::new(False), Box::new(False));
                }
                true
            }
            Ite(i, t, e) => {
                if !i.next(depth - 1, nvars)
                    && !t.next(depth - 1, nvars)
                    && !e.next(depth - 1, nvars)
                {
                    return false;
                }
                true
            }
        }
    }
}

fn test_simple_formulas<B: BooleanFunction>(manager: &B::ManagerRef) {
    use Prop::*;

    for op1 in [false, true] {
        Prop::from(op1)
            .build_and_check::<B>(manager, &[], &[])
            .unwrap();

        Not(Box::new(op1.into()))
            .build_and_check::<B>(manager, &[], &[])
            .unwrap();

        for op2 in [false, true] {
            for binop in [And, Or, Xor, Equiv, Nand, Nor, Imp, ImpStrict] {
                binop(Box::new(op1.into()), Box::new(op2.into()))
                    .build_and_check::<B>(manager, &[], &[])
                    .unwrap();
            }

            for op3 in [false, true] {
                Ite(
                    Box::new(op1.into()),
                    Box::new(op2.into()),
                    Box::new(op3.into()),
                )
                .build_and_check::<B>(manager, &[], &[])
                .unwrap();
            }
        }
    }
}

#[test]
fn bdd_test_simple_formulas() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<BDDFunction>(&mref);
}

#[test]
fn cbdd_test_simple_formulas() {
    let mref = oxidd::cbdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<CBDDFunction>(&mref);
}

#[test]
fn zbdd_test_simple_formulas() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<ZBDDSet>(&mref);
}

fn test_depth3_3vars<B: BooleanFunction>(manager: &B::ManagerRef, vars: &[B], var_roots: &[B]) {
    assert_eq!(vars.len(), 3);
    let mut f = Prop::False;
    loop {
        f.build_and_check(manager, vars, var_roots).unwrap();

        if !f.next(3, 3) {
            break;
        }
    }
}

#[cfg(not(miri))]
#[test]
fn bdd_test_depth3_3vars() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    let vars = mref.with_manager_exclusive(|manager| {
        [
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::new_var(manager).unwrap(),
        ]
    });
    test_depth3_3vars(&mref, &vars, &vars);
}

#[cfg(not(miri))]
#[test]
fn cbdd_test_depth3_3vars() {
    let mref = oxidd::cbdd::new_manager(65536, 1024, 2);
    let vars = mref.with_manager_exclusive(|manager| {
        [
            CBDDFunction::new_var(manager).unwrap(),
            CBDDFunction::new_var(manager).unwrap(),
            CBDDFunction::new_var(manager).unwrap(),
        ]
    });
    test_depth3_3vars(&mref, &vars, &vars);
}

#[cfg(not(miri))]
#[test]
fn zbdd_test_depth3_3vars() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    let singletons = mref.with_manager_exclusive(|manager| {
        [
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::new_singleton(manager).unwrap(),
        ]
    });
    let vars = mref.with_manager_shared(|manager| {
        [
            ZBDDSet::from_edge(
                manager,
                oxidd::zbdd::var_boolean_function(manager, singletons[0].as_edge(manager)).unwrap(),
            ),
            ZBDDSet::from_edge(
                manager,
                oxidd::zbdd::var_boolean_function(manager, singletons[1].as_edge(manager)).unwrap(),
            ),
            ZBDDSet::from_edge(
                manager,
                oxidd::zbdd::var_boolean_function(manager, singletons[2].as_edge(manager)).unwrap(),
            ),
        ]
    });
    test_depth3_3vars(&mref, &vars, &singletons);
}
