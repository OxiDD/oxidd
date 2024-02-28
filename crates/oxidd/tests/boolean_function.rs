//! Tests for BooleanFunction implementations

#![cfg_attr(miri, allow(unused))]

use std::fmt;
use std::fmt::Write;

use oxidd::bcdd::BCDDFunction;
use oxidd::bdd::BDDFunction;
use oxidd::zbdd::ZBDDSet;
use oxidd::AllocResult;
use oxidd::BooleanFunction;
use oxidd::BooleanFunctionQuant;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::ManagerRef;

// spell-checker:ignore nvars,mref

#[test]
fn bdd_node_count() {
    let mref = oxidd::bdd::new_manager(1024, 128, 2);

    let (x0, x1, ff, tt) = mref.with_manager_exclusive(|manager| {
        (
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::f(manager),
            BDDFunction::t(manager),
        )
    });

    assert_eq!(ff.node_count(), 1);
    assert_eq!(tt.node_count(), 1);

    assert_eq!(x0.node_count(), 3);
    assert_eq!(x1.node_count(), 3);

    let g = x0.and(&x1).unwrap().not().unwrap();
    assert_eq!(g.node_count(), 4);
}

#[test]
fn bcdd_node_count() {
    let mref = oxidd::bcdd::new_manager(1024, 128, 2);

    let (x0, x1, ff, tt) = mref.with_manager_exclusive(|manager| {
        (
            BCDDFunction::new_var(manager).unwrap(),
            BCDDFunction::new_var(manager).unwrap(),
            BCDDFunction::f(manager),
            BCDDFunction::t(manager),
        )
    });

    assert_eq!(ff.node_count(), 1);
    assert_eq!(tt.node_count(), 1);

    assert_eq!(x0.node_count(), 2);
    assert_eq!(x1.node_count(), 2);

    let g = x0.and(&x1).unwrap().not().unwrap();
    assert_eq!(g.node_count(), 3);
}

// TODO: move this test to its own module?
#[test]
fn zbdd_node_count() {
    let mref = oxidd::zbdd::new_manager(1024, 128, 2);

    let (x0, x1, ee, bb) = mref.with_manager_exclusive(|manager| {
        (
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::empty(manager),
            ZBDDSet::base(manager),
        )
    });

    assert_eq!(ee.node_count(), 1);
    assert_eq!(bb.node_count(), 1);

    assert_eq!(x0.node_count(), 3);
    assert_eq!(x1.node_count(), 3);

    let p = x0.union(&x1).unwrap();
    assert_eq!(p.node_count(), 4);
}

#[test]
fn bdd_cofactors() {
    let mref = oxidd::bdd::new_manager(1024, 128, 2);

    let (x0, x1, ff, tt) = mref.with_manager_exclusive(|manager| {
        (
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::f(manager),
            BDDFunction::t(manager),
        )
    });
    let g = x0.and(&x1).unwrap().not().unwrap();

    assert!(ff.cofactors().is_none());
    assert!(ff.cofactor_true().is_none());
    assert!(ff.cofactor_false().is_none());
    assert!(tt.cofactors().is_none());
    assert!(tt.cofactor_true().is_none());
    assert!(tt.cofactor_false().is_none());

    for v in [&x0, &x1] {
        assert_eq!(v.node_count(), 3);

        let (vt, vf) = v.cofactors().unwrap();
        assert!(vt == v.cofactor_true().unwrap());
        assert!(vf == v.cofactor_false().unwrap());

        assert!(vt == tt);
        assert!(vf == ff);
    }

    let (gt, gf) = g.cofactors().unwrap();
    assert!(gt == g.cofactor_true().unwrap());
    assert!(gf == g.cofactor_false().unwrap());
    assert!(gt == x1.not().unwrap());
    assert!(gf == tt);
}

#[test]
fn bcdd_cofactors() {
    let mref = oxidd::bcdd::new_manager(1024, 128, 2);

    let (x0, x1, ff, tt) = mref.with_manager_exclusive(|manager| {
        (
            BCDDFunction::new_var(manager).unwrap(),
            BCDDFunction::new_var(manager).unwrap(),
            BCDDFunction::f(manager),
            BCDDFunction::t(manager),
        )
    });
    let g = x0.and(&x1).unwrap().not().unwrap();

    assert!(ff.cofactors().is_none());
    assert!(ff.cofactor_true().is_none());
    assert!(ff.cofactor_false().is_none());
    assert!(tt.cofactors().is_none());
    assert!(tt.cofactor_true().is_none());
    assert!(tt.cofactor_false().is_none());

    for v in [&x0, &x1] {
        let (vt, vf) = v.cofactors().unwrap();
        assert!(vt == v.cofactor_true().unwrap());
        assert!(vf == v.cofactor_false().unwrap());

        assert!(vt == tt);
        assert!(vf == ff);
    }

    let (gt, gf) = g.cofactors().unwrap();
    assert!(gt == g.cofactor_true().unwrap());
    assert!(gf == g.cofactor_false().unwrap());
    assert!(gt == x1.not().unwrap());
    assert!(gf == tt);
}

#[test]
fn zbdd_cofactors() {
    let mref = oxidd::zbdd::new_manager(1024, 128, 2);

    let (x0, x1, ee, bb) = mref.with_manager_exclusive(|manager| {
        (
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::empty(manager),
            ZBDDSet::base(manager),
        )
    });
    let g = x0.union(&x1).unwrap();

    assert!(ee.cofactors().is_none());
    assert!(ee.cofactor_true().is_none());
    assert!(ee.cofactor_false().is_none());
    assert!(bb.cofactors().is_none());
    assert!(bb.cofactor_true().is_none());
    assert!(bb.cofactor_false().is_none());

    for v in [&x0, &x1] {
        let (vt, vf) = v.cofactors().unwrap();
        assert!(vt == v.cofactor_true().unwrap());
        assert!(vf == v.cofactor_false().unwrap());

        assert!(vt == bb);
        assert!(vf == ee);
    }

    let (gt, gf) = g.cofactors().unwrap();
    assert!(gt == g.cofactor_true().unwrap());
    assert!(gf == g.cofactor_false().unwrap());
    assert!(gt == bb);
    assert!(gf == x1);
}

type Var = u32;
type VarSet = u32;

/// Propositional logic formula
#[derive(Clone, PartialEq, Eq, Debug)]
enum Prop {
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
            Unique(vars, p) => write!(f, "U {vars:b}. {p}"),
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
            Restrict(positive, negative, p) => p.eval((env | positive) & !negative),
            Exists(vars, p) => p.eval(env | vars) || p.eval(env & !vars),
            Forall(vars, p) => p.eval(env | vars) && p.eval(env & !vars),
            Unique(vars, p) => p.eval(env | vars) ^ p.eval(env & !vars),
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

    fn build_and_check_quant<B: BooleanFunctionQuant>(
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
        let f = self.cons_q(manager, vars).expect("out of memory");
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
    ///
    /// If `QUANT` is set to `true`, this will also enumerate operators defined
    /// in the [`BooleanFunctionQuant`] trait.
    fn next<const QUANT: bool>(&mut self, depth: u32, nvars: u32) -> bool {
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
                if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = And(Box::new(False), Box::new(False));
                }
                true
            }
            And(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Or(Box::new(False), Box::new(False));
                }
                true
            }
            Or(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Xor(Box::new(False), Box::new(False));
                }
                true
            }
            Xor(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Equiv(Box::new(False), Box::new(False));
                }
                true
            }
            Equiv(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Nand(Box::new(False), Box::new(False));
                }
                true
            }
            Nand(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Nor(Box::new(False), Box::new(False));
                }
                true
            }
            Nor(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Imp(Box::new(False), Box::new(False));
                }
                true
            }
            Imp(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = ImpStrict(Box::new(False), Box::new(False));
                }
                true
            }
            ImpStrict(p, q) => {
                if !p.next::<QUANT>(depth - 1, nvars) && !q.next::<QUANT>(depth - 1, nvars) {
                    *self = Ite(Box::new(False), Box::new(False), Box::new(False));
                }
                true
            }
            Ite(i, t, e) => {
                if !i.next::<QUANT>(depth - 1, nvars)
                    && !t.next::<QUANT>(depth - 1, nvars)
                    && !e.next::<QUANT>(depth - 1, nvars)
                {
                    if QUANT {
                        *self = Restrict(0, 0, Box::new(False));
                    } else {
                        return false;
                    }
                }
                true
            }
            Restrict(positive, negative, p) => {
                if *positive + 1 < (1 << nvars) {
                    // Given `negative`, compute the next `positive` such that
                    // both don't share any bits, i.e., we don't assign both
                    // true and false to a variable.
                    *positive += 1;
                    loop {
                        let both = *positive & *negative;
                        if both == 0 {
                            break;
                        }
                        *positive += 1 << both.trailing_zeros()
                    }
                } else if *negative + 1 < (1 << nvars) {
                    *positive = 0;
                    *negative += 1;
                } else if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = Exists(0, Box::new(False))
                }
                true
            }
            Exists(v, p) => {
                if *v + 1 < (1 << nvars) {
                    *v += 1;
                } else if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = Forall(0, Box::new(False))
                }
                true
            }
            Forall(v, p) => {
                if *v + 1 < (1 << nvars) {
                    *v += 1;
                } else if !p.next::<QUANT>(depth - 1, nvars) {
                    *self = Unique(0, Box::new(False))
                }
                true
            }
            Unique(v, p) => {
                if *v + 1 < (1 << nvars) {
                    *v += 1;
                } else if !p.next::<QUANT>(depth - 1, nvars) {
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
fn bcdd_test_simple_formulas() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<BCDDFunction>(&mref);
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

        if !f.next::<false>(3, 3) {
            break;
        }
    }
}

fn test_depth3_3vars_quant<B: BooleanFunctionQuant>(
    manager: &B::ManagerRef,
    vars: &[B],
    var_roots: &[B],
) {
    assert_eq!(vars.len(), 3);
    let mut f = Prop::False;
    loop {
        f.build_and_check_quant(manager, vars, var_roots).unwrap();

        if !f.next::<false>(3, 3) {
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
    test_depth3_3vars_quant(&mref, &vars, &vars);
}

#[cfg(not(miri))]
#[test]
fn bcdd_test_depth3_3vars() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    let vars = mref.with_manager_exclusive(|manager| {
        [
            BCDDFunction::new_var(manager).unwrap(),
            BCDDFunction::new_var(manager).unwrap(),
            BCDDFunction::new_var(manager).unwrap(),
        ]
    });
    test_depth3_3vars_quant(&mref, &vars, &vars);
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
