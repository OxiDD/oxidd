//! Tests for BooleanFunction implementations

#![cfg_attr(miri, allow(unused))]

mod boolean_prop;
mod util;

use oxidd::bcdd::BCDDFunction;
use oxidd::bdd::BDDFunction;
use oxidd::zbdd::ZBDDManagerRef;
use oxidd::zbdd::ZBDDSet;
use oxidd::BooleanFunction;
use oxidd::BooleanFunctionQuant;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::ManagerRef;

use boolean_prop::Prop;
use util::progress::Progress;

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
fn bdd_test_simple_formulas_t1() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 1);
    test_simple_formulas::<BDDFunction>(&mref);
}

#[test]
fn bdd_test_simple_formulas_t2() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<BDDFunction>(&mref);
}

#[test]
fn bcdd_test_simple_formulas_t1() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 1);
    test_simple_formulas::<BCDDFunction>(&mref);
}

#[test]
fn bcdd_test_simple_formulas_t2() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<BCDDFunction>(&mref);
}

#[test]
fn zbdd_test_simple_formulas_t1() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 1);
    test_simple_formulas::<ZBDDSet>(&mref);
}

#[test]
fn zbdd_test_simple_formulas_t2() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<ZBDDSet>(&mref);
}

/// Works for BDDs & BCDDs
fn bdd_3_vars<B: BooleanFunction>(mref: &B::ManagerRef) -> [B; 3] {
    mref.with_manager_exclusive(|manager| {
        [
            B::new_var(manager).unwrap(),
            B::new_var(manager).unwrap(),
            B::new_var(manager).unwrap(),
        ]
    })
}

fn zbdd_3_singletons_vars(mref: &ZBDDManagerRef) -> ([ZBDDSet; 3], [ZBDDSet; 3]) {
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
    (singletons, vars)
}

#[test]
fn bcdd_test_restrict() {
    let mref = oxidd::bcdd::new_manager(1024, 1024, 2);
    let vars = bdd_3_vars::<BCDDFunction>(&mref);
    use Prop::*;
    let formulas = [
        Restrict(
            0b010,
            0b001,
            Box::new(Or(Box::new(Var(0)), Box::new(Var(1)))),
        ),
        Restrict(0b110, 0, Box::new(Var(0))),
        Restrict(0b110, 0, Box::new(Not(Box::new(Var(0))))),
        Restrict(0b0, 0b10, Box::new(And(Box::new(Var(0)), Box::new(Var(1))))),
        Restrict(
            0b100,
            0b001,
            Box::new(And(Box::new(Var(1)), Box::new(Var(2)))),
        ),
    ];
    for formula in formulas {
        formula.build_and_check_quant(&mref, &vars, &vars).unwrap();
    }
}

fn test_depth2_3vars<B: BooleanFunction>(manager: &B::ManagerRef, vars: &[B], var_roots: &[B]) {
    assert_eq!(vars.len(), 3);
    let mut f = Prop::False;
    let mut progress = Progress::new(38_493_515);
    loop {
        f.build_and_check(manager, vars, var_roots).unwrap();
        progress.step();

        if !f.next::<false>(2, 3) {
            break;
        }
    }
    progress.done();
}

fn test_depth2_3vars_quant<B: BooleanFunctionQuant>(
    manager: &B::ManagerRef,
    vars: &[B],
    var_roots: &[B],
) {
    assert_eq!(vars.len(), 3);
    let mut f = Prop::False;
    let mut progress = Progress::new(208_194_485);
    loop {
        f.build_and_check_quant(manager, vars, var_roots).unwrap();
        progress.step();

        if !f.next::<true>(2, 3) {
            break;
        }
    }
    progress.done();
}

// The following tests are expensive, hence we use `#[ignore]`, see
// https://doc.rust-lang.org/book/ch11-02-running-tests.html#ignoring-some-tests-unless-specifically-requested.
// You can set the `OXIDD_TESTING_PROGRESS` environment variable to get a simple
// progress indicator.

#[test]
#[ignore]
fn bdd_test_depth2_3vars_t1() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 1);
    let vars = bdd_3_vars::<BDDFunction>(&mref);
    test_depth2_3vars_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn bdd_test_depth2_3vars_t2() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    let vars = bdd_3_vars::<BDDFunction>(&mref);
    test_depth2_3vars_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn bcdd_test_depth2_3vars_t1() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 1);
    let vars = bdd_3_vars::<BCDDFunction>(&mref);
    test_depth2_3vars_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn bcdd_test_depth2_3vars_t2() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    let vars = bdd_3_vars::<BCDDFunction>(&mref);
    test_depth2_3vars_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn zbdd_test_depth2_3vars_t1() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 1);
    let (singletons, vars) = zbdd_3_singletons_vars(&mref);
    test_depth2_3vars(&mref, &vars, &singletons);
}

#[test]
#[ignore]
fn zbdd_test_depth2_3vars_t2() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    let (singletons, vars) = zbdd_3_singletons_vars(&mref);
    test_depth2_3vars(&mref, &vars, &singletons);
}
