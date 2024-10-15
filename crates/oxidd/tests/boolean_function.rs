//! Tests for BooleanFunction implementations

#![cfg_attr(miri, allow(unused))]

mod boolean_prop;
mod util;

use std::fmt;

use rustc_hash::FxHashMap;

use oxidd::bcdd::BCDDFunction;
use oxidd::bdd::BDDFunction;
use oxidd::util::OptBool;
use oxidd::zbdd::ZBDDFunction;
use oxidd::zbdd::ZBDDManagerRef;
use oxidd::{
    BooleanFunction, BooleanFunctionQuant, BooleanOperator, BooleanVecSet, Function, FunctionSubst,
    HasWorkers, InnerNode, Manager, ManagerRef, WorkerPool,
};

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
            ZBDDFunction::new_singleton(manager).unwrap(),
            ZBDDFunction::new_singleton(manager).unwrap(),
            ZBDDFunction::empty(manager),
            ZBDDFunction::base(manager),
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
            ZBDDFunction::new_singleton(manager).unwrap(),
            ZBDDFunction::new_singleton(manager).unwrap(),
            ZBDDFunction::empty(manager),
            ZBDDFunction::base(manager),
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
fn bdd_simple_formulas_t1() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 1);
    test_simple_formulas::<BDDFunction>(&mref);
}

#[test]
fn bdd_simple_formulas_t2() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<BDDFunction>(&mref);
}

#[test]
fn bcdd_simple_formulas_t1() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 1);
    test_simple_formulas::<BCDDFunction>(&mref);
}

#[test]
fn bcdd_simple_formulas_t2() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<BCDDFunction>(&mref);
}

#[test]
fn zbdd_simple_formulas_t1() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 1);
    test_simple_formulas::<ZBDDFunction>(&mref);
}

#[test]
fn zbdd_simple_formulas_t2() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    test_simple_formulas::<ZBDDFunction>(&mref);
}

/// Works for BDDs & BCDDs
fn bdd_vars<B: BooleanFunction>(mref: &B::ManagerRef, n: usize) -> Vec<B> {
    mref.with_manager_exclusive(|manager| (0..n).map(|_| B::new_var(manager).unwrap()).collect())
}

fn zbdd_singletons_vars(mref: &ZBDDManagerRef, n: usize) -> (Vec<ZBDDFunction>, Vec<ZBDDFunction>) {
    let singletons: Vec<ZBDDFunction> = mref.with_manager_exclusive(|manager| {
        (0..n)
            .map(|_| ZBDDFunction::new_singleton(manager).unwrap())
            .collect()
    });
    let vars = mref.with_manager_shared(|manager| {
        singletons
            .iter()
            .map(|s| {
                ZBDDFunction::from_edge(
                    manager,
                    oxidd::zbdd::var_boolean_function(manager, s.as_edge(manager)).unwrap(),
                )
            })
            .collect()
    });
    (singletons, vars)
}

#[test]
fn bcdd_restrict() {
    let mref = oxidd::bcdd::new_manager(1024, 1024, 2);
    let vars = bdd_vars::<BCDDFunction>(&mref, 3);
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

fn test_prop_depth2<B: BooleanFunction>(manager: &B::ManagerRef, vars: &[B], var_handles: &[B]) {
    assert_eq!(vars.len(), var_handles.len());
    assert!(vars.len() < 32);
    let mut f = Prop::False;
    let mut progress = Progress::new(38_493_515);
    loop {
        f.build_and_check(manager, vars, var_handles).unwrap();
        progress.step();

        if !f.next::<false>(2, vars.len() as u32) {
            break;
        }
    }
    progress.done();
}

fn test_prop_depth2_quant<B: BooleanFunctionQuant>(
    manager: &B::ManagerRef,
    vars: &[B],
    var_handles: &[B],
) {
    assert_eq!(vars.len(), var_handles.len());
    assert!(vars.len() < 32);
    let mut f = Prop::False;
    let mut progress = Progress::new(208_194_485);
    loop {
        f.build_and_check_quant(manager, vars, var_handles).unwrap();
        progress.step();

        if !f.next::<true>(2, vars.len() as u32) {
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
fn bdd_prop_depth2_3vars_t1() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 1);
    let vars = bdd_vars::<BDDFunction>(&mref, 3);
    test_prop_depth2_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn bdd_prop_depth2_3vars_t2() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    let vars = bdd_vars::<BDDFunction>(&mref, 3);
    test_prop_depth2_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn bcdd_prop_depth2_3vars_t1() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 1);
    let vars = bdd_vars::<BCDDFunction>(&mref, 3);
    test_prop_depth2_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn bcdd_prop_depth2_3vars_t2() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    let vars = bdd_vars::<BCDDFunction>(&mref, 3);
    test_prop_depth2_quant(&mref, &vars, &vars);
}

#[test]
#[ignore]
fn zbdd_prop_depth2_3vars_t1() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 1);
    let (singletons, vars) = zbdd_singletons_vars(&mref, 3);
    test_prop_depth2(&mref, &vars, &singletons);
}

#[test]
#[ignore]
fn zbdd_prop_depth2_3vars_t2() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    let (singletons, vars) = zbdd_singletons_vars(&mref, 3);
    test_prop_depth2(&mref, &vars, &singletons);
}

/// Explicit representation of a Boolean function (a column in a truth table)
type ExplicitBFunc = u32;

/// Test operations on all Boolean function for a given variable set
struct TestAllBooleanFunctions<'a, B: BooleanFunction> {
    mref: &'a B::ManagerRef,
    vars: &'a [B],
    var_handles: &'a [B],
    /// Stores all possible Boolean functions over `vars.len()` variables
    boolean_functions: Vec<B>,
    /// Map from Boolean functions as decision diagrams to their explicit
    /// (truth table) representations
    dd_to_boolean_func: FxHashMap<B, ExplicitBFunc>,
    /// Map from variables (`0..vars.len()`) to Boolean functions
    ///
    /// Example for three variables: `[0b01010101, 0b00110011, 0b00001111]`
    var_functions: Vec<ExplicitBFunc>,
}

impl<'a, B: BooleanFunction> TestAllBooleanFunctions<'a, B> {
    #[track_caller]
    fn check(
        &self,
        desc: impl fmt::Display,
        actual: ExplicitBFunc,
        expected: ExplicitBFunc,
        operands: &[ExplicitBFunc],
        sets: &[u32],
    ) {
        if actual != expected {
            let vars = self.vars.len() as u32;
            let mut columns = Vec::with_capacity(operands.len() + 2);
            let op_it = operands.iter().copied();
            if operands.len() <= 3 {
                columns.extend(["f", "g", "h"].iter().map(|n| n.to_string()).zip(op_it))
            } else {
                columns.extend((0..operands.len()).map(|i| format!("f{i}")).zip(op_it))
            };
            columns.push(("expected".to_string(), expected));
            columns.push(("actual".to_string(), actual));
            let table = util::debug::TruthTable {
                vars,
                columns: &columns,
            };
            if sets.is_empty() {
                panic!("Operation {desc} failed\n\n{table}");
            } else {
                panic!(
                    "Operation {desc} failed\n\n{table}\nwith {:?}",
                    util::debug::SetList { vars, sets }
                );
            }
        }
    }

    #[track_caller]
    fn panic(&self, desc: impl fmt::Display, columns: &[(impl AsRef<str>, ExplicitBFunc)]) -> ! {
        panic!(
            "{desc}\n\n{}",
            util::debug::TruthTable {
                vars: self.vars.len() as u32,
                columns,
            }
        )
    }

    #[track_caller]
    fn check_cond(
        &self,
        cond: bool,
        desc: impl fmt::Display,
        columns: &[(impl AsRef<str>, ExplicitBFunc)],
    ) {
        if !cond {
            self.panic(desc, columns)
        }
    }

    /// Initialize the test, generating DDs for all Boolean functions for the
    /// given variable set. `vars` are the Boolean functions representing the
    /// variables identified by `var_handles`. For BDDs, the two coincide, but
    /// not for ZBDDs.
    pub fn init(mref: &'a B::ManagerRef, vars: &'a [B], var_handles: &'a [B]) -> Self {
        assert_eq!(vars.len(), var_handles.len());
        assert!(
            ExplicitBFunc::BITS.ilog2() as usize >= vars.len(),
            "too many variables, only {} are possible",
            ExplicitBFunc::BITS.ilog2()
        ); // actually, only 3 are possible in a feasible amount of time

        let nvars = vars.len() as u32;
        let num_assignments = 1u32 << nvars;
        let num_functions: ExplicitBFunc = 1 << num_assignments;
        // Stores all possible Boolean functions with `nvars` vars
        let mut boolean_functions = Vec::with_capacity(num_functions as usize);
        let mut dd_to_boolean_func =
            FxHashMap::with_capacity_and_hasher(num_functions as usize, Default::default());
        mref.with_manager_shared(|manager| {
            for explicit_f in 0..num_functions {
                // naïve DD construction from the on-set
                let mut f = B::f(manager);
                for assignment in 0..num_assignments {
                    if explicit_f & (1 << assignment) == 0 {
                        continue; // not part of the on-set
                    }
                    let mut cube = B::t(manager);
                    for var in 0..nvars {
                        if assignment & (1 << var) != 0 {
                            cube = cube.and(&vars[var as usize]).unwrap();
                        } else {
                            cube = cube.and(&vars[var as usize].not().unwrap()).unwrap();
                        }
                    }
                    f = f.or(&cube).unwrap();
                }

                // check that evaluating the function yields the desired values
                for assignment in 0..(1u32 << nvars) {
                    let expected = explicit_f & (1 << assignment) != 0;
                    let actual = f.eval(
                        var_handles
                            .iter()
                            .enumerate()
                            .map(|(i, var)| (var, assignment & (1 << i) != 0)),
                    );
                    assert_eq!(actual, expected);
                }

                boolean_functions.push(f.clone());
                let res = dd_to_boolean_func.insert(f, explicit_f);
                assert!(
                    res.is_none(),
                    "two different Boolean functions have the same representation"
                );
            }
        });

        // Example for 3 vars: [0b01010101, 0b00110011, 0b00001111]
        let var_functions: Vec<ExplicitBFunc> = (0..nvars)
            .map(|i| {
                let mut f = 0;
                for assignment in 0..num_assignments {
                    f |= (((assignment >> i) & 1) as ExplicitBFunc) << assignment;
                }
                f
            })
            .collect();

        Self {
            mref,
            vars,
            var_handles,
            boolean_functions,
            dd_to_boolean_func,
            var_functions,
        }
    }

    fn make_var_set(&self, vars: u32) -> B {
        let mut set = self.boolean_functions.last().unwrap().clone();
        for (i, var) in self.vars.iter().enumerate() {
            if vars & (1 << i) != 0 {
                set = set.and(var).unwrap();
            }
        }
        set
    }

    fn make_cube(&self, positive: u32, negative: u32) -> B {
        assert_eq!(positive & negative, 0);

        let mut cube = self.boolean_functions.last().unwrap().clone();
        for (i, var) in self.vars.iter().enumerate() {
            if (positive >> i) & 1 != 0 {
                cube = cube.and(var).unwrap();
            } else if (negative >> i) & 1 != 0 {
                cube = cube.and(&var.not().unwrap()).unwrap();
            }
        }

        cube
    }

    /// Test basic operations on all Boolean functions
    pub fn basic(&self) {
        let nvars = self.vars.len() as u32;
        let num_assignments = 1u32 << nvars;
        let num_functions: ExplicitBFunc = 1 << num_assignments;
        let func_mask = num_functions - 1;

        // false & true
        self.mref.with_manager_shared(|manager| {
            assert!(B::f(manager) == self.boolean_functions[0]);
            assert!(&B::t(manager) == self.boolean_functions.last().unwrap());
        });

        // vars
        for (i, var) in self.vars.iter().enumerate() {
            let mut expected = 0;
            for assignment in 0..num_assignments {
                expected |= ((assignment >> i) & 1) << assignment;
            }
            let actual = self.dd_to_boolean_func[var];
            assert_eq!(actual, expected);
        }

        // arity >= 1
        for (f_explicit, f) in self.boolean_functions.iter().enumerate() {
            let f_explicit = f_explicit as ExplicitBFunc;

            // not
            let expected = !f_explicit & func_mask;
            let actual = self.dd_to_boolean_func[&f.not().unwrap()];
            self.check("¬f", actual, expected, &[f_explicit], &[]);

            // arity >= 2
            for (g_explicit, g) in self.boolean_functions.iter().enumerate() {
                let g_explicit = g_explicit as ExplicitBFunc;

                // and
                let expected = f_explicit & g_explicit;
                let actual = self.dd_to_boolean_func[&f.and(g).unwrap()];
                self.check("f ∧ g", actual, expected, &[f_explicit, g_explicit], &[]);

                // or
                let expected = f_explicit | g_explicit;
                let actual = self.dd_to_boolean_func[&f.or(g).unwrap()];
                self.check("f ∨ g", actual, expected, &[f_explicit, g_explicit], &[]);

                // xor
                let expected = f_explicit ^ g_explicit;
                let actual = self.dd_to_boolean_func[&f.xor(g).unwrap()];
                self.check("f ⊕ g", actual, expected, &[f_explicit, g_explicit], &[]);

                // equiv
                let expected = !(f_explicit ^ g_explicit) & func_mask;
                let actual = self.dd_to_boolean_func[&f.equiv(g).unwrap()];
                self.check("f ↔ g", actual, expected, &[f_explicit, g_explicit], &[]);

                // nand
                let expected = !(f_explicit & g_explicit) & func_mask;
                let actual = self.dd_to_boolean_func[&f.nand(g).unwrap()];
                self.check("f ⊼ g", actual, expected, &[f_explicit, g_explicit], &[]);

                // nor
                let expected = !(f_explicit | g_explicit) & func_mask;
                let actual = self.dd_to_boolean_func[&f.nor(g).unwrap()];
                self.check("f ⊽ g", actual, expected, &[f_explicit, g_explicit], &[]);

                // implication
                let expected = (!f_explicit | g_explicit) & func_mask;
                let actual = self.dd_to_boolean_func[&f.imp(g).unwrap()];
                self.check("f → g", actual, expected, &[f_explicit, g_explicit], &[]);

                // strict implication
                let expected = !f_explicit & g_explicit;
                let actual = self.dd_to_boolean_func[&f.imp_strict(g).unwrap()];
                self.check("f < g", actual, expected, &[f_explicit, g_explicit], &[]);

                // arity >= 3
                for (h_explicit, h) in self.boolean_functions.iter().enumerate() {
                    let h_explicit = h_explicit as ExplicitBFunc;

                    // ite
                    self.check(
                        "if f { g } else { h }",
                        self.dd_to_boolean_func[&f.ite(g, h).unwrap()],
                        (f_explicit & g_explicit) | (!f_explicit & h_explicit),
                        &[f_explicit, g_explicit, h_explicit],
                        &[],
                    );
                }
            }

            for assignment in 0..num_assignments {
                // At first, we test `pick_cube()` and `pick_cube_dd()`, specifically that:
                //
                // - The closure is called as specified (at most once for each level, `edge`
                //   points to a node at that level)
                // - The special case where `f` is ⊥ is handled correctly
                // - In all other cases, the results of `pick_cube()` and `pick_cube_dd()`
                //   * are equivalent (and the result of `pick_cube_dd()` can thus be
                //     represented as a conjunction of literals)
                //   * imply the function
                // - If the choice function was called for some level, the respective choice is
                //   taken into account
                // - Whenever there is a choice for a variable, either the choice function is
                //   called or the cube is independent of that variable (don't care)
                //
                // We do not (yet) check that don't cares are preserved, since that would
                // require taking the variable order into account. It is not clear to me whether
                // this would work for kinds of DDs without a linear variable order.

                let mut choice_requested = 0u32;
                let cube = f.pick_cube([], |manager, edge, level| {
                    manager
                        .get_node(edge)
                        .unwrap_inner()
                        .assert_level_matches(level);
                    if choice_requested & (1 << level) != 0 {
                        panic!("choice requested twice for x{level}");
                    } else {
                        choice_requested |= 1 << level;
                    }
                    assignment & (1u32 << level) != 0
                });
                let mut choice_requested_sym = 0u32;
                let dd_cube = f
                    .pick_cube_dd(|manager, edge, level| {
                        manager
                            .get_node(edge)
                            .unwrap_inner()
                            .assert_level_matches(level);
                        if choice_requested_sym & (1 << level) != 0 {
                            panic!("choice requested twice for x{level}");
                        } else {
                            choice_requested_sym |= 1 << level;
                        }
                        assignment & (1u32 << level) != 0
                    })
                    .unwrap();

                let actual = self.dd_to_boolean_func[&dd_cube];
                if f_explicit == 0 {
                    self.check("f.pick_cube_dd(..)", actual, 0, &[f_explicit], &[]);
                    assert_eq!(cube, None);
                    assert_eq!(choice_requested, 0);
                } else {
                    let cube =
                        cube.expect("f.pick_cube(..) returned None for a satisfiable function");
                    assert_eq!(cube.len(), nvars as usize);
                    self.check_cond(
                        actual & !f_explicit == 0,
                        "f.pick_cube_dd(..) does not imply f",
                        &[("f", f_explicit), ("f.pick_cube_dd(..)", actual)],
                    );

                    let mut cube_func = func_mask;
                    for (var, (&literal, &var_func)) in
                        cube.iter().zip(&self.var_functions).enumerate()
                    {
                        if choice_requested & (1 << var) != 0 {
                            assert_eq!(
                                literal,
                                OptBool::from(assignment & (1 << var) != 0),
                                "If a choice was requested, the cube should reflect the choice"
                            );
                        } else if literal != OptBool::None {
                            // The decision should have been enforced, i.e., the
                            // function c' obtained flipping the literal in the
                            // cube does not imply f:
                            let flipped = if literal == OptBool::True {
                                actual >> (1 << var)
                            } else {
                                actual << (1 << var)
                            };

                            self.check_cond(
                                flipped & !f_explicit != 0,
                                format_args!("f.pick_cube_dd(..) should call the choice function or leave a don't care for x{var}"),
                                &[
                                    ("f", f_explicit),
                                    ("f.pick_cube_dd(..)", actual),
                                    ("cube with flipped literal", flipped),
                                ],
                            );
                        }
                        match literal {
                            OptBool::False => cube_func &= !var_func,
                            OptBool::True => cube_func &= var_func,
                            _ => {}
                        }
                    }

                    self.check_cond(
                        cube_func & !f_explicit == 0,
                        "f.pick_cube(..) does not imply f",
                        &[("f", f_explicit), ("f.pick_cube(..)", actual)],
                    );

                    self.check(
                        "f.pick_cube_dd(choice_fn) does not agree with f.pick_cube([], choice_fn)",
                        actual,
                        cube_func,
                        &[f_explicit],
                        &[assignment],
                    );
                }

                assert_eq!(
                    choice_requested, choice_requested_sym,
                    "pick_cube should request a choice iff pick_cube_dd requests a choice"
                );
            }

            for pos in 0..num_assignments {
                for neg in 0..num_assignments {
                    if pos & neg != 0 {
                        continue;
                    }

                    let literal_set = self.make_cube(pos, neg);
                    let literal_set_explicit = self.dd_to_boolean_func[&literal_set];
                    let actual =
                        self.dd_to_boolean_func[&f.pick_cube_dd_set(&literal_set).unwrap()];

                    if f_explicit == 0 {
                        self.check(
                            "f.pick_cube_dd_set(g)",
                            actual,
                            0,
                            &[f_explicit, literal_set_explicit],
                            &[],
                        );
                    } else {
                        self.check_cond(
                            actual & !f_explicit == 0,
                            "f.pick_cube_dd_set(literal_set) does not imply f",
                            &[
                                ("f", f_explicit),
                                ("literal_set", literal_set_explicit),
                                ("f.pick_cube_dd_set(literal_set)", actual),
                            ],
                        );

                        for (var, var_func) in self.var_functions.iter().enumerate() {
                            if (actual & var_func) >> (1 << var) == actual & !var_func {
                                continue; // var is don't care
                            }
                            let selected = if actual & var_func == 0 {
                                // var is set to false
                                if pos & (1 << var) == 0 {
                                    continue; // was not requested to be true
                                }
                                false
                            } else if actual & !var_func == 0 {
                                // var is set to true
                                if neg & (1 << var) == 0 {
                                    continue; // was not requested to be false
                                }
                                true
                            } else {
                                self.panic(
                                    format_args!("f.pick_cube_dd_set(literal_set) is not a cube (checking x{var})"),
                                    &[
                                        ("f", f_explicit),
                                        ("literal_set", literal_set_explicit),
                                        ("f.pick_cube_dd_set(literal_set)", actual),
                                    ],
                                )
                            };

                            // If the variable was selected to be the opposite
                            // of the request, then the reason must be that the
                            // cube would not have implied the function. We test this now.
                            let flipped = if selected {
                                actual >> (1 << var)
                            } else {
                                actual << (1 << var)
                            };

                            self.check_cond(
                                flipped & !f_explicit != 0,
                                format_args!("f.pick_cube_dd_set(literal_set) does not follow the requirements from literal_set (selecting {selected} for x{var})"),
                                &[
                                    ("f", f_explicit),
                                    ("literal_set", literal_set_explicit),
                                    ("f.pick_cube_dd_set(literal_set)", actual),
                                    ("flipped", flipped),
                                ],
                            );
                        }
                    }
                }
            }
        }
    }
}

impl<B: BooleanFunction + FunctionSubst> TestAllBooleanFunctions<'_, B> {
    /// Test all possible substitutions
    pub fn subst(&self) {
        self.subst_rec(&mut vec![None; self.vars.len()], 0);
    }

    fn subst_rec(&self, replacements: &mut [Option<ExplicitBFunc>], current_var: u32) {
        debug_assert_eq!(replacements.len(), self.vars.len());
        if (current_var as usize) < self.vars.len() {
            replacements[current_var as usize] = None;
            self.subst_rec(replacements, current_var + 1);
            for f in 0..self.boolean_functions.len() as ExplicitBFunc {
                replacements[current_var as usize] = Some(f);
                self.subst_rec(replacements, current_var + 1);
            }
        } else {
            let nvars = self.vars.len() as u32;
            let num_assignments = 1u32 << nvars;

            let mut subst_vars = Vec::with_capacity(self.vars.len());
            let mut subst_repl = Vec::with_capacity(self.vars.len());
            for (i, &repl) in replacements.iter().enumerate() {
                if let Some(repl) = repl {
                    subst_vars.push(self.var_handles[i].clone());
                    subst_repl.push(self.boolean_functions[repl as usize].clone());
                }
            }
            let subst = oxidd_core::util::Subst::new(&subst_vars, &subst_repl);

            for (f_explicit, f) in self.boolean_functions.iter().enumerate() {
                let f_explicit = f_explicit as ExplicitBFunc;

                let mut expected = 0;
                // To compute the expected truth table, we first compute a
                // mapped assignment that we look up in the truth table for `f`
                for assignment in 0..num_assignments {
                    let mut mapped_assignment = 0;
                    for (var, repl) in replacements.iter().enumerate() {
                        let val = if let Some(repl) = repl {
                            // replacement function evaluated for `assignment`
                            repl >> assignment
                        } else {
                            // `var` is set in `assignment`?
                            assignment >> var
                        } & 1;
                        mapped_assignment |= val << var;
                    }
                    expected |= ((f_explicit >> mapped_assignment) & 1) << assignment;
                }

                let actual = self.dd_to_boolean_func[&f.substitute(&subst).unwrap()];
                assert_eq!(actual, expected);
            }
        }
    }
}

impl<B: BooleanFunctionQuant> TestAllBooleanFunctions<'_, B> {
    /// Test quantification operations on all Boolean function
    pub fn quant(&self) {
        let nvars = self.vars.len() as ExplicitBFunc;
        let num_assignments = 1u32 << nvars;
        let num_functions: ExplicitBFunc = 1 << num_assignments;
        let func_mask = num_functions - 1;

        // restrict
        for pos in 0..num_assignments {
            for neg in 0..num_assignments {
                if pos & neg != 0 {
                    continue; // positive and negative set must be disjoint
                }

                let dd_literal_set = self.make_cube(pos, neg);

                for (f_explicit, f) in self.boolean_functions.iter().enumerate() {
                    let f_explicit = f_explicit as ExplicitBFunc;

                    let mut expected = 0;
                    for assignment in 0..num_assignments {
                        let assignment_restricted = (assignment | pos) & !neg;
                        expected |= ((f_explicit >> assignment_restricted) & 1) << assignment;
                    }

                    let actual = self.dd_to_boolean_func[&f.restrict(&dd_literal_set).unwrap()];
                    assert_eq!(actual, expected);
                }
            }
        }

        // quantification
        let mut assignment_to_mask: Vec<ExplicitBFunc> = vec![0; num_assignments as usize];
        for var_set in 0..num_assignments {
            let dd_var_set = self.make_var_set(var_set);

            // precompute `assignment_to_mask`
            for (assignment, mask) in assignment_to_mask.iter_mut().enumerate() {
                let mut tmp: ExplicitBFunc = func_mask;
                for (i, func) in self.var_functions.iter().copied().enumerate() {
                    if (var_set >> i) & 1 == 0 {
                        tmp &= if (assignment >> i) & 1 == 0 {
                            !func
                        } else {
                            func
                        };
                    }
                }
                *mask = tmp;
            }

            for (f_explicit, f) in self.boolean_functions.iter().enumerate() {
                let f_explicit = f_explicit as ExplicitBFunc;

                let mut exist_expected: ExplicitBFunc = 0;
                let mut forall_expected: ExplicitBFunc = 0;
                let mut unique_expected: ExplicitBFunc = 0;
                for (assignment, mask) in assignment_to_mask.iter().copied().enumerate() {
                    let exist_bit = f_explicit & mask != 0; // or of all bits under mask
                    let forall_bit = f_explicit & mask == mask; // and of all bits under mask
                    let unique_bit = (f_explicit & mask).count_ones() & 1; // xor of all bits under mask
                    exist_expected |= (exist_bit as ExplicitBFunc) << assignment;
                    forall_expected |= (forall_bit as ExplicitBFunc) << assignment;
                    unique_expected |= (unique_bit as ExplicitBFunc) << assignment;
                }

                self.check(
                    "∃v. f",
                    self.dd_to_boolean_func[&f.exist(&dd_var_set).unwrap()],
                    exist_expected,
                    &[f_explicit],
                    &[var_set],
                );

                self.check(
                    "∀v. f",
                    self.dd_to_boolean_func[&f.forall(&dd_var_set).unwrap()],
                    forall_expected,
                    &[f_explicit],
                    &[var_set],
                );

                self.check(
                    "∃!v. f",
                    self.dd_to_boolean_func[&f.unique(&dd_var_set).unwrap()],
                    unique_expected,
                    &[f_explicit],
                    &[var_set],
                );

                // Apply and quantification algorithms. Here, we only compare the naive and
                // optimized implementations.
                let f_explicit = f_explicit as ExplicitBFunc;
                for (g_explicit, g) in self.boolean_functions.iter().enumerate() {
                    let g_explicit = g_explicit as ExplicitBFunc;
                    use BooleanOperator::*;
                    for op in [And, Or, Xor, Equiv, Nand, Nor, Imp, ImpStrict] {
                        let (inner, inner_symbol) = match op {
                            And => (f.and(g).unwrap(), "∧"),
                            Or => (f.or(g).unwrap(), "∨"),
                            Xor => (f.xor(g).unwrap(), "⊕"),
                            Equiv => (f.equiv(g).unwrap(), "↔"),
                            Nand => (f.nand(g).unwrap(), "⊼"),
                            Nor => (f.nor(g).unwrap(), "⊽"),
                            Imp => (f.imp(g).unwrap(), "→"),
                            ImpStrict => (f.imp_strict(g).unwrap(), "<"),
                        };

                        self.check(
                            format_args!("∃v. f {inner_symbol} g"),
                            self.dd_to_boolean_func[&f.apply_exist(op, g, &dd_var_set).unwrap()],
                            self.dd_to_boolean_func[&inner.exist(&dd_var_set).unwrap()],
                            &[f_explicit, g_explicit],
                            &[var_set],
                        );

                        self.check(
                            format_args!("∀v. f {inner_symbol} g"),
                            self.dd_to_boolean_func[&f.apply_forall(op, g, &dd_var_set).unwrap()],
                            self.dd_to_boolean_func[&inner.forall(&dd_var_set).unwrap()],
                            &[f_explicit, g_explicit],
                            &[var_set],
                        );

                        self.check(
                            format_args!("∃!v. f {inner_symbol} g"),
                            self.dd_to_boolean_func[&f.apply_unique(op, g, &dd_var_set).unwrap()],
                            self.dd_to_boolean_func[&inner.unique(&dd_var_set).unwrap()],
                            &[f_explicit, g_explicit],
                            &[var_set],
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn bdd_all_boolean_functions_2vars_t1() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 1);
    let vars = bdd_vars::<BDDFunction>(&mref, 2);
    let test = TestAllBooleanFunctions::init(&mref, &vars, &vars);
    test.basic();
    test.subst();
    test.quant();
}

#[test]
#[cfg_attr(miri, ignore)]
fn bdd_all_boolean_functions_2vars_t2() {
    let mref = oxidd::bdd::new_manager(65536, 1024, 2);
    mref.workers().set_split_depth(Some(u32::MAX));
    let vars = bdd_vars::<BDDFunction>(&mref, 2);
    let test = TestAllBooleanFunctions::init(&mref, &vars, &vars);
    test.basic();
    test.subst();
    test.quant();
}

#[test]
#[cfg_attr(miri, ignore)]
fn bcdd_all_boolean_functions_2vars_t1() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 1);
    let vars = bdd_vars::<BCDDFunction>(&mref, 2);
    let test = TestAllBooleanFunctions::init(&mref, &vars, &vars);
    test.basic();
    test.subst();
    test.quant();
}

#[test]
#[cfg_attr(miri, ignore)]
fn bcdd_all_boolean_functions_2vars_t2() {
    let mref = oxidd::bcdd::new_manager(65536, 1024, 2);
    mref.workers().set_split_depth(Some(u32::MAX));
    let vars = bdd_vars::<BCDDFunction>(&mref, 2);
    let test = TestAllBooleanFunctions::init(&mref, &vars, &vars);
    test.basic();
    test.subst();
    test.quant();
}

#[test]
#[cfg_attr(miri, ignore)]
fn zbdd_all_boolean_functions_2vars_t1() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 1);
    let (singletons, vars) = zbdd_singletons_vars(&mref, 2);
    let test = TestAllBooleanFunctions::init(&mref, &vars, &singletons);
    test.basic();
}

#[test]
#[cfg_attr(miri, ignore)]
fn zbdd_all_boolean_functions_2vars_t2() {
    let mref = oxidd::zbdd::new_manager(65536, 1024, 2);
    mref.workers().set_split_depth(Some(u32::MAX));
    let (singletons, vars) = zbdd_singletons_vars(&mref, 2);
    let test = TestAllBooleanFunctions::init(&mref, &vars, &singletons);
    test.basic();
}
