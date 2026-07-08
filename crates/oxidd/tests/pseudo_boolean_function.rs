//! Tests for `PseudoBooleanFunction` implementations

#![cfg_attr(miri, allow(unused))]

use oxidd::mtbdd::terminal::I64;
use oxidd::mtbdd::MTBDDFunction;
use oxidd::{Function, Manager, ManagerRef, NumberBase, PseudoBooleanFunction, VarNo};

// spell-checker:ignore nvars

const NUM_VARS: VarNo = 3;

/// The "textbook" definition of `ite` in terms of the other arithmetic
/// operators, i.e., the workaround that [`PseudoBooleanFunction::ite_edge()`]
/// obsoletes for implementations providing a dedicated single-pass
/// implementation.
fn naive_ite<'id>(
    manager: &<MTBDDFunction<I64> as Function>::Manager<'id>,
    f: &MTBDDFunction<I64>,
    g: &MTBDDFunction<I64>,
    h: &MTBDDFunction<I64>,
) -> MTBDDFunction<I64> {
    let one = MTBDDFunction::constant(manager, I64::Num(1)).unwrap();
    let keep_h = one.sub(f).unwrap();
    h.mul(&keep_h).unwrap().add(&f.mul(g).unwrap()).unwrap()
}

/// Evaluate `fun` at every one of the `2^NUM_VARS` valuations and return the
/// results in a canonical order (`i`-th bit of the index selects variable
/// `i`).
fn eval_all(fun: &MTBDDFunction<I64>) -> Vec<I64> {
    (0..(1u32 << NUM_VARS))
        .map(|assignment| {
            fun.eval((0..NUM_VARS).map(|v| (v, (assignment >> v) & 1 != 0)))
        })
        .collect()
}

fn setup() -> (
    oxidd::mtbdd::MTBDDManagerRef<I64>,
    Vec<MTBDDFunction<I64>>,
) {
    let mref = oxidd::mtbdd::new_manager(1024, 1024, 1024, 1);
    let vars = mref
        .with_manager_exclusive(|manager| {
            manager
                .add_named_vars((0..NUM_VARS).map(|i| format!("x{i}")))
                .unwrap();
            (0..NUM_VARS)
                .map(|i| MTBDDFunction::var(manager, i))
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap();
    (mref, vars)
}

#[test]
fn ite_matches_naive_definition() {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let c = |v: i64| MTBDDFunction::constant(manager, I64::Num(v)).unwrap();

        // A representative set of 0-1-valued conditions (constants and
        // variable-based indicators, combined via arithmetic so that the
        // result stays within {0, 1}).
        let conditions = vec![
            c(0),
            c(1),
            vars[0].clone(),
            // x0 * x1 == x0 AND x1
            vars[0].mul(&vars[1]).unwrap(),
            // x0 + x1 - x0*x1 == x0 OR x1
            vars[0]
                .add(&vars[1])
                .unwrap()
                .sub(&vars[0].mul(&vars[1]).unwrap())
                .unwrap(),
        ];

        // Note: NaN/±∞ branches are deliberately excluded here. The naive
        // arithmetic definition evaluates *both* branches and combines them
        // via `0 * x`, which is not the identity in this semiring for NaN or
        // ±∞ (e.g. `0 * PlusInf == PlusInf`, see `I64`'s `Mul` impl) — so it
        // would incorrectly "leak" the untaken branch's value. The real
        // `ite` never evaluates the untaken branch at all; see
        // `ite_ignores_untaken_branch_value` below for that distinction.
        let branches = vec![
            c(0),
            c(1),
            c(-7),
            c(42),
            vars[1].clone(),
            vars[2].clone(),
            vars[0].add(&vars[2]).unwrap(),
        ];

        for f in &conditions {
            for g in &branches {
                for h in &branches {
                    let fast = f.ite(g, h).unwrap();
                    let naive = naive_ite(manager, f, g, h);
                    assert_eq!(
                        eval_all(&fast),
                        eval_all(&naive),
                        "ite mismatch for f={:?}, g={:?}, h={:?}",
                        eval_all(f),
                        eval_all(g),
                        eval_all(h)
                    );
                }
            }
        }
    });
}

#[test]
fn ite_short_circuits_equal_branches() {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let g = vars[0].add(&vars[1]).unwrap();
        let h = g.clone();
        // Even a condition containing NaN must not disturb the result, since
        // both branches agree unconditionally.
        let f = MTBDDFunction::constant(manager, I64::NaN).unwrap();
        let res = f.ite(&g, &h).unwrap();
        assert!(res == g, "expected the g branch's edge to be reused as-is");
    });
}

#[test]
fn ite_constant_condition() {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let g = vars[0].clone();
        let h = vars[1].clone();
        let tt = MTBDDFunction::constant(manager, I64::Num(1)).unwrap();
        let ff = MTBDDFunction::constant(manager, I64::Num(0)).unwrap();

        assert!(tt.ite(&g, &h).unwrap() == g);
        assert!(ff.ite(&g, &h).unwrap() == h);
    });
}

#[test]
fn ite_condition_nan_propagates() {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let g = vars[0].clone();
        let h = vars[1].clone();
        let nan = MTBDDFunction::constant(manager, I64::NaN).unwrap();
        let res = nan.ite(&g, &h).unwrap();
        for v in eval_all(&res) {
            assert!(v.is_nan(), "expected NaN, got {v:?}");
        }
    });
}

#[test]
fn ite_ignores_untaken_branch_value() {
    // Unlike the naive `h*(1-f) + f*g` definition, a real `ite` never
    // evaluates the untaken branch, so a non-finite value there (NaN, or an
    // infinity that would "leak" through `0 * x` in this semiring) must not
    // affect the result. `f` is a variable here (not a constant), so this
    // also exercises the recursive case, not just the terminal shortcuts.
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let good = vars[1].clone();
        for bad_value in [I64::NaN, I64::PlusInf, I64::MinusInf] {
            let bad = MTBDDFunction::constant(manager, bad_value).unwrap();

            // x0=1 selects `good`, `bad` is untaken.
            let res = vars[0].ite(&good, &bad).unwrap();
            assert_eq!(
                res.eval([(0, true), (1, true)]),
                good.eval([(0, true), (1, true)])
            );
            assert_eq!(
                res.eval([(0, true), (1, false)]),
                good.eval([(0, true), (1, false)])
            );

            // x0=0 selects `good`, `bad` is untaken.
            let res = vars[0].ite(&bad, &good).unwrap();
            assert_eq!(
                res.eval([(0, false), (1, true)]),
                good.eval([(0, false), (1, true)])
            );
        }
    });
}

#[test]
fn ite_overwrite_use_case() {
    // Mirrors the classical "overwrite a single leaf" use case: given a
    // 0-1-valued `indicator` selecting exactly one path, `ite(indicator,
    // value, seen)` keeps `seen` everywhere else and replaces the value
    // reached by `indicator`'s cube.
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let seen = vars[0]
            .add(&vars[1])
            .unwrap()
            .add(&vars[2])
            .unwrap();

        // Indicator of the cube x0=1, x1=0, x2=1.
        let not_x1 = MTBDDFunction::constant(manager, I64::Num(1))
            .unwrap()
            .sub(&vars[1])
            .unwrap();
        let indicator = vars[0].mul(&not_x1).unwrap().mul(&vars[2]).unwrap();

        let value = MTBDDFunction::constant(manager, I64::Num(100)).unwrap();
        let overwritten = indicator.ite(&value, &seen).unwrap();
        let naive = naive_ite(manager, &indicator, &value, &seen);
        assert_eq!(eval_all(&overwritten), eval_all(&naive));

        for assignment in 0..(1u32 << NUM_VARS) {
            let args = (0..NUM_VARS)
                .map(|v| (v, (assignment >> v) & 1 != 0))
                .collect::<Vec<_>>();
            let got = overwritten.eval(args.iter().copied());
            let expect = if assignment & 0b111 == 0b101 {
                I64::Num(100)
            } else {
                seen.eval(args.iter().copied())
            };
            assert_eq!(got, expect, "mismatch at assignment {assignment:#05b}");
        }
    });
}
