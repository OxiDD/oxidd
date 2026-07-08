//! Tests for `PseudoBooleanFunction` implementations

use oxidd::mtbdd::MTBDDFunction;
use oxidd::mtbdd::terminal::I64;
use oxidd::util::AllocResult;
use oxidd::{Function, Manager, ManagerRef, PseudoBooleanFunction, VarNo};

// spell-checker:ignore mref,nvars

const NUM_VARS: VarNo = 3;

/// Evaluate `fun` at every one of the `2^NUM_VARS` valuations and return the
/// results in a canonical order (`i`-th bit of the index selects variable `i`).
fn eval_all(fun: &MTBDDFunction<I64>) -> Vec<I64> {
    (0..(1u32 << NUM_VARS))
        .map(|assignment| fun.eval((0..NUM_VARS).map(|v| (v, (assignment >> v) & 1 != 0))))
        .collect()
}

fn setup() -> (oxidd::mtbdd::MTBDDManagerRef<I64>, Vec<MTBDDFunction<I64>>) {
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

/// A positive (`polarity == true`) or negative literal for `var`, encoded as
/// a 0-1-valued `MTBDDFunction`.
fn literal(
    one: &MTBDDFunction<I64>,
    var: &MTBDDFunction<I64>,
    polarity: bool,
) -> AllocResult<MTBDDFunction<I64>> {
    if polarity {
        Ok(var.clone())
    } else {
        one.sub(var)
    }
}

/// The conjunction (cube) of `literal(one, vars[v], b)` for each `(v, b)` in
/// `assignment`.
fn cube(
    one: &MTBDDFunction<I64>,
    vars: &[MTBDDFunction<I64>],
    assignment: &[(VarNo, bool)],
) -> AllocResult<MTBDDFunction<I64>> {
    let mut result = one.clone();
    for &(v, b) in assignment {
        result = result.mul(&literal(one, &vars[v as usize], b)?)?;
    }
    Ok(result)
}

/// `assignment` with the variables named in `fixed` overridden to their fixed
/// values.
fn apply_fixed(assignment: u32, fixed: &[(VarNo, bool)]) -> Vec<(VarNo, bool)> {
    let mut bits: Vec<(VarNo, bool)> = (0..NUM_VARS)
        .map(|v| (v, (assignment >> v) & 1 != 0))
        .collect();
    for &(v, b) in fixed {
        bits[v as usize] = (v, b);
    }
    bits
}

#[test]
fn restrict_matches_manual_cofactor() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let one = MTBDDFunction::constant(manager, I64::Num(1))?;

        let functions = vec![
            vars[0].add(&vars[1])?.sub(&vars[2])?,
            vars[0].mul(&vars[1])?.add(&vars[2])?,
            vars[0].add(&vars[1])?.add(&vars[2])?,
        ];

        let fixed_sets: Vec<Vec<(VarNo, bool)>> = vec![
            vec![],
            vec![(0, true)],
            vec![(0, false)],
            vec![(1, true), (2, false)],
            vec![(0, false), (1, true), (2, true)],
        ];

        for f in &functions {
            for fixed in &fixed_sets {
                let c = cube(&one, &vars, fixed)?;
                let restricted = f.restrict(&c)?;

                for assignment in 0..(1u32 << NUM_VARS) {
                    let free_args = (0..NUM_VARS).map(|v| (v, (assignment >> v) & 1 != 0));
                    let got = restricted.eval(free_args);
                    let expect = f.eval(apply_fixed(assignment, fixed));
                    assert_eq!(
                        got, expect,
                        "mismatch for fixed={fixed:?}, assignment={assignment:#05b}"
                    );
                }
            }
        }
        Ok(())
    })
}

#[test]
fn restrict_by_true_cube_is_identity() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let f = vars[0].add(&vars[1])?;
        let empty_cube = MTBDDFunction::constant(manager, I64::Num(1))?;
        assert!(f.restrict(&empty_cube)? == f);
        Ok(())
    })
}

#[test]
fn restrict_of_constant_is_identity() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let f = MTBDDFunction::constant(manager, I64::Num(42))?;
        let one = MTBDDFunction::constant(manager, I64::Num(1))?;
        let c = cube(&one, &vars, &[(0, true), (1, false)])?;
        assert!(f.restrict(&c)? == f);
        Ok(())
    })
}

#[test]
fn restrict_shrinks_node_count() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let one = MTBDDFunction::constant(manager, I64::Num(1))?;
        let f = vars[0].add(&vars[1])?.add(&vars[2])?;
        let before = f.node_count();

        let c = cube(&one, &vars, &[(0, true)])?;
        let restricted = f.restrict(&c)?;
        assert!(
            restricted.node_count() < before,
            "expected fewer nodes after fixing x0: before={before}, after={}",
            restricted.node_count()
        );

        // Fixing all variables must yield a single terminal node.
        let full = cube(&one, &vars, &[(0, true), (1, false), (2, true)])?;
        let fully_restricted = f.restrict(&full)?;
        assert_eq!(fully_restricted.node_count(), 1);

        let expected_value = f.eval([(0, true), (1, false), (2, true)]);
        let expected = MTBDDFunction::constant(manager, expected_value)?;
        assert!(fully_restricted == expected);
        Ok(())
    })
}

/// The "textbook" definition of `ite` in terms of the other arithmetic
/// operators, i.e., the workaround that [`PseudoBooleanFunction::ite_edge()`]
/// obsoletes for implementations providing a dedicated single-pass
/// implementation.
fn naive_ite<'id>(
    manager: &<MTBDDFunction<I64> as Function>::Manager<'id>,
    f: &MTBDDFunction<I64>,
    g: &MTBDDFunction<I64>,
    h: &MTBDDFunction<I64>,
) -> AllocResult<MTBDDFunction<I64>> {
    let one = MTBDDFunction::constant(manager, I64::Num(1))?;
    let keep_h = one.sub(f)?;
    h.mul(&keep_h)?.add(&f.mul(g)?)
}

#[test]
#[cfg_attr(miri, ignore)]
fn ite_matches_naive_definition() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let c = |v: i64| MTBDDFunction::constant(manager, I64::Num(v));

        // A representative set of 0-1-valued conditions (constants and
        // variable-based indicators, combined via arithmetic so that the
        // result stays within {0, 1}).
        let conditions = vec![
            c(0)?,
            c(1)?,
            vars[0].clone(),
            // x0 * x1 == x0 AND x1
            vars[0].mul(&vars[1])?,
            // x0 + x1 - x0*x1 == x0 OR x1
            vars[0].add(&vars[1])?.sub(&vars[0].mul(&vars[1])?)?,
        ];

        // Note: NaN/±∞ branches are deliberately excluded here. The naive
        // arithmetic definition evaluates *both* branches and combines them
        // via `0 * x`, which is not the identity in this semiring for NaN or
        // ±∞ (e.g. `0 * PlusInf == PlusInf`, see `I64`'s `Mul` impl) — so it
        // would incorrectly "leak" the untaken branch's value. The real
        // `ite` never evaluates the untaken branch at all; see
        // `ite_ignores_untaken_branch_value` below for that distinction.
        let branches = vec![
            c(0)?,
            c(1)?,
            c(-7)?,
            c(42)?,
            vars[1].clone(),
            vars[2].clone(),
            vars[0].add(&vars[2])?,
        ];

        for f in &conditions {
            for g in &branches {
                for h in &branches {
                    let fast = f.ite(g, h)?;
                    let naive = naive_ite(manager, f, g, h)?;
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
        Ok(())
    })
}

#[test]
fn ite_short_circuits_equal_branches() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let g = vars[0].add(&vars[1])?;
        let h = g.clone();
        // Even a condition containing NaN must not disturb the result, since
        // both branches agree unconditionally.
        let f = MTBDDFunction::constant(manager, I64::NaN)?;
        let res = f.ite(&g, &h)?;
        assert!(res == g, "expected the g branch's edge to be reused as-is");
        Ok(())
    })
}

#[test]
fn ite_constant_condition() -> AllocResult<()> {
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let g = vars[0].clone();
        let h = vars[1].clone();
        let tt = MTBDDFunction::constant(manager, I64::Num(1))?;
        let ff = MTBDDFunction::constant(manager, I64::Num(0))?;

        assert!(tt.ite(&g, &h)? == g);
        assert!(ff.ite(&g, &h)? == h);
        Ok(())
    })
}

#[test]
fn ite_ignores_untaken_branch_value() -> AllocResult<()> {
    // Unlike the naive `h*(1-f) + f*g` definition, a real `ite` never
    // evaluates the untaken branch, so a non-finite value there (NaN, or an
    // infinity that would "leak" through `0 * x` in this semiring) must not
    // affect the result. `f` is a variable here (not a constant), so this
    // also exercises the recursive case, not just the terminal shortcuts.
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let good = vars[1].clone();
        for bad_value in [I64::NaN, I64::PlusInf, I64::MinusInf] {
            let bad = MTBDDFunction::constant(manager, bad_value)?;

            // x0=1 selects `good`, `bad` is untaken.
            let res = vars[0].ite(&good, &bad)?;
            assert_eq!(
                res.eval([(0, true), (1, true)]),
                good.eval([(0, true), (1, true)])
            );
            assert_eq!(
                res.eval([(0, true), (1, false)]),
                good.eval([(0, true), (1, false)])
            );

            // x0=0 selects `good`, `bad` is untaken.
            let res = vars[0].ite(&bad, &good)?;
            assert_eq!(
                res.eval([(0, false), (1, true)]),
                good.eval([(0, false), (1, true)])
            );
        }
        Ok(())
    })
}

#[test]
fn ite_overwrite_use_case() -> AllocResult<()> {
    // Mirrors the classical "overwrite a single leaf" use case: given a
    // 0-1-valued `indicator` selecting exactly one path, `ite(indicator,
    // value, seen)` keeps `seen` everywhere else and replaces the value
    // reached by `indicator`'s cube.
    let (mref, vars) = setup();
    mref.with_manager_shared(|manager| {
        let seen = vars[0].add(&vars[1])?.add(&vars[2])?;

        // Indicator of the cube x0=1, x1=0, x2=1.
        let not_x1 = MTBDDFunction::constant(manager, I64::Num(1))?.sub(&vars[1])?;
        let indicator = vars[0].mul(&not_x1)?.mul(&vars[2])?;

        let value = MTBDDFunction::constant(manager, I64::Num(100))?;
        let overwritten = indicator.ite(&value, &seen)?;
        let naive = naive_ite(manager, &indicator, &value, &seen)?;
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

        Ok(())
    })
}
