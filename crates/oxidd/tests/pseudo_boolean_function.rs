//! Tests for `PseudoBooleanFunction` implementations

use oxidd::mtbdd::MTBDDFunction;
use oxidd::mtbdd::terminal::I64;
use oxidd::util::AllocResult;
use oxidd::{Function, Manager, ManagerRef, PseudoBooleanFunction, VarNo};

// spell-checker:ignore mref

const NUM_VARS: VarNo = 3;

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
