#![cfg(any(feature = "bcdd", feature = "bdd", feature = "zbdd"))]

// spell-checker:ignore nvars,mref

use oxidd::util::AllocResult;
use oxidd::{
    BooleanFunction, HasLevel, HasWorkers, InnerNode, LevelNo, Manager, ManagerRef, VarNo,
};
use oxidd_core::function::{INodeOfFunc, RulesOfFunc};
use oxidd_core::{LevelSwapper, LevelView};
use oxidd_reorder::set_var_order;

#[cfg(feature = "bcdd")]
use oxidd::bcdd::BCDDFunction;
#[cfg(feature = "bdd")]
use oxidd::bdd::BDDFunction;
#[cfg(feature = "zbdd")]
use oxidd::zbdd::ZBDDFunction;

#[cfg(feature = "bdd")]
#[test]
fn reverse_empty_bdd() {
    let mref = oxidd::bdd::new_manager(1024, 128, 2);
    mref.with_manager_exclusive(|manager| {
        let nvars = if cfg!(miri) { 128 } else { 1024 };
        let mut order = Vec::from_iter(manager.add_vars(nvars));

        set_var_order(manager, &order);
        assert!((0..nvars).all(|i| manager.level_to_var(i) == i));

        order.reverse();
        set_var_order(manager, &order);
        assert!((0..nvars).all(|i| manager.level_to_var(i) == nvars - i - 1));
    });
}

fn check_order<M: Manager>(manager: &M, order: &[VarNo]) {
    assert_eq!(manager.num_levels() as usize, order.len());
    for (i, &v) in order.iter().enumerate() {
        assert_eq!(manager.level_to_var(i as LevelNo), v);
    }
}

fn reorder_nonempty<B: BooleanFunction>(mref: B::ManagerRef) -> AllocResult<()>
where
    for<'id> B::Manager<'id>: HasWorkers,
    for<'id> INodeOfFunc<'id, B>: HasLevel,
    for<'id> RulesOfFunc<'id, B>: LevelSwapper<B::Manager<'id>>,
{
    mref.with_manager_exclusive(|manager| manager.add_vars(5));

    let [x1, x3, x4, f] = mref.with_manager_shared(|manager| {
        let x1 = B::var(manager, 1)?;
        let x3 = B::var(manager, 3)?;
        let x4 = B::var(manager, 4)?;
        let f = x1.ite(&x3, &x4)?;
        Ok([x1, x3, x4, f])
    })?;

    let check = |order| {
        mref.with_manager_shared(|manager| {
            // order
            check_order(manager, order);

            // structure
            for level in manager.levels() {
                let i = level.level_no();
                for e in level.iter() {
                    let node = manager.get_node(e).unwrap_inner();
                    assert!(node.children().all(|e| manager.get_node(e).level() > i));
                }
            }
        });

        // semantics
        for a in [false, true] {
            assert_eq!(x1.eval([(1, a)]), a);
            assert_eq!(x3.eval([(3, a)]), a);
            assert_eq!(x4.eval([(4, a)]), a);
            for b in [false, true] {
                for c in [false, true] {
                    assert_eq!(f.eval([(1, a), (3, b), (4, c)]), if a { b } else { c });
                }
            }
        }
    };

    let reorder = |order| mref.with_manager_exclusive(|manager| set_var_order(manager, order));

    check(&[0, 1, 2, 3, 4]);
    reorder(&[2, 3, 1]);
    check(&[0, 2, 3, 1, 4]);
    reorder(&[1, 4, 3]);
    check(&[0, 2, 1, 4, 3]);
    reorder(&[4, 1]);
    check(&[0, 2, 4, 1, 3]);
    reorder(&[3, 1, 4]);
    check(&[0, 2, 3, 1, 4]);

    Ok(())
}

#[cfg(feature = "bcdd")]
#[test]
fn reorder_nonempty_bcdd() -> AllocResult<()> {
    reorder_nonempty::<BCDDFunction>(oxidd::bcdd::new_manager(1024, 128, 2))
}

#[cfg(feature = "bdd")]
#[test]
fn reorder_nonempty_bdd() -> AllocResult<()> {
    reorder_nonempty::<BDDFunction>(oxidd::bdd::new_manager(1024, 128, 2))
}

#[cfg(feature = "zbdd")]
#[test]
fn reorder_nonempty_zbdd() -> AllocResult<()> {
    reorder_nonempty::<ZBDDFunction>(oxidd::zbdd::new_manager(1024, 128, 2))
}

#[cfg(feature = "bdd")]
#[test]
fn reorder_drops_child_referenced_only_by_parent() -> AllocResult<()> {
    let mref = oxidd::bdd::new_manager(1024, 128, 1);
    mref.with_manager_exclusive(|manager| manager.add_vars(2));

    let _conj = mref.with_manager_shared(|manager| {
        let x0 = BDDFunction::var(manager, 0)?;
        let x1 = BDDFunction::var(manager, 1)?;
        x0.and(&x1)
    })?;
    // Only `conj` is still alive; the node for `x1 ∧ x2` is referenced by the
    // root node of `conj` alone.

    mref.with_manager_exclusive(|manager| set_var_order(manager, &[1, 0]));
    Ok(())
}
