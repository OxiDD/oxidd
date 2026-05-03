// spell-checker:ignore nvars,mref

use oxidd::{
    bdd::BDDFunction, util::AllocResult, BooleanFunction, InnerNode, LevelNo, Manager, ManagerRef,
    VarNo,
};
use oxidd_core::LevelView;
use oxidd_reorder::set_var_order;

#[test]
fn reverse_empty_bdd() {
    let mref = oxidd::bdd::new_manager(1024, 128, 2);
    mref.with_manager_exclusive(|manager| {
        let nvars = 1024;
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

#[test]
fn reorder_nonempty_bdd() -> AllocResult<()> {
    let mref = oxidd::bdd::new_manager(1024, 128, 2);
    mref.with_manager_exclusive(|manager| manager.add_vars(5));

    let [x1, x3, conj] = mref.with_manager_shared(|manager| {
        let x1 = BDDFunction::var(manager, 1)?;
        let x3 = BDDFunction::var(manager, 3)?;
        let conj = x1.and(&x3)?;
        Ok([x1, x3, conj])
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
                    assert!(node.children().all(|e| manager.get_node(&e).level() > i));
                }
            }
        });

        // semantics
        for x1v in [false, true] {
            assert_eq!(x1.eval([(1, x1v)]), x1v);
            for x3v in [false, true] {
                assert_eq!(conj.eval([(1, x1v), (3, x3v)]), x1v && x3v);
            }
        }
        for x3v in [false, true] {
            assert_eq!(x3.eval([(3, x3v)]), x3v);
        }
    };

    check(&[0, 1, 2, 3, 4]);
    mref.with_manager_exclusive(|manager| set_var_order(manager, &[2, 3, 1]));
    check(&[0, 2, 3, 1, 4]);

    Ok(())
}
