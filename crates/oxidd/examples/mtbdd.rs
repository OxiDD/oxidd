use oxidd::mtbdd::terminal::I64;
use oxidd::mtbdd::MTBDDFunction;
use oxidd::mtbdd::MTBDDManagerRef;
use oxidd::util::AllocResult;
use oxidd::ManagerRef;
use oxidd::PseudoBooleanFunction;
use oxidd_core::Manager;
use oxidd_dump::dot::dump_all;
use oxidd_dump::visualize::visualize;

fn main() -> AllocResult<()> {
    let manager_ref: MTBDDManagerRef<I64> = oxidd::mtbdd::new_manager(1024, 1024, 1024, 1);
    let [x1, x2, x3, x4] = manager_ref.with_manager_exclusive(|manager| {
        [
            MTBDDFunction::new_var(manager).unwrap(),
            MTBDDFunction::new_var(manager).unwrap(),
            MTBDDFunction::new_var(manager).unwrap(),
            MTBDDFunction::new_var(manager).unwrap(),
        ]
    });

    manager_ref.with_manager_shared(|manager| {
        let c3 = &MTBDDFunction::constant(manager, I64::Num(3))?;
        let res = x1.add(&x2)?.mul(&c3.sub(&x4)?)?;

        manager.gc();

        let file = std::fs::File::create("mtbdd.dot").expect("could not create `tdd.dot`");
        dump_all(
            file,
            manager,
            [(&x1, "x1"), (&x2, "x2"), (&x3, "x3"), (&x4, "x4")],
            [(&res, "(x1 + x2) * (3 - x4)")],
        )
        .expect("dot export failed");

        visualize(
            manager,
            "diagram",
            &[&x1, &x2, &x3, &x4],
            Some(&["x1", "x2", "x3", "x4"]),
            &[&res],
            Some(&["f"]),
            None,
        )
        .expect("visualization failed");

        Ok(())
    })
}
