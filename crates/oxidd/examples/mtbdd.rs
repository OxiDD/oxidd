use oxidd::mtbdd::terminal::Int64;
use oxidd::mtbdd::MTBDDFunction;
use oxidd::mtbdd::MTBDDManagerRef;
use oxidd::ManagerRef;
use oxidd::PseudoBooleanFunction;
use oxidd_core::Manager;
use oxidd_dump::dot::dump_all;

fn main() {
    let manager_ref: MTBDDManagerRef<Int64> = oxidd::mtbdd::new_manager(1024, 1024, 1024, 1);
    let [x1, x2, x3, x4] = manager_ref.with_manager_exclusive(|manager| {
        [
            MTBDDFunction::new_var(manager).unwrap(),
            MTBDDFunction::new_var(manager).unwrap(),
            MTBDDFunction::new_var(manager).unwrap(),
            MTBDDFunction::new_var(manager).unwrap(),
        ]
    });

    manager_ref.with_manager_shared(|manager| {
        let res = x1
            .add(&x2)
            .unwrap()
            .mul(
                &MTBDDFunction::constant(manager, Int64::Num(3))
                    .unwrap()
                    .sub(&x4)
                    .unwrap(),
            )
            .unwrap();

        manager.gc();

        let file = std::fs::File::create("mtbdd.dot").expect("could not create `tdd.dot`");
        dump_all(
            file,
            manager,
            [(&x1, "x1"), (&x2, "x2"), (&x3, "x3"), (&x4, "x4")],
            [(&res, "(x1 + x2) * (3 - x4)")],
        )
        .expect("dot export failed");
    });
}
