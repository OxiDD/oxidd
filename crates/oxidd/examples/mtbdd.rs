use oxidd::mtbdd::terminal::I64;
use oxidd::mtbdd::{MTBDDFunction, MTBDDManagerRef};
use oxidd::util::AllocResult;
use oxidd::ManagerRef;
use oxidd::PseudoBooleanFunction;
use oxidd_core::Manager;
use oxidd_dump::dot::dump_all;
use oxidd_dump::Visualizer;

fn main() -> AllocResult<()> {
    let manager_ref: MTBDDManagerRef<I64> = oxidd::mtbdd::new_manager(1024, 1024, 1024, 1);
    let [x1, x2, x4] = manager_ref.with_manager_exclusive(|manager| {
        manager.add_named_vars(["x1", "x2", "x3", "x4"]).unwrap();
        Ok([
            MTBDDFunction::var(manager, 0)?,
            MTBDDFunction::var(manager, 1)?,
            MTBDDFunction::var(manager, 3)?,
        ])
    })?;

    manager_ref.with_manager_shared(|manager| {
        let c3 = &MTBDDFunction::constant(manager, I64::Num(3))?;
        let res = x1.add(&x2)?.mul(&c3.sub(&x4)?)?;

        manager.gc();

        let file = std::fs::File::create("mtbdd.dot").expect("could not create `tdd.dot`");
        dump_all(file, manager, [(&res, "(x1 + x2) * (3 - x4)")]).expect("dot export failed");

        Visualizer::new()
            .add("Sample MTBDD", manager, [&res, c3])
            .serve()
            .ok();

        Ok(())
    })
}
