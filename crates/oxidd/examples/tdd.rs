use oxidd::tdd::TDDFunction;
use oxidd::util::AllocResult;
use oxidd::{Manager, ManagerRef, TVLFunction};
use oxidd_dump::dot::dump_all;

fn main() -> AllocResult<()> {
    let manager_ref = oxidd::tdd::new_manager(1024, 1024, 1);
    let (x1, x2, _x3, x4) = manager_ref.with_manager_exclusive(|manager| {
        manager.add_named_vars(["x1", "x2", "x3", "x4"]).unwrap();
        Ok((
            TDDFunction::var(manager, 0)?,
            TDDFunction::var(manager, 1)?,
            TDDFunction::var(manager, 2)?,
            TDDFunction::var(manager, 3)?,
        ))
    })?;

    manager_ref.with_manager_shared(|manager| {
        let res = x1.and(&x2)?.or(&x4)?;

        manager.gc();

        let file = std::fs::File::create("tdd.dot").expect("could not create `tdd.dot`");
        dump_all(file, manager, [(&res, "(x1 ∧ x2) ∨ x4")]).expect("dot export failed");
        Ok(())
    })
}
