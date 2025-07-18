use oxidd::bdd::BDDFunction;
use oxidd::util::AllocResult;
use oxidd::{BooleanFunction, Manager, ManagerRef};

fn main() -> AllocResult<()> {
    let manager_ref = oxidd::bdd::new_manager(1024, 1024, 1);
    let (x1, x2, x3) = manager_ref.with_manager_exclusive(|manager| {
        manager.add_named_vars(["x", "y", "z"]).unwrap();
        Ok((
            BDDFunction::var(manager, 0)?,
            BDDFunction::var(manager, 1)?,
            BDDFunction::var(manager, 2)?,
        ))
    })?;

    let res = x1.and(&x2)?.or(&x3)?;
    println!("{}", res.satisfiable());
    Ok(())
}
