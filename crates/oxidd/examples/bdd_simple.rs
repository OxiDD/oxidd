use oxidd::bdd::BDDFunction;
use oxidd::util::AllocResult;
use oxidd::BooleanFunction;
use oxidd::ManagerRef;

fn main() -> AllocResult<()> {
    let manager_ref = oxidd::bdd::new_manager(1024, 1024, 1);
    let (x1, x2, x3) = manager_ref.with_manager_exclusive(|manager| {
        (
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::new_var(manager).unwrap(),
            BDDFunction::new_var(manager).unwrap(),
        )
    });

    let res = x1.and(&x2)?.or(&x3)?;
    println!("{}", res.satisfiable());
    Ok(())
}
