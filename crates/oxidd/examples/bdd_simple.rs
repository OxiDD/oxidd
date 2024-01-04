use oxidd::bdd::BDDFunction;
use oxidd::AllocResult;
use oxidd::BooleanFunction;
use oxidd::ManagerRef;

fn main() -> AllocResult<()> {
    let mref = oxidd::bdd::new_manager(1024, 1024, 1);
    let (x1, x2, x3) = mref.with_manager_exclusive(|manager| {
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
