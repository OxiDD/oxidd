use std::hash::BuildHasherDefault;

use oxidd::util::num::Saturating;
use oxidd::util::AllocResult;
use oxidd::util::SatCountCache;
use oxidd::zbdd::ZBDDFunction;
use oxidd::BooleanFunction;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd_dump::dot::dump_all;
use rustc_hash::FxHasher;

fn main() -> AllocResult<()> {
    let manager_ref = oxidd::zbdd::new_manager(1024, 1024, 1);
    let (a, b, c) = manager_ref.with_manager_exclusive(|manager| {
        manager.add_named_vars(["a", "b", "c"]).unwrap();
        Ok((
            ZBDDFunction::singleton(manager, 0)?,
            ZBDDFunction::singleton(manager, 1)?,
            ZBDDFunction::singleton(manager, 2)?,
        ))
    })?;

    manager_ref.with_manager_shared(|manager| {
        let n1 = ZBDDFunction::from_edge(
            manager,
            oxidd::zbdd::make_node(
                manager,
                a.as_edge(manager),
                manager.clone_edge(b.as_edge(manager)),
                ZBDDFunction::empty_edge(manager),
            )?,
        );
        let n2 = ZBDDFunction::from_edge(
            manager,
            oxidd::zbdd::make_node(
                manager,
                a.as_edge(manager),
                ZBDDFunction::empty_edge(manager),
                manager.clone_edge(b.as_edge(manager)),
            )?,
        );
        assert!(n2 == b);

        let mut count_cache: SatCountCache<Saturating<u64>, BuildHasherDefault<FxHasher>> =
            SatCountCache::default();
        assert!(a.sat_count(3, &mut count_cache).0 == 1);
        assert!(b.sat_count(3, &mut count_cache).0 == 1);
        assert!(c.sat_count(3, &mut count_cache).0 == 1);

        assert!(ZBDDFunction::t(manager).sat_count(3, &mut count_cache).0 == 1 << 3);

        let ab = a.union(&b)?;
        assert!(ab == b.union(&a)?);
        assert!(ab.sat_count(3, &mut count_cache).0 == 2);
        let abc = ab.union(&c)?;
        assert!(abc == c.union(&b)?.union(&a)?);
        assert!(abc.sat_count(3, &mut count_cache).0 == 3);

        assert!(abc.intsec(&ZBDDFunction::t(manager))? == abc);

        let complement_a = ZBDDFunction::t(manager).diff(&a)?;
        assert!(complement_a.sat_count(3, &mut count_cache).0 == (1 << 3) - 1);

        let func_a = ZBDDFunction::var(manager, 0)?;
        let not_a = func_a.not()?;
        let not_not_a = not_a.not()?;

        let file = std::fs::File::create("zbdd.dot").expect("could not create `zbdd.dot`");
        dump_all(
            file,
            manager,
            [
                (&n1, "n1"),
                (&n2, "n2"),
                (&ab, "ab"),
                (&abc, "abc"),
                (&complement_a, "complement a"),
                (&func_a, "func a"),
                (&not_a, "¬a"),
                (&not_not_a, "¬¬a"),
                (&ZBDDFunction::t(manager), "⊤"),
            ],
        )
        .expect("dot export failed");

        Ok(())
    })?;

    Ok(())
}
