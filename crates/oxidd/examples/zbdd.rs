use std::hash::BuildHasherDefault;

use oxidd::util::num::Saturating;
use oxidd::util::SatCountCache;
use oxidd::zbdd::ZBDDFunction;
use oxidd::BooleanFunction;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd_dump::dot::dump_all;
use rustc_hash::FxHasher;

fn main() {
    let manager_ref = oxidd::zbdd::new_manager(1024, 1024, 1);
    let (a, b, c) = manager_ref.with_manager_exclusive(|manager| {
        (
            ZBDDFunction::new_singleton(manager).unwrap(),
            ZBDDFunction::new_singleton(manager).unwrap(),
            ZBDDFunction::new_singleton(manager).unwrap(),
        )
    });

    manager_ref.with_manager_shared(|manager| {
        let n1 = ZBDDFunction::from_edge(
            manager,
            oxidd::zbdd::make_node(
                manager,
                a.as_edge(manager),
                manager.clone_edge(b.as_edge(manager)),
                ZBDDFunction::empty_edge(manager),
            )
            .unwrap(),
        );
        let n2 = ZBDDFunction::from_edge(
            manager,
            oxidd::zbdd::make_node(
                manager,
                a.as_edge(manager),
                ZBDDFunction::empty_edge(manager),
                manager.clone_edge(b.as_edge(manager)),
            )
            .unwrap(),
        );
        assert!(n2 == b);

        let mut count_cache: SatCountCache<Saturating<u64>, BuildHasherDefault<FxHasher>> =
            SatCountCache::default();
        assert!(a.sat_count(3, &mut count_cache).0 == 1);
        assert!(b.sat_count(3, &mut count_cache).0 == 1);
        assert!(c.sat_count(3, &mut count_cache).0 == 1);

        assert!(ZBDDFunction::t(manager).sat_count(3, &mut count_cache).0 == 1 << 3);

        let ab = a.union(&b).unwrap();
        assert!(ab == b.union(&a).unwrap());
        assert!(ab.sat_count(3, &mut count_cache).0 == 2);
        let abc = ab.union(&c).unwrap();
        assert!(abc == c.union(&b).unwrap().union(&a).unwrap());
        assert!(abc.sat_count(3, &mut count_cache).0 == 3);

        assert!(abc.intsec(&ZBDDFunction::t(manager)).unwrap() == abc);

        let complement_a = ZBDDFunction::t(manager).diff(&a).unwrap();
        assert!(complement_a.sat_count(3, &mut count_cache).0 == (1 << 3) - 1);

        let func_a = ZBDDFunction::from_edge(
            manager,
            oxidd::zbdd::var_boolean_function(manager, a.as_edge(manager)).unwrap(),
        );
        let not_a = func_a.not().unwrap();
        let not_not_a = not_a.not().unwrap();

        let file = std::fs::File::create("zbdd.dot").expect("could not create `zbdd.dot`");
        dump_all(
            file,
            manager,
            [(&a, "a"), (&b, "b"), (&c, "c")],
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
    });
}
