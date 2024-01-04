use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use oxidd::util::num::Saturating;
use oxidd::zbdd::ZBDDSet;
use oxidd::BooleanFunction;
use oxidd::BooleanVecSet;
use oxidd::Function;
use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd_dump::dot::dump_all;
use rustc_hash::FxHasher;

fn main() {
    let mref = oxidd::zbdd::new_manager(1024, 1024, 1);
    let (a, b, c) = mref.with_manager_exclusive(|manager| {
        (
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::new_singleton(manager).unwrap(),
            ZBDDSet::new_singleton(manager).unwrap(),
        )
    });

    mref.with_manager_shared(|manager| {
        let n1 = ZBDDSet::from_edge(
            manager,
            oxidd::zbdd::make_node(
                manager,
                a.as_edge(manager),
                manager.clone_edge(b.as_edge(manager)),
                ZBDDSet::empty_edge(manager),
            )
            .unwrap(),
        );
        let n2 = ZBDDSet::from_edge(
            manager,
            oxidd::zbdd::make_node(
                manager,
                a.as_edge(manager),
                ZBDDSet::empty_edge(manager),
                manager.clone_edge(b.as_edge(manager)),
            )
            .unwrap(),
        );
        assert!(n2 == b);

        let mut count_cache: HashMap<_, Saturating<u64>, BuildHasherDefault<FxHasher>> =
            HashMap::default();
        assert!(a.sat_count(3, &mut count_cache).0 == 1);
        assert!(b.sat_count(3, &mut count_cache).0 == 1);
        assert!(c.sat_count(3, &mut count_cache).0 == 1);

        assert!(ZBDDSet::t(manager).sat_count(3, &mut count_cache).0 == 1 << 3);

        let ab = a.union(&b).unwrap();
        assert!(ab == b.union(&a).unwrap());
        assert!(ab.sat_count(3, &mut count_cache).0 == 2);
        let abc = ab.union(&c).unwrap();
        assert!(abc == c.union(&b).unwrap().union(&a).unwrap());
        assert!(abc.sat_count(3, &mut count_cache).0 == 3);

        assert!(abc.intsec(&ZBDDSet::t(manager)).unwrap() == abc);

        let compl_a = ZBDDSet::t(manager).diff(&a).unwrap();
        assert!(compl_a.sat_count(3, &mut count_cache).0 == (1 << 3) - 1);

        let func_a = ZBDDSet::from_edge(
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
                (&compl_a, "compl a"),
                (&func_a, "func a"),
                (&not_a, "¬a"),
                (&not_not_a, "¬¬a"),
                (&ZBDDSet::t(manager), "⊤"),
            ],
        )
        .expect("dot export failed");
    });
}
