//! Tests for the list decision diagram (LDD) implementation
#![cfg(feature = "ldd")]

use std::collections::BTreeSet;

use oxidd::ldd::{LDDFunction, LDDManagerRef, Value};
use oxidd::util::AllocResult;
use oxidd::{Function, HasWorkers, InnerNode, LevelNo, Manager as _, ManagerRef, WorkerPool};

// spell-checker:ignore mref

/// Length of the vectors in the representative sets
const N: LevelNo = 3;

type Vectors = BTreeSet<Vec<Value>>;
type Manager<'id> = <LDDFunction as Function>::Manager<'id>;

/// Creates a manager with room for vectors of length `N` and relations of
/// length `2 * N`. For `threads > 1`, every recursive call is split, so the
/// parallel code paths are exercised even for small LDDs.
fn setup(threads: u32) -> LDDManagerRef {
    let mref = oxidd::ldd::new_manager(1 << 16, 1 << 16, threads);
    mref.with_manager_exclusive(|mgr| mgr.add_vars(2 * N));
    if threads > 1 {
        mref.with_manager_exclusive(|manager| manager.workers().set_split_depth(Some(u32::MAX)));
    }
    mref
}

/// Builds the LDD for the set of `vectors`.
fn from_vectors<V: AsRef<[Value]>>(
    manager: &Manager<'_>,
    vectors: impl IntoIterator<Item = V>,
) -> AllocResult<LDDFunction> {
    let mut result = LDDFunction::empty_set(manager);
    for v in vectors {
        result = result.union(&LDDFunction::singleton(manager, v.as_ref())?)?;
    }
    Ok(result)
}

/// Enumerates the vectors in `ldd` by walking its nodes.
fn to_vectors(ldd: &LDDFunction) -> Vectors {
    fn rec(ldd: &LDDFunction, prefix: &mut Vec<Value>, out: &mut Vectors) {
        match ldd.node() {
            None => {
                if ldd.is_empty_vector() {
                    out.insert(prefix.clone());
                }
            }
            Some((value, down, right)) => {
                prefix.push(value);
                rec(&down, prefix, out);
                prefix.pop();
                rec(&right, prefix, out);
            }
        }
    }
    let mut out = Vectors::new();
    rec(ldd, &mut Vec::new(), &mut out);
    out
}

/// All vectors of length `len` with values in `0..values`.
fn all_vectors(len: LevelNo, values: Value) -> Vectors {
    let mut out = Vectors::from([Vec::new()]);
    for _ in 0..len {
        out = out
            .into_iter()
            .flat_map(|v| {
                (0..values).map(move |x| {
                    let mut v = v.clone();
                    v.push(x);
                    v
                })
            })
            .collect();
    }
    out
}

/// A representative list of sets of vectors of length `len`: the empty set,
/// single vectors, vectors sharing a prefix or suffix, dense sets and sets
/// with non-contiguous values.
fn representative_sets(len: LevelNo) -> Vec<Vectors> {
    let filter = |values, pred: fn(&[Value]) -> bool| {
        all_vectors(len, values)
            .into_iter()
            .filter(|v| pred(v))
            .collect::<Vectors>()
    };
    let constant = |x: Value| Vectors::from([vec![x; len as usize]]);
    let mut sets = vec![
        Vectors::new(),
        constant(0),
        constant(7),
        all_vectors(len, 2),
        filter(3, |v| v.windows(2).all(|w| w[0] <= w[1])),
        filter(3, |v| v.iter().sum::<Value>() % 3 == 0),
        filter(3, |v| v.first() == Some(&1)),
        filter(3, |v| v.last() == Some(&2)),
    ];
    // Non-contiguous values that differ only in the last position
    let mut sparse = constant(7);
    sparse.insert((0..len).map(|i| if i + 1 == len { 0 } else { 7 }).collect());
    sets.push(sparse);
    sets
}

/// All subsets of `0..len`.
fn subsets(len: LevelNo) -> impl Iterator<Item = Vec<LevelNo>> {
    (0..1u32 << len).map(move |bits| (0..len).filter(|i| bits & (1 << i) != 0).collect())
}

/// All read/write configurations over the positions `0..len`.
fn configurations(len: LevelNo) -> Vec<(Vec<LevelNo>, Vec<LevelNo>)> {
    let mut result = Vec::new();
    for read in subsets(len) {
        for write in subsets(len) {
            result.push((read.clone(), write));
        }
    }
    result
}

/// The definition of [`LDDFunction::project()`].
fn naive_project(set: &Vectors, proj: &[LevelNo]) -> Vectors {
    set.iter()
        .map(|x| {
            (0..x.len() as LevelNo)
                .filter(|i| proj.contains(i))
                .map(|i| x[i as usize])
                .collect()
        })
        .collect()
}

/// Applies the relation vector `t` to `x`, returning the successor if `x`
/// agrees with the read values of `t`.
fn applies(x: &[Value], t: &[Value], read: &[LevelNo], write: &[LevelNo]) -> Option<Vec<Value>> {
    let mut y = x.to_vec();
    let mut k = 0;
    for i in 0..x.len() as LevelNo {
        if read.contains(&i) {
            if x[i as usize] != t[k] {
                return None;
            }
            k += 1;
        }
        if write.contains(&i) {
            y[i as usize] = t[k];
            k += 1;
        }
    }
    Some(y)
}

/// The definition of [`LDDFunction::relational_successors()`].
fn naive_successors(set: &Vectors, rel: &Vectors, read: &[LevelNo], write: &[LevelNo]) -> Vectors {
    set.iter()
        .flat_map(|x| rel.iter().filter_map(|t| applies(x, t, read, write)))
        .collect()
}

/// The definition of [`LDDFunction::relational_predecessors()`].
fn naive_predecessors(
    set: &Vectors,
    rel: &Vectors,
    read: &[LevelNo],
    write: &[LevelNo],
    universe: &Vectors,
) -> Vectors {
    universe
        .iter()
        .filter(|x| {
            rel.iter()
                .any(|t| applies(x, t, read, write).is_some_and(|y| set.contains(&y)))
        })
        .cloned()
        .collect()
}

/// A representative list of distinct relations for `read` and `write`, whose
/// vectors have one value per read and one per write position.
fn representative_relations(read: &[LevelNo], write: &[LevelNo]) -> Vec<Vectors> {
    let len = (read.len() + write.len()) as LevelNo;
    let mut relations = vec![
        Vectors::new(),
        Vectors::from([vec![2; len as usize]]),
        all_vectors(len, 2),
        all_vectors(len, 3)
            .into_iter()
            .filter(|v| v.iter().sum::<Value>() % 4 == 1)
            .collect(),
    ];
    // Without read or write positions, several of these coincide
    relations.sort();
    relations.dedup();
    relations
}

#[test]
fn terminals() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let empty = LDDFunction::empty_set(manager);
        let eps = LDDFunction::empty_vector(manager);
        assert!(empty != eps);

        assert!(empty.is_empty());
        assert!(!empty.is_empty_vector());
        assert_eq!(empty.len(), 0);
        assert!(empty.node().is_none());

        assert!(!eps.is_empty());
        assert!(eps.is_empty_vector());
        assert_eq!(eps.len(), 1);
        assert!(eps.node().is_none());

        // The singleton of the empty vector is the `True` terminal
        assert!(LDDFunction::singleton(manager, &[])? == eps);
        Ok(())
    })
}

#[test]
fn singleton_roundtrip() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        for v in [
            vec![],
            vec![0],
            vec![0, 0, 0],
            vec![3, 1, 2],
            vec![Value::MAX, 0, Value::MAX],
            vec![1; 2 * N as usize],
        ] {
            let f = LDDFunction::singleton(manager, &v)?;
            assert_eq!(to_vectors(&f), Vectors::from([v.clone()]), "for {v:?}");
            assert_eq!(f.len(), 1);
        }
        Ok(())
    })
}

#[test]
#[should_panic(expected = "exceeds")]
fn singleton_exceeding_levels_panics() {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let _ = LDDFunction::singleton(manager, &vec![0; 2 * N as usize + 1]);
    });
}

#[test]
#[should_panic(expected = "exceeds")]
fn make_node_exceeding_levels_panics() {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let eps = LDDFunction::empty_vector(manager);
        let empty = LDDFunction::empty_set(manager);
        let _ = LDDFunction::make_node(manager, 2 * N, 0, &eps, &empty);
    });
}

#[test]
fn make_node_matches_node() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        // `down` must reside at level 1, so take it from a node at level 0
        let (_, down, _) = from_vectors(manager, [[0, 1, 2], [0, 3, 4]])?
            .node()
            .unwrap();
        let right = from_vectors(manager, [[5, 0, 0]])?;
        let f = LDDFunction::make_node(manager, 0, 2, &down, &right)?;

        let (value, d, r) = f.node().unwrap();
        assert_eq!(value, 2);
        assert!(d == down);
        assert!(r == right);
        assert_eq!(
            to_vectors(&f),
            Vectors::from([vec![2, 1, 2], vec![2, 3, 4], vec![5, 0, 0]])
        );

        // `make_node` with the children of an existing node yields that node
        let g = from_vectors(manager, [[2, 1, 2], [2, 3, 4], [5, 0, 0]])?;
        assert!(f == g);
        Ok(())
    })
}

#[test]
fn insertion_order_is_irrelevant() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        for set in representative_sets(N) {
            let forward = from_vectors(manager, &set)?;
            let backward = from_vectors(manager, set.iter().rev())?;
            assert!(forward == backward, "for {set:?}");
            assert_eq!(to_vectors(&forward), set);
        }
        Ok(())
    })
}

#[test]
fn len_counts_vectors() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        for set in representative_sets(N) {
            assert_eq!(from_vectors(manager, &set)?.len(), set.len(), "for {set:?}");
        }
        // Shared sub-diagrams are counted once per path
        let f = from_vectors(manager, all_vectors(2 * N, 2))?;
        assert_eq!(f.len(), 1 << (2 * N));
        Ok(())
    })
}

fn set_ops_match_naive(threads: u32) -> AllocResult<()> {
    let mref = setup(threads);
    mref.with_manager_shared(|manager| {
        let sets = representative_sets(N);
        for a in &sets {
            let fa = from_vectors(manager, a)?;
            for b in &sets {
                let fb = from_vectors(manager, b)?;
                let union = fa.union(&fb)?;
                let intsec = fa.intsec(&fb)?;
                let diff = fa.diff(&fb)?;

                assert_eq!(to_vectors(&union), a | b, "{a:?} ∪ {b:?}");
                assert_eq!(to_vectors(&intsec), a & b, "{a:?} ∩ {b:?}");
                assert_eq!(to_vectors(&diff), a - b, "{a:?} \\ {b:?}");

                // Results are canonical
                assert!(union == from_vectors(manager, a | b)?);
                assert!(intsec == from_vectors(manager, a & b)?);
                assert!(diff == from_vectors(manager, a - b)?);
            }
        }
        Ok(())
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn set_ops_match_naive_t1() -> AllocResult<()> {
    set_ops_match_naive(1)
}

#[test]
#[cfg_attr(miri, ignore)]
fn set_ops_match_naive_t2() -> AllocResult<()> {
    set_ops_match_naive(2)
}

#[test]
fn set_ops_identities() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let empty = LDDFunction::empty_set(manager);
        let eps = LDDFunction::empty_vector(manager);
        let x = from_vectors(manager, [[0, 1, 2], [0, 2, 1]])?;

        // Only combined with themselves and `∅`, since `{ε}` and `x` contain
        // vectors of different lengths
        for f in [&empty, &eps, &x] {
            assert!(f.union(f)? == *f);
            assert!(f.intsec(f)? == *f);
            assert!(f.diff(f)? == empty);
            assert!(f.union(&empty)? == *f);
            assert!(empty.union(f)? == *f);
            assert!(f.intsec(&empty)? == empty);
            assert!(empty.intsec(f)? == empty);
            assert!(f.diff(&empty)? == *f);
            assert!(empty.diff(f)? == empty);
        }
        Ok(())
    })
}

#[test]
fn edge_ops_on_sub_diagrams() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let a_full = from_vectors(manager, [[0, 1, 2], [0, 3, 4]])?;
        let b_full = from_vectors(manager, [[5, 1, 2], [5, 3, 0]])?;

        let down = |edge| manager.get_node(edge).unwrap_inner().child(0);

        // Both down-branches reside at level 1
        let a = down(a_full.as_edge(manager));
        let b = down(b_full.as_edge(manager));

        let union = LDDFunction::union_edge(manager, 1, a, b)?;
        let intsec = LDDFunction::intsec_edge(manager, 1, a, b)?;
        let diff = LDDFunction::diff_edge(manager, 1, a, b)?;
        assert_eq!(
            to_vectors(&union),
            Vectors::from([vec![1, 2], vec![3, 0], vec![3, 4]])
        );
        assert_eq!(to_vectors(&intsec), Vectors::from([vec![1, 2]]));
        assert_eq!(to_vectors(&diff), Vectors::from([vec![3, 4]]));

        // The results reside at level 1, so they can be put below a level 0
        // node
        let empty = LDDFunction::empty_set(manager);
        let f = LDDFunction::make_node(manager, 0, 9, &union, &empty)?;
        assert!(f == from_vectors(manager, [[9, 1, 2], [9, 3, 0], [9, 3, 4]])?);
        Ok(())
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn project_matches_naive() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        for proj in subsets(N) {
            let meta = LDDFunction::projection_meta(manager, &proj)?;
            for set in representative_sets(N) {
                let result = from_vectors(manager, &set)?.project(&meta)?;
                let expected = naive_project(&set, &proj);
                assert_eq!(
                    to_vectors(&result),
                    expected,
                    "project {set:?} onto {proj:?}"
                );
                assert!(result == from_vectors(manager, &expected)?);
            }
        }
        Ok(())
    })
}

#[test]
fn projection_meta_ignores_order_and_duplicates() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let sorted = LDDFunction::projection_meta(manager, &[0, 2])?;
        assert!(LDDFunction::projection_meta(manager, &[2, 0])? == sorted);
        assert!(LDDFunction::projection_meta(manager, &[2, 0, 2, 0])? == sorted);
        Ok(())
    })
}

#[test]
fn project_onto_all_positions_is_identity() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::projection_meta(manager, &[0, 1, 2])?;
        for set in representative_sets(N) {
            let f = from_vectors(manager, &set)?;
            assert!(f.project(&meta)? == f, "for {set:?}");
        }
        Ok(())
    })
}

#[test]
fn project_onto_prefix() -> AllocResult<()> {
    // The meta-LDD ends at the last projected position, so the remaining
    // positions are quantified without being visited.
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::projection_meta(manager, &[0])?;
        let f = from_vectors(manager, [[0, 1, 2], [0, 2, 1], [4, 0, 0]])?;
        assert!(f.project(&meta)? == from_vectors(manager, [[0], [4]])?);
        Ok(())
    })
}

#[test]
fn project_onto_nothing() -> AllocResult<()> {
    // Projecting onto no positions yields `{ε}` for a non-empty set, and `∅`
    // for the empty set.
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::projection_meta(manager, &[])?;
        let eps = LDDFunction::empty_vector(manager);
        let empty = LDDFunction::empty_set(manager);

        assert!(from_vectors(manager, [[0, 1, 2]])?.project(&meta)? == eps);
        assert!(eps.project(&meta)? == eps);
        assert!(empty.project(&meta)? == empty);
        Ok(())
    })
}

#[test]
fn relation_meta_positions() -> AllocResult<()> {
    // The example from the documentation of `relation_meta`. The values of the
    // meta-LDD are private to `oxidd-rules-ldd`, so only its shape is checked.
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[0, 3], &[1, 3])?;
        assert_eq!(meta.read_positions, [0, 2]);
        assert_eq!(meta.write_positions, [1, 3]);
        assert_eq!(to_vectors(&meta.meta).len(), 1);
        assert_eq!(to_vectors(&meta.meta).first().unwrap().len(), 5);

        // Order and duplicates are ignored
        let other = LDDFunction::relation_meta(manager, &[3, 0, 3], &[3, 1])?;
        assert!(other.meta == meta.meta);
        assert_eq!(other.read_positions, meta.read_positions);
        assert_eq!(other.write_positions, meta.write_positions);
        Ok(())
    })
}

fn relational_ops_match_naive(threads: u32) -> AllocResult<()> {
    let mref = setup(threads);
    mref.with_manager_shared(|manager| {
        let sets = representative_sets(N);
        let universes = [all_vectors(N, 3), sets[4].clone(), sets[6].clone()];

        for (read, write) in configurations(N) {
            let meta = LDDFunction::relation_meta(manager, &read, &write)?.meta;
            for rel in representative_relations(&read, &write) {
                let rel_ldd = from_vectors(manager, &rel)?;
                for set in &sets {
                    let set_ldd = from_vectors(manager, set)?;

                    let successors = set_ldd.relational_successors(&rel_ldd, &meta)?;
                    let expected = naive_successors(set, &rel, &read, &write);
                    assert_eq!(
                        to_vectors(&successors),
                        expected,
                        "successors of {set:?} via {rel:?}, read {read:?}, write {write:?}"
                    );
                    assert!(successors == from_vectors(manager, &expected)?);

                    for universe in &universes {
                        let universe_ldd = from_vectors(manager, universe)?;
                        let predecessors =
                            set_ldd.relational_predecessors(&rel_ldd, &meta, &universe_ldd)?;
                        let expected = naive_predecessors(set, &rel, &read, &write, universe);
                        assert_eq!(
                            to_vectors(&predecessors),
                            expected,
                            "predecessors of {set:?} in {universe:?} via {rel:?}, read {read:?}, \
                             write {write:?}"
                        );
                        assert!(predecessors == from_vectors(manager, &expected)?);
                    }
                }
            }
        }
        Ok(())
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn relational_ops_match_naive_t1() -> AllocResult<()> {
    relational_ops_match_naive(1)
}

#[test]
#[cfg_attr(miri, ignore)]
fn relational_ops_match_naive_t2() -> AllocResult<()> {
    relational_ops_match_naive(2)
}

#[test]
fn relational_ops_without_reads_or_writes() -> AllocResult<()> {
    // With no read or write positions, the relation vectors are empty, so the
    // only relations are `{ε}` (the identity) and `∅`.
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[], &[])?.meta;
        let eps = LDDFunction::empty_vector(manager);
        let empty = LDDFunction::empty_set(manager);
        let x = from_vectors(manager, [[0, 1, 2], [3, 4, 5]])?;

        assert!(x.relational_successors(&eps, &meta)? == x);
        assert!(x.relational_successors(&empty, &meta)? == empty);
        assert!(x.relational_predecessors(&eps, &meta, &x)? == x);
        assert!(x.relational_predecessors(&empty, &meta, &x)? == empty);
        Ok(())
    })
}

#[test]
fn relational_ops_empty_operands() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[0], &[1])?.meta;
        let empty = LDDFunction::empty_set(manager);
        let x = from_vectors(manager, [[0, 1, 2]])?;
        let rel = from_vectors(manager, [[0, 5]])?;

        assert!(empty.relational_successors(&rel, &meta)? == empty);
        assert!(x.relational_successors(&empty, &meta)? == empty);
        assert!(empty.relational_predecessors(&rel, &meta, &x)? == empty);
        assert!(x.relational_predecessors(&empty, &meta, &x)? == empty);
        assert!(x.relational_predecessors(&rel, &meta, &empty)? == empty);
        Ok(())
    })
}

#[test]
fn successors_read_only_filters() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[1], &[])?.meta;
        let x = from_vectors(manager, [[0, 1, 2], [0, 2, 2], [3, 1, 0], [4, 4, 4]])?;
        let rel = from_vectors(manager, [[1], [4]])?;
        assert!(
            x.relational_successors(&rel, &meta)?
                == from_vectors(manager, [[0, 1, 2], [3, 1, 0], [4, 4, 4]])?
        );
        Ok(())
    })
}

#[test]
fn successors_write_only_overwrites() -> AllocResult<()> {
    // The set has several values at the write position, which all lead to the
    // same successors
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[], &[1])?.meta;
        let x = from_vectors(manager, [[0, 1, 2], [0, 2, 2], [0, 3, 1]])?;
        let rel = from_vectors(manager, [[5], [6]])?;
        assert!(
            x.relational_successors(&rel, &meta)?
                == from_vectors(manager, [[0, 5, 1], [0, 5, 2], [0, 6, 1], [0, 6, 2]])?
        );
        Ok(())
    })
}

#[test]
fn successors_read_write_same_position() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[0], &[0])?.meta;
        let x = from_vectors(manager, [[0, 1, 2], [1, 1, 2]])?;
        // 0 -> 1 and 1 -> 0, 2 -> 2 is not applicable
        let rel = from_vectors(manager, [[0, 1], [1, 0], [2, 2]])?;
        assert!(x.relational_successors(&rel, &meta)? == x);

        let rel = from_vectors(manager, [[0, 1]])?;
        assert!(x.relational_successors(&rel, &meta)? == from_vectors(manager, [[1, 1, 2]])?);
        Ok(())
    })
}

#[test]
fn successors_copies_untouched_suffix() -> AllocResult<()> {
    // The meta-LDD ends after position 0, so positions 1 and 2 are copied
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[0], &[0])?.meta;
        let x = from_vectors(manager, [[0, 1, 2], [0, 2, 1]])?;
        let rel = from_vectors(manager, [[0, 9]])?;
        assert!(
            x.relational_successors(&rel, &meta)? == from_vectors(manager, [[9, 1, 2], [9, 2, 1]])?
        );
        Ok(())
    })
}

#[test]
fn successors_cache_distinguishes_levels() -> AllocResult<()> {
    // The example from the documentation of `apply_relational_successors`: in
    // both cases the last write half has the operands `(True, rel, meta)`
    // with equal `rel` and `meta`, but the results reside on different
    // levels.
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let rel = from_vectors(manager, [[0, 5, 0, 9]])?;

        let meta2 = LDDFunction::relation_meta(manager, &[0, 1], &[0, 1])?.meta;
        let x2 = from_vectors(manager, [[0, 0]])?;
        let meta3 = LDDFunction::relation_meta(manager, &[0, 2], &[1, 2])?.meta;
        let x3 = from_vectors(manager, [[0, 0, 0]])?;

        assert!(x2.relational_successors(&rel, &meta2)? == from_vectors(manager, [[5, 9]])?);
        assert!(x3.relational_successors(&rel, &meta3)? == from_vectors(manager, [[0, 5, 9]])?);
        // And in the reverse order, with the cache already populated
        assert!(x2.relational_successors(&rel, &meta2)? == from_vectors(manager, [[5, 9]])?);
        Ok(())
    })
}

#[test]
fn successors_distributes_over_union() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let sets = representative_sets(N);
        for (read, write) in [(vec![0], vec![2]), (vec![1], vec![1]), (vec![], vec![0, 1])] {
            let meta = LDDFunction::relation_meta(manager, &read, &write)?.meta;
            let rel = from_vectors(
                manager,
                representative_relations(&read, &write).pop().unwrap(),
            )?;
            for a in &sets {
                let fa = from_vectors(manager, a)?;
                for b in &sets {
                    let fb = from_vectors(manager, b)?;
                    let joint = fa.union(&fb)?.relational_successors(&rel, &meta)?;
                    let separate = fa
                        .relational_successors(&rel, &meta)?
                        .union(&fb.relational_successors(&rel, &meta)?)?;
                    assert!(joint == separate, "for {a:?}, {b:?}");
                }
            }
        }
        Ok(())
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn predecessors_of_successors() -> AllocResult<()> {
    // `pre(post(X), X)` are the vectors of `X` that have a successor, so their
    // successors are exactly those of `X`. This does not rely on the naive
    // definitions.
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        for (read, write) in configurations(N) {
            let meta = LDDFunction::relation_meta(manager, &read, &write)?.meta;
            for rel in representative_relations(&read, &write) {
                let rel = from_vectors(manager, rel)?;
                for set in representative_sets(N) {
                    let x = from_vectors(manager, &set)?;
                    let post = x.relational_successors(&rel, &meta)?;
                    let pre = post.relational_predecessors(&rel, &meta, &x)?;
                    assert!(pre.diff(&x)?.is_empty(), "for {set:?}");
                    assert!(
                        pre.relational_successors(&rel, &meta)? == post,
                        "for {set:?}"
                    );
                }
            }
        }
        Ok(())
    })
}

#[test]
fn predecessors_within_universe() -> AllocResult<()> {
    let mref = setup(1);
    mref.with_manager_shared(|manager| {
        let meta = LDDFunction::relation_meta(manager, &[0], &[2])?.meta;
        let set = from_vectors(manager, [[0, 1, 5]])?;
        let rel = from_vectors(manager, [[0, 5]])?;
        // [0, 1, x] for any x reaches [0, 1, 5]
        let universe = from_vectors(manager, [[0, 1, 0], [0, 1, 7], [0, 2, 0], [1, 1, 0]])?;
        assert!(
            set.relational_predecessors(&rel, &meta, &universe)?
                == from_vectors(manager, [[0, 1, 0], [0, 1, 7]])?
        );
        Ok(())
    })
}
