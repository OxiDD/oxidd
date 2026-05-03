use std::sync::atomic::Ordering::Relaxed;

use is_sorted::IsSorted;

use oxidd_core::{
    AtomicLevelNo, HasLevel, HasWorkers, LevelNo, LevelView, Manager, VarNo, WorkerPool,
};

mod segtree;
use segtree::MinSegTree;

/// Reorder the variables according to `order`
///
/// Sequential version of [`set_var_order()`].
///
/// If a variable `x` occurs before variable `y` in `order`, then `x` will be
/// above `y` in the decision diagram when this function returns. Variables not
/// mentioned in `order` will be placed in a position such that the least number
/// of adjacent level swaps need to be performed. Panics if a variable occurs
/// twice in `order`.
///
/// There is no need for the caller to call
/// [`manager.reorder()`][Manager::reorder], this is done internally.
pub fn set_var_order_seq<M: Manager>(manager: &mut M, order: &[VarNo])
where
    M::InnerNode: HasLevel,
{
    if order.len() <= 1 {
        return; // nothing to do
    }
    set_var_order_common(manager, order, bubble_sort, update_levels_seq);
}

/// Reorder the variables according to `order`
///
/// Like [`set_var_order_seq()`], but with concurrent swap operations.
///
/// If a variable `x` occurs before variable `y` in `order`, then `x` will be
/// above `y` in the decision diagram when this function returns. Variables not
/// mentioned in `order` will be placed in a position such that the least number
/// of adjacent level swaps need to be performed. Panics if a variable occurs
/// twice in `order`.
///
/// There is no need for the caller to call
/// [`manager.reorder()`][Manager::reorder], this is done internally.
pub fn set_var_order<M>(manager: &mut M, order: &[VarNo])
where
    M: Manager + HasWorkers,
    M::InnerNode: HasLevel,
{
    if order.len() <= 1 {
        return; // nothing to do
    }
    let sequential =
        manager.workers().current_num_threads() == 1 || manager.approx_num_inner_nodes() < 65536;
    let (sort, update): (SortFn<M>, UpdateLevelFn<M>) = if sequential {
        (bubble_sort, update_levels_seq)
    } else {
        (concurrent_bubble_sort, update_levels)
    };
    set_var_order_common(manager, order, sort, update);
}

type SwapFn<'a, M> = &'a (dyn for<'b> Fn(&'b M, u32) + Sync);
type SortFn<M> = fn(&M, &mut [u32], SwapFn<'_, M>);
type UpdateLevelFn<M> = fn(&M, Vec<AtomicLevelNo>);

#[inline(never)]
fn set_var_order_common<M: Manager>(
    manager: &mut M,
    order: &[VarNo],
    sort: SortFn<M>,
    update_levels: UpdateLevelFn<M>,
) where
    M::InnerNode: HasLevel,
{
    let num_levels = manager.num_levels();

    let mut target_order = sort_order(num_levels, order.iter().map(|&v| manager.var_to_level(v)));

    // "ne" stands for non-empty levels here. In the first step, we focus on
    // these levels and bring them into the correct relative order.

    // Subsequence of `target_order`, including the target positions of
    // non-empty levels only
    let mut ne_target_order = Vec::with_capacity(num_levels as usize);
    // Mapping from indices in `ne_target_order` to indices in `target_order`
    let mut from_ne = Vec::with_capacity(num_levels as usize);
    let mut sorted = true; // whether all levels are already sorted
    let mut ne_sorted = true; // whether non-empty levels are sorted
    let mut last_ne_target = 0; // target position of the last non-empty level
    for (level, &target) in manager.levels().zip(&target_order) {
        sorted = sorted && level.level_no() == target;
        if !level.is_empty() {
            from_ne.push(level.level_no());
            ne_target_order.push(target);
            ne_sorted = ne_sorted && target >= last_ne_target;
            last_ne_target = target;
        }
    }

    if sorted {
        return;
    }

    manager.reorder(|manager| {
        // Use a scope guard to always execute the final step, even when
        // unwinding: Update the level information in nodes.
        let to_pre = &mut scopeguard::guard(
            Vec::from_iter((0..num_levels).map(AtomicLevelNo::new)),
            |to_pre| update_levels(manager, to_pre),
        )[..];

        // First step: Bring the non-empty levels into the desired order,
        // relatively to each other.
        if !ne_sorted {
            let swap: SwapFn<'_, M> = &|manager, i| {
                let u = from_ne[i as usize];
                let l = from_ne[(i + 1) as usize];
                let up = to_pre[u as usize].load(Relaxed);
                let lp = to_pre[l as usize].load(Relaxed);
                unsafe { crate::level_swap(manager, u, l, up, lp) };
                to_pre[u as usize].store(lp, Relaxed);
                to_pre[l as usize].store(up, Relaxed);
            };
            sort(manager, &mut ne_target_order, swap);

            if from_ne.len() == target_order.len() {
                return;
            }

            for (i, v) in from_ne.into_iter().zip(ne_target_order) {
                target_order[i as usize] = v;
            }
        }

        // Second step: Establish the target order, including empty levels. We
        // do not modify nodes here, the running time is linear in the number of
        // levels.
        let mut i = 0;
        while let Some(&j) = target_order.get(i as usize) {
            if j == i {
                i += 1;
            } else {
                unsafe {
                    manager
                        .level_unchecked(i)
                        .swap(&mut manager.level_unchecked(j))
                };
                to_pre.swap(i as usize, j as usize);
                target_order.swap(i as usize, j as usize);
            }
        }
    });

    debug_assert!(IsSorted::is_sorted(
        &mut order.iter().map(|&v| manager.var_to_level(v))
    ));
}

/// Translate the relative positioning of levels given by `input_order` into a
/// mapping from the current level number to the target level number.
///
/// If a level number `i` occurs before a level number `j` in `input_order`, the
/// variable associated with `i` should be above the variable associated with
/// `j` after the reordering operation. If a level `k` does not occur in
/// `input_order`, it may be placed anywhere. This algorithm places these levels
/// such that the number of adjacent level swaps between the current and the
/// target order is minimal. Whenever there are multiple optimal positions for a
/// level, it chooses the top-most position.
///
/// Assuming that there are `m` levels mentioned by `input_order`, the current
/// implementation has a time complexity of O(`num_levels` + m log m) and a
/// space complexity of O(`num_levels`).
#[track_caller]
fn sort_order(num_levels: u32, input_order: impl IntoIterator<Item = LevelNo>) -> Vec<u32> {
    let mut target_order = vec![u32::MAX; num_levels as usize];
    let mut input_order_len = 0;
    for level in input_order {
        assert_eq!(
            target_order[level as usize],
            u32::MAX,
            "`order` contains level {level} twice"
        );
        target_order[level as usize] = input_order_len;
        input_order_len += 1;
    }

    if input_order_len != num_levels {
        // The given order is not total. Make it total with the minimum number
        // of swap operations possible.
        assert!(
            input_order_len < i32::MAX as u32 - 1,
            "too many variables in input order for this algorithm"
        );
        let mut segtree = MinSegTree::new(0..(input_order_len as i32 + 1));
        for i in &mut target_order {
            if *i == u32::MAX {
                *i = segtree.min_index() as u32;
            } else {
                segtree.add_split((*i + 1) as usize, 1, -1);
            }
        }
        drop(segtree);
        debug_assert!(!target_order.contains(&u32::MAX));

        // `target_order` now maps the levels from `input_order` to "position
        // indicators" in range `0..input_order_len`. The levels not mentioned
        // by `input_order` are mapped to position indicators in range
        // `0..=input_order_len`. If such a level not mentioned by `input_order`
        // has a position indicator `i`, it should be placed between the levels
        // with position indicator `i` and `i-1` of `input_order` (or at the
        // very top or bottom, respectively).
        //
        // We now translate the position indicators to unique position numbers
        // in range `0..num_levels`, preserving the less-than-relation on
        // position indicators. Among the same position indicators, the one at
        // the lowest index in the current `target_order` should go first, since
        // this leads to the lowest number of adjacent level swaps.

        // First, count how often each position indicator appears.
        let mut counts = vec![0u32; (input_order_len + 1) as usize];
        for &i in &target_order {
            counts[i as usize] += 1;
        }
        // Now accumulate these counts such that `counts[i]` describes how many
        // position indicators less than `i` there are.
        let mut sum = 0;
        for c in &mut counts {
            let val_count = *c;
            *c = sum;
            sum += val_count;
        }
        // Finally, write the new `target_order`
        for i in &mut target_order {
            let c = &mut counts[*i as usize];
            *i = *c;
            *c += 1;
        }
    }

    debug_assert!(is_permutation(&target_order));

    target_order
}

/// Check if `slice` is a permutation of the numbers in range `0..slice.len()`
fn is_permutation(slice: &[u32]) -> bool {
    // This is not optimized, but we currently use the function in debug
    // assertions only.
    let mut found = vec![false; slice.len()];
    for &v in slice {
        match found.get_mut(v as usize) {
            Some(f) if !*f => *f = true,
            _ => return false,
        }
    }
    true
}

/// Sorts the given sequence by swapping adjacent levels only. For every swap
/// operation, `swap` is called with the smaller index.
fn bubble_sort<M>(manager: &M, seq: &mut [u32], swap: SwapFn<'_, M>) {
    let mut n = seq.len();
    while n > 1 {
        let mut new_n = 0;
        for i in 1..n {
            if seq[i - 1] > seq[i] {
                seq.swap(i - 1, i);
                swap(manager, (i - 1) as LevelNo);
                new_n = i;
            }
        }
        n = new_n;
    }
}

/// Sorts the given sequence by swapping adjacent levels only. For every swap
/// operation, `swap` is called with the smaller index.
///
/// There may be multiple swap operations at the same time, but the
/// implementation ensures that if there is a swap operation for indices `i` and
/// `i + 1`, there is no other swap operation with `i` and `i + 1` at the same
/// time.
fn concurrent_bubble_sort<M>(manager: &M, seq: &mut [u32], swap: SwapFn<'_, M>)
where
    M: HasWorkers,
{
    let (task_sender, task_receiver) = flume::unbounded::<u32>();
    let (done_sender, done_receiver) = flume::unbounded::<u32>();

    std::thread::scope(|scope| {
        scope.spawn(move || {
            let swap = |seq: &mut [u32], blocked: &mut [bool], upper: usize| {
                let lower = upper + 1;
                seq.swap(upper, lower);
                blocked[upper] = true;
                blocked[lower] = true;
                task_sender.send(upper as u32).unwrap();
            };

            let mut blocked = vec![false; seq.len()];
            let mut i = 0;
            let mut tasks = 0u32;
            while i + 1 < seq.len() {
                if seq[i] > seq[i + 1] {
                    swap(seq, &mut blocked, i);
                    tasks += 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            while tasks != 0 {
                let i = done_receiver.recv().unwrap() as usize;
                tasks -= 1;
                blocked[i] = false;
                blocked[i + 1] = false;
                if i > 0 && seq[i - 1] > seq[i] && !blocked[i - 1] {
                    swap(seq, &mut blocked, i - 1);
                    tasks += 1;
                }
                if i + 2 < seq.len() && seq[i + 1] > seq[i + 2] && !blocked[i + 2] {
                    swap(seq, &mut blocked, i + 1);
                    tasks += 1;
                }
            }
            debug_assert!(IsSorted::is_sorted(&mut seq.iter()));
        });

        manager.workers().broadcast(|_| {
            while let Ok(upper) = task_receiver.recv() {
                swap(manager, upper);
                done_sender.send(upper).unwrap();
            }
        });
    });
}

fn update_levels_seq<M: Manager>(manager: &M, to_pre: Vec<AtomicLevelNo>)
where
    M::InnerNode: HasLevel,
{
    for (level, p) in manager.levels().zip(to_pre) {
        if level.level_no() != p.into_inner() {
            crate::update_level_no(manager, &level);
        }
    }
}

fn update_levels<M: Manager + HasWorkers>(manager: &M, to_pre: Vec<AtomicLevelNo>)
where
    M::InnerNode: HasLevel,
{
    let (sender, receiver) = flume::unbounded();
    for (level, p) in manager.levels().zip(to_pre) {
        if level.level_no() != p.into_inner() {
            if level.len() >= 1024 {
                sender.send(level.level_no()).unwrap();
            } else {
                crate::update_level_no(manager, &level);
            }
        }
    }
    drop(sender);
    manager.workers().broadcast(|_| {
        while let Ok(l) = receiver.recv() {
            let level = unsafe { manager.level_unchecked(l) };
            crate::update_level_no(manager, &level);
        }
    });
}

#[cfg(test)]
mod test {
    use oxidd_test_utils::edge::DummyManager;
    use std::sync::atomic::Ordering::Relaxed;

    use super::*;

    #[test]
    fn test_sort_order() {
        assert_eq!(sort_order(4, [0, 1, 2, 3]), vec![0, 1, 2, 3]);
        assert_eq!(sort_order(4, [1, 0, 2, 3]), vec![1, 0, 2, 3]);
        assert_eq!(sort_order(4, [0, 2, 3, 1]), vec![0, 3, 1, 2]);

        assert_eq!(sort_order(3, [2, 0]), vec![2, 0, 1]);
        assert_eq!(
            sort_order(10, [6, 3, 0, 4, 1, 9]),
            vec![3, 5, 0, 2, 4, 6, 1, 7, 8, 9]
        );
        assert_eq!(
            sort_order(8, [7, 3, 0, 5, 6, 1]),
            vec![3, 7, 0, 2, 4, 5, 6, 1]
        );
    }

    fn bubble_sort_test(f: SortFn<DummyManager>) {
        fn case<const N: usize>(f: SortFn<DummyManager>, mut seq: [u32; N]) {
            let mut sorted = seq;
            sorted.sort();

            let seq2 = seq.map(::std::sync::atomic::AtomicU32::new);
            f(&DummyManager, &mut seq, &|_, i| {
                let a = seq2[i as usize].load(Relaxed);
                let b = seq2[(i + 1) as usize].swap(a, Relaxed);
                seq2[i as usize].store(b, Relaxed);
            });

            assert_eq!(seq, sorted);
            assert!(seq2
                .into_iter()
                .zip(sorted)
                .all(|(a, b)| a.into_inner() == b));
        }

        case(f, []);
        case(f, [0]);
        case(f, [1]);
        case(f, [0, 1, 2]);
        case(f, [1, 4, 5]);
        case(f, [3, 2, 0]);

        f(&DummyManager, &mut [0, 0, 1, 1], &|_, _| {
            panic!("sort is unstable")
        });
    }

    #[test]
    fn test_bubble_sort() {
        bubble_sort_test(bubble_sort);
    }
    #[test]
    fn test_concurrent_bubble_sort() {
        bubble_sort_test(concurrent_bubble_sort);
    }
}
