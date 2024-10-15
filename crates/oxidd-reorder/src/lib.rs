//! Functionality to reorder levels in a decision diagram

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

use is_sorted::IsSorted;
use oxidd_core::util::OutOfMemory;
use oxidd_core::HasWorkers;
use oxidd_core::WorkerPool;
use smallvec::SmallVec;

use oxidd_core::function::Function;
use oxidd_core::util::AbortOnDrop;
use oxidd_core::util::Borrowed;
use oxidd_core::util::DropWith;
use oxidd_core::DiagramRules;
use oxidd_core::Edge;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::LevelView;
use oxidd_core::Manager;
use oxidd_core::ReducedOrNew;

/// Swap the level given by `upper_no` with the level directly below.
///
/// # Safety
///
/// Must be called from inside the closure of
/// [`manager.reorder()`][Manager::reorder]. `manager` must be derived from a
/// `&mut M` reference. This function may modify nodes at the level of
/// `upper_no` and `upper_no + 1` (i.e. the level below). There must not be any
/// concurrent modification of any nodes at these levels.
pub unsafe fn level_down<M: Manager>(manager: &M, upper_no: LevelNo)
where
    M::InnerNode: HasLevel,
{
    let abort_on_panic = AbortOnDrop("Some operation in `oxidd_reorder::level_down` panicked.");

    let lower_no = upper_no + 1;
    debug_assert_ne!(lower_no, LevelNo::MAX);
    // Note that the `Manager::level()` may or may not acquire a lock.
    let mut upper = manager.level(upper_no);
    let mut lower = manager.level(lower_no);
    unsafe { upper.swap(&mut lower) };

    // `lower` now refers to the new lower level, `upper` to the new upper
    // level. Note that the level numbers for each node still need to be
    // adjusted, which we do below. Furthermore, nodes on the new lower level
    // may refer to the new upper level, which is not allowed. We also fix this
    // below.

    for e in upper.iter() {
        let node = manager.get_node(e).unwrap_inner();
        debug_assert_eq!(node.level(), lower_no);
        // SAFETY: we have exclusive access to all nodes at the new upper level
        unsafe { node.set_level(upper_no) };
    }

    let old_upper = LevelView::take(&mut lower);
    lower.reserve(old_upper.len());
    for e in old_upper.iter() {
        let node = manager.get_node(e).unwrap_inner();
        debug_assert_eq!(node.level(), upper_no);

        // TODO: Maybe use `N::ARITY` once `generic_const_exprs` becomes
        // stable, and `arrayvec` instead of `smallvec`?
        let children: SmallVec<[Borrowed<M::Edge>; 2]> = node.children().collect();
        debug_assert_eq!(children.len(), M::InnerNode::ARITY);

        if children
            .iter()
            .all(|c| manager.get_node(c).unwrap_inner().level() > lower_no)
        {
            // All children are below the lower level, we just move the node to
            // the lower level.
            drop(children);
            // SAFETY: we have exclusive access to all nodes at the old upper level
            unsafe { node.set_level(lower_no) };
            lower.insert(manager.clone_edge(e));
            continue;
        }

        // The node refers to the upper level. To fix this, we compute the
        // cofactors of the cofactors of this node, create new nodes at the
        // lower level, and keep the original node at the upper level (with the
        // children replaced by the newly created ones).

        let grandchildren: SmallVec<[_; 2]> = children
            .iter()
            .map(|c| {
                let node = manager.get_node(c).unwrap_inner();
                if node.level() == lower_no {
                    // We have exclusive access to the node
                    let children: SmallVec<[_; 2]> = M::Rules::cofactors(c.tag(), node).collect();
                    debug_assert_eq!(children.len(), M::InnerNode::ARITY);
                    children
                } else {
                    // The child is below the lower level, so we always have
                    // this child
                    (0..M::InnerNode::ARITY).map(|_| c.borrowed()).collect()
                }
            })
            .collect();

        let new_children: SmallVec<[_; 2]> = (0..M::InnerNode::ARITY)
            .map(|i| {
                let res = <M::Rules as DiagramRules<_, _, _>>::reduce(
                    manager,
                    lower_no,
                    grandchildren.iter().map(|v| manager.clone_edge(&v[i])),
                );
                match res {
                    ReducedOrNew::Reduced(e) => e,
                    ReducedOrNew::New(node, tag) => if let Some(e) = old_upper.get(&node) {
                        node.drop_with_manager(manager);
                        manager.clone_edge(e)
                    } else {
                        match lower.get_or_insert(node) {
                            Ok(e) => e,
                            Err(OutOfMemory) => {
                                eprintln!("Out of memory");
                                std::process::abort();
                            }
                        }
                    }
                    .with_tag_owned(tag),
                }
            })
            .collect();

        drop(grandchildren);
        for child in children {
            // Revisit the "old" children of `e`. If these are the only
            // children, we may remove them, if they are on the current upper
            // level. (A child might also be at some lower level, in which case
            // the node could also be removed. However we must not access such
            // a node.)
            let child_node = manager.get_node(&*child).unwrap_inner();
            if child_node.level() == upper_no && child_node.ref_count() == 1 {
                // The main reference corresponds to the old node, which is
                // deleted below. The weak reference is the one in the unique
                // table. Hence, we can remove the node.
                unsafe { upper.remove(child_node) };
            }
        }

        upper.insert(manager.clone_edge(e));
        for (i, child) in new_children.into_iter().enumerate() {
            // SAFETY: we have exclusive access to all nodes at the old upper
            // level and no child is borrowed.
            manager.drop_edge(unsafe { node.set_child(i, child) });
        }
    }

    abort_on_panic.defuse();
}

/// Reorder the variables such that the edges in `order` are sorted by their
/// levels.
///
/// Sequential version of [`set_var_order()`].
///
/// The caller must not call [`manager.reorder()`][Manager::reorder].
pub fn set_var_order_seq<'id, F: Function>(manager: &mut F::Manager<'id>, order: &[F])
where
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    if order.len() <= 1 {
        return; // nothing to do
    }

    let mut target_order = sort_order(
        manager.num_levels(),
        order.iter().map(|f| {
            manager
                .get_node(f.as_edge(manager))
                .expect_inner("order must not contain (const) terminals")
                .level()
        }),
    );

    manager.reorder(|manager| {
        // Finally a use case for bubble sort :)
        bubble_sort(&mut target_order, |upper_no| unsafe {
            level_down(manager, upper_no)
        });
    });

    debug_assert!(IsSorted::is_sorted(&mut order.iter().map(|f| {
        let edge = f.as_edge(manager);
        manager.get_node(edge).unwrap_inner().level()
    })));
}

/// Reorder the variables such that the edges in `order` are sorted by their
/// levels.
///
/// Like [`set_var_order_seq()`] but with concurrent swap operations.
///
/// The caller must not call [`manager.reorder()`][Manager::reorder].
pub fn set_var_order<'id, F: Function>(manager: &mut F::Manager<'id>, order: &[F])
where
    F::Manager<'id>: Manager + HasWorkers,
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    if order.len() <= 1 {
        return; // nothing to do
    }

    let num_levels = manager.num_levels();
    let mut target_order = sort_order(
        num_levels,
        order.iter().map(|f| {
            manager
                .get_node(f.as_edge(manager))
                .expect_inner("order must not contain (const) terminals")
                .level()
        }),
    );

    manager.reorder(|manager| {
        if num_levels <= 8 {
            bubble_sort(&mut target_order, |upper_no| unsafe {
                level_down(manager, upper_no)
            });
        } else {
            concurrent_bubble_sort(manager, target_order, |upper_no| unsafe {
                level_down(manager, upper_no)
            });
        }
    });

    debug_assert!(IsSorted::is_sorted(&mut order.iter().map(|f| {
        let edge = f.as_edge(manager);
        manager.get_node(edge).unwrap_inner().level()
    })));
}

/// Transform the `input_order` into a target order suitable for sorting, that
/// is: if the value at index `i` is greater than the value at index `i + 1`,
/// then level `i` and `i + 1` need to be swapped.
///
/// `input_order` conceptually describes how the variables should be ordered in
/// the end, i.e. the variables corresponding to the given levels should be
/// reordered such that the level numbers would be increasing in that order.
fn sort_order(num_levels: u32, input_order: impl IntoIterator<Item = LevelNo>) -> Vec<LevelNo> {
    let mut target_order = vec![LevelNo::MAX; num_levels as usize];
    let mut input_order_len = 0;
    for level in input_order {
        assert_eq!(
            target_order[level as usize],
            u32::MAX,
            "`order` contains level {level} twice but it must be a permutation of the present levels"
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
    }

    debug_assert!(!target_order.contains(&LevelNo::MAX));

    target_order
}

/// A segment tree that allows additive range updates (in O(log n)) and queries
/// for the index of some minimal element (in O(log n), could be optimized to
/// O(1) by adding a `min_index` field to `MinSegTreeEntry`)
#[derive(PartialEq, Eq)]
struct MinSegTree(Vec<MinSegTreeEntry>);

#[derive(Copy, Clone, PartialEq, Eq)]
struct MinSegTreeEntry {
    delta: i32,
    /// The minimum value of the children + self.delta
    min: i32,
}

impl MinSegTree {
    #[inline]
    fn parent(i: usize) -> usize {
        i / 2
    }
    #[inline]
    fn left(i: usize) -> usize {
        2 * i
    }
    #[inline]
    fn right(i: usize) -> usize {
        2 * i + 1
    }
    #[inline]
    fn is_left_child(i: usize) -> bool {
        i % 2 == 0
    }

    fn new(data: impl ExactSizeIterator<Item = i32>) -> Self {
        let size = data.len().next_power_of_two();
        let mut tree = Vec::with_capacity(2 * size);
        tree.resize(size, MinSegTreeEntry { delta: 0, min: 0 });
        tree.extend(data.map(|v| MinSegTreeEntry { delta: v, min: v }));
        tree.resize(
            2 * size,
            MinSegTreeEntry {
                delta: i32::MAX,
                min: i32::MAX,
            },
        );

        for i in (1..size).rev() {
            tree[i].min =
                tree[i].delta + std::cmp::min(tree[Self::left(i)].min, tree[Self::right(i)].min);
        }

        MinSegTree(tree)
    }

    /// Add `left` to all elements in range `..i` and `right` to all elements in
    /// `i..`
    fn add_split(&mut self, i: usize, left: i32, right: i32) {
        let mut size = self.0.len() / 2;
        assert!(i <= size);

        #[inline]
        fn update(entry: &mut MinSegTreeEntry, delta: i32) {
            if entry.min != i32::MAX {
                entry.delta += delta;
                entry.min += delta;
            }
        }

        // Special case: borders. Either there is nothing to the left to modify,
        // or nothing to the right.
        if i == 0 {
            update(&mut self.0[1], right);
            return;
        }
        if i == size {
            update(&mut self.0[1], left);
            return;
        }

        let mut levels_from_bot = size.trailing_zeros();
        let mut node = 1;

        loop {
            levels_from_bot -= 1;
            size /= 2;
            if i & (size - 1) /* i % size */ == 0 {
                break;
            }
            if Self::is_left_child(i >> levels_from_bot) {
                node = Self::left(node);
                update(&mut self.0[node + 1], right);
            } else {
                node = Self::right(node);
                update(&mut self.0[node - 1], left);
            }
        }
        update(&mut self.0[Self::left(node)], left);
        update(&mut self.0[Self::right(node)], right);

        loop {
            let l = Self::left(node);
            let r = Self::right(node);
            self.0[node].min = self.0[node].delta + std::cmp::min(self.0[l].min, self.0[r].min);
            if node == 1 {
                break;
            }
            node = Self::parent(node);
        }
    }

    /// Get the index of the minimal element with the lowest index
    fn min_index(&self) -> usize {
        let size = self.0.len() / 2;
        let mut i = 1;
        while i < size {
            let l = Self::left(i);
            let r = Self::right(i);
            i = if self.0[l].min <= self.0[r].min { l } else { r };
        }
        i - size
    }

    /// Project the segment tree down to an array (apply all deltas)
    ///
    /// For debugging purposes
    #[allow(unused)]
    fn proj(&self) -> Vec<i32> {
        fn rec(inp: &[MinSegTreeEntry], out: &mut [i32], i: usize, sum: i32) {
            let sum = sum + inp[i].delta;
            if i >= out.len() {
                out[i - out.len()] = sum;
            } else {
                rec(inp, out, MinSegTree::left(i), sum);
                rec(inp, out, MinSegTree::right(i), sum);
            }
        }

        let size = self.0.len() / 2;
        let mut res: Vec<i32> = vec![0; size];
        rec(&self.0, &mut res, 1, 0);
        res
    }
}

impl std::fmt::Debug for MinSegTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut i = 1;
        writeln!(f, "MinSegTree {{")?;
        write!(f, "    ({}/{})", self.0[0].delta, self.0[0].min)?;
        while i < self.0.len() {
            write!(f, "\n   ")?;
            for entry in &self.0[i..2 * i] {
                write!(f, " {}/{}", entry.delta, entry.min)?;
            }
            i *= 2;
        }
        write!(f, "\n}}")
    }
}

/// Sorts the given sequence by swapping adjacent levels only. For every swap
/// operation, `swap` is called with the smaller index.
fn bubble_sort(seq: &mut [LevelNo], mut swap: impl FnMut(LevelNo)) {
    let mut n = seq.len();
    while n > 1 {
        let mut new_n = 0;
        for i in 1..n {
            if seq[i - 1] > seq[i] {
                seq.swap(i - 1, i);
                swap((i - 1) as LevelNo);
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
fn concurrent_bubble_sort<M, F>(manager: &M, seq: Vec<LevelNo>, swap: F)
where
    M: HasWorkers,
    F: Fn(LevelNo) + Sync,
{
    if IsSorted::is_sorted(&mut seq.iter()) {
        return; // avoid spawning a new thread in this case
    }

    let (task_sender, task_receiver) = flume::unbounded::<u32>();
    let (done_sender, done_receiver) = flume::unbounded::<u32>();

    std::thread::spawn(move || {
        let swap = |seq: &mut [u32], blocked: &mut [bool], upper: usize| {
            let lower = upper + 1;
            seq.swap(upper, lower);
            blocked[upper] = true;
            blocked[lower] = true;
            task_sender.send(upper as u32).unwrap();
        };

        let mut seq = seq;
        let mut blocked = vec![false; seq.len()];
        let mut i = 0;
        let mut tasks = 0u32;
        while i + 1 < seq.len() {
            if seq[i] > seq[i + 1] {
                swap(&mut seq, &mut blocked, i);
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
                swap(&mut seq, &mut blocked, i - 1);
                tasks += 1;
            }
            if i + 2 < seq.len() && seq[i + 1] > seq[i + 2] && !blocked[i + 2] {
                swap(&mut seq, &mut blocked, i + 1);
                tasks += 1;
            }
        }
        debug_assert!(IsSorted::is_sorted(&mut seq.iter()));
    });

    manager.workers().broadcast(|_| {
        while let Ok(upper) = task_receiver.recv() {
            swap(upper);
            done_sender.send(upper).unwrap();
        }
    });
}

#[cfg(test)]
mod test {
    use oxidd_test_utils::edge::DummyManager;
    use std::sync::atomic::Ordering::Relaxed;

    use super::*;

    macro_rules! segtree {
        ($($d:expr,$m:expr);*;) => {
            MinSegTree(vec![
                MinSegTreeEntry { delta: 0, min: 0 },
                $(MinSegTreeEntry { delta: $d, min: $m }),*
            ])
        };
    }

    #[test]
    fn test_sort_order() {
        assert_eq!(sort_order(4, [0, 1, 2, 3]), vec![0, 1, 2, 3]);
        assert_eq!(sort_order(4, [1, 0, 2, 3]), vec![1, 0, 2, 3]);
        assert_eq!(sort_order(4, [0, 2, 3, 1]), vec![0, 3, 1, 2]);

        assert_eq!(
            sort_order(10, [6, 3, 0, 4, 1, 9]),
            vec![2, 4, 0, 1, 3, 5, 0, 5, 5, 5]
        );
        assert_eq!(
            sort_order(8, [7, 3, 0, 5, 6, 1]),
            vec![2, 5, 0, 1, 3, 3, 4, 0]
        );
    }

    #[test]
    fn test_segtree_new() {
        assert_eq!(MinSegTree::new(2..3), segtree![2,2;]);

        assert_eq!(
            MinSegTree::new([-2, 5, 3, -2, -3, 4].into_iter()),
            segtree![
                0,-3;
                0,-2;                   0,-3;
                0,-2;       0,-2;       0,-3;       0,i32::MAX;
                -2,-2; 5,5; 3,3; -2,-2; -3,-3; 4,4; i32::MAX,i32::MAX; i32::MAX,i32::MAX;
            ]
        );
    }

    #[test]
    fn test_segtree_min_idx() {
        assert_eq!(MinSegTree::new(-42..42).min_index(), 0);
        assert_eq!(MinSegTree::new([2, -3, 1].into_iter()).min_index(), 1);
        assert_eq!(MinSegTree::new([2, -3, -11].into_iter()).min_index(), 2);
        assert_eq!(MinSegTree::new([4, 3, 2, 0].into_iter()).min_index(), 3);
        assert_eq!(MinSegTree::new([4, 0, 2, 0].into_iter()).min_index(), 1);
    }

    #[test]
    fn test_segtree_add_split() {
        let mut st = MinSegTree::new([4, 0, 2, 1].into_iter());
        st.add_split(0, -2, 2);
        assert_eq!(
            st,
            segtree![
                2,2;
                0,0;      0,1;
                4,4; 0,0; 2,2; 1,1;
            ]
        );
        st.add_split(4, -1, 2);
        assert_eq!(
            st,
            segtree![
                1,1;
                0,0;      0,1;
                4,4; 0,0; 2,2; 1,1;
            ]
        );
        st.add_split(2, 1, -2);
        assert_eq!(
            st,
            segtree![
                1,0;
                1,1;      -2,-1;
                4,4; 0,0; 2,2; 1,1;
            ]
        );
        st.add_split(1, -4, 1);
        assert_eq!(
            st,
            segtree![
                1,1;
                1,1;      -1,0;
                0,0; 1,1; 2,2; 1,1;
            ]
        );

        assert_eq!(st.min_index(), 3);
    }

    #[test]
    fn test_segtree_add_split_unused() {
        let mut st = MinSegTree::new([0, 1, 2, 3, 4].into_iter());
        st.add_split(5, 1, -1);
        assert_eq!(
            st,
            segtree![
                0,1;
                1,1;                0,5;
                0,0;      0,2;      0,5;                    0,i32::MAX;
                0,0; 1,1; 2,2; 3,3; 5,5; i32::MAX,i32::MAX; i32::MAX,i32::MAX; i32::MAX,i32::MAX;
            ]
        );
        assert_eq!(st.min_index(), 0);
    }

    macro_rules! bubble_sort_test_case {
        ($($xs:expr),*) => {
            {
                let mut seq = [$($xs),*];
                let mut seq2 = [$($xs),*];
                bubble_sort(&mut seq, |i| seq2.swap(i as usize, i as usize + 1));
                let mut sorted = [$($xs),*];
                sorted.sort();
                assert_eq!(seq, sorted);
                assert_eq!(seq, seq2);
            }
        };
    }

    #[test]
    fn test_bubble_sort() {
        bubble_sort_test_case![];
        bubble_sort_test_case![0];
        bubble_sort_test_case![1];
        bubble_sort_test_case![0, 1, 2];
        bubble_sort_test_case![1, 4, 5];
        bubble_sort_test_case![3, 2, 0];

        bubble_sort(&mut [0, 0, 1, 1], |_| panic!("sort is unstable"));
    }

    macro_rules! atomic_u32_array {
        ($($xs:expr),*) => {
            [$(::std::sync::atomic::AtomicU32::new($xs)),*]
        };
    }

    macro_rules! concurrent_bubble_sort_test_case {
        ($($xs:expr),*) => {
            {
                let seq: &[::std::sync::atomic::AtomicU32] = &atomic_u32_array![$($xs),*];
                concurrent_bubble_sort(&DummyManager, vec![$($xs),*], |i| {
                    let a = seq[i as usize].load(Relaxed);
                    let b = seq[(i + 1) as usize].swap(a, Relaxed);
                    seq[i as usize].store(b, Relaxed);
                });
                let mut sorted = [$($xs),*];
                sorted.sort();
                assert!(seq
                    .into_iter()
                    .zip(sorted)
                    .all(|(a, b)| a.load(Relaxed) == b));
            }
        };
    }

    #[test]
    fn test_concurrent_bubble_sort() {
        concurrent_bubble_sort_test_case![];
        concurrent_bubble_sort_test_case![1];
        concurrent_bubble_sort_test_case![0, 1, 2];
        concurrent_bubble_sort_test_case![1, 4, 5];
        concurrent_bubble_sort_test_case![3, 2, 0];

        concurrent_bubble_sort(&DummyManager, vec![0, 0, 1, 1], |_| {
            panic!("sort is unstable")
        });
    }
}
