/// A segment tree that allows additive range updates (in O(log n)) and queries
/// for the index of some minimal element (in O(log n), could be optimized to
/// O(1) by adding a `min_index` field to `MinSegTreeEntry`)
#[derive(PartialEq, Eq)]
pub struct MinSegTree(Vec<MinSegTreeEntry>);

#[derive(Copy, Clone, PartialEq, Eq)]
struct MinSegTreeEntry {
    delta: i32,
    /// The minimum value of the children + self.delta
    min: i32,
}

impl MinSegTreeEntry {
    #[inline]
    fn update(&mut self, delta: i32) {
        if self.min != i32::MAX {
            self.delta += delta;
            self.min += delta;
        }
    }
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

    pub fn new(data: impl ExactSizeIterator<Item = i32>) -> Self {
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
    pub fn add_split(&mut self, i: usize, left: i32, right: i32) {
        let t = &mut self.0[..];
        let mut size = t.len() / 2;
        assert!(i <= size);

        // Special case: borders. Either there is nothing to the left to modify,
        // or nothing to the right.
        if i == 0 {
            t[1].update(right);
            return;
        }
        if i == size {
            t[1].update(left);
            return;
        }

        let mut node = 1;
        // vertical distance between node and the leaf level
        let mut levels_from_bot = size.trailing_zeros();

        loop {
            size /= 2;
            if i & (size - 1) /* i % size */ == 0 {
                break;
            }
            levels_from_bot -= 1;
            debug_assert_ne!(levels_from_bot, 0);
            if Self::is_left_child(i >> levels_from_bot) {
                node = Self::left(node);
                t[node + 1].update(right);
            } else {
                node = Self::right(node);
                t[node - 1].update(left);
            }
        }
        // Now, `node` is exactly the node such that every leaf in the left
        // subtree represents an element to the left of the split and every leaf
        // in the right subtree represents an element to the right of the split.
        t[Self::left(node)].update(left);
        t[Self::right(node)].update(right);

        loop {
            let l = Self::left(node);
            let r = Self::right(node);
            t[node].min = t[node].delta + std::cmp::min(t[l].min, t[r].min);
            if node == 1 {
                break;
            }
            node = Self::parent(node);
        }
    }

    /// Get the index of the minimal element with the lowest index
    pub fn min_index(&self) -> usize {
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

#[cfg(test)]
mod test {
    use super::{MinSegTree, MinSegTreeEntry};

    macro_rules! segtree {
        ($($d:expr,$m:expr);*;) => {
            MinSegTree(vec![
                MinSegTreeEntry { delta: 0, min: 0 },
                $(MinSegTreeEntry { delta: $d, min: $m }),*
            ])
        };
    }

    #[test]
    fn test_new() {
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
    fn test_min_idx() {
        assert_eq!(MinSegTree::new(-42..42).min_index(), 0);
        assert_eq!(MinSegTree::new([2, -3, 1].into_iter()).min_index(), 1);
        assert_eq!(MinSegTree::new([2, -3, -11].into_iter()).min_index(), 2);
        assert_eq!(MinSegTree::new([4, 3, 2, 0].into_iter()).min_index(), 3);
        assert_eq!(MinSegTree::new([4, 0, 2, 0].into_iter()).min_index(), 1);
    }

    #[test]
    fn test_add_split() {
        let mut st = MinSegTree::new([4, 0, 2, 1].into_iter());
        st.add_split(0, -2, 2);
        assert_eq!(st.proj()[..], [6, 2, 4, 3]);
        assert_eq!(
            st,
            segtree![
                2,2;
                0,0;      0,1;
                4,4; 0,0; 2,2; 1,1;
            ]
        );
        st.add_split(4, -1, 2);
        assert_eq!(st.proj()[..], [5, 1, 3, 2]);
        assert_eq!(
            st,
            segtree![
                1,1;
                0,0;      0,1;
                4,4; 0,0; 2,2; 1,1;
            ]
        );
        st.add_split(2, 1, -2);
        assert_eq!(st.proj()[..], [6, 2, 1, 0]);
        assert_eq!(
            st,
            segtree![
                1,0;
                1,1;      -2,-1;
                4,4; 0,0; 2,2; 1,1;
            ]
        );
        st.add_split(1, -4, 1);
        assert_eq!(st.proj()[..], [2, 3, 2, 1]);
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
    fn test_add_split_unused() {
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
}
