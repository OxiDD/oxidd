//! Compact representation of a `Vec<Vec<T>>`

use std::fmt;
use std::iter::FusedIterator;

/// Compact representation of a `Vec<Vec<T>>`
#[derive(Clone, PartialEq, Eq)]
pub struct Vec2d<T> {
    /// Start indices of the inner vectors in `data`
    ///
    /// If the 2d vector is non-empty, the first element should be 0, but this
    /// is not a requirement for correctness. With debug assertions disabled,
    /// the user could push elements before creating the first inner vector.
    index: Vec<usize>,
    data: Vec<T>,
}

impl<T> Default for Vec2d<T> {
    fn default() -> Self {
        Self {
            index: Default::default(),
            data: Default::default(),
        }
    }
}

impl<T> Vec2d<T> {
    /// Create a new empty `Vec2d`.
    ///
    /// This is equivalent to
    /// [`Self::with_capacity(0, 0)`][Self::with_capacity].
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new empty `Vec2d` with at least the specified capacities.
    ///
    /// A `Vec2d` internally contains two [`Vec`]s, one for the elements of all
    /// the inner vectors and one for the indices where these inner vectors
    /// begin. `vectors` is the capacity for the latter and `elements_sum` is
    /// the capacity for the former `Vec`.
    #[inline(always)]
    pub fn with_capacity(vectors: usize, elements_sum: usize) -> Self {
        Self {
            index: Vec::with_capacity(vectors),
            data: Vec::with_capacity(elements_sum),
        }
    }

    /// Reserve space for at least `additional` more inner vectors. Note that
    /// this will not reserve space for elements of the inner vectors. To that
    /// end, use [`Self::reserve_elements()`].
    ///
    /// This is essentially a wrapper around [`Vec::reserve()`], so the
    /// documentation there provides more details.
    #[inline(always)]
    pub fn reserve_vectors(&mut self, additional: usize) {
        self.index.reserve(additional);
    }
    /// Reserve space for at least `additional` more elements in the inner
    /// vectors. The space is shared between the inner vectors: After reserving
    /// space for `n` additional elements, pushing, e.g., two vectors with `k1 +
    /// k2 <= n` elements will not lead to a reallocation.
    ///
    /// This is essentially a wrapper around [`Vec::reserve()`], so the
    /// documentation there provides more details.
    #[inline(always)]
    pub fn reserve_elements(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Get the number of inner vectors
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.index.len()
    }
    /// `true` iff there are no inner vectors
    ///
    /// Equivalent to [`self.len() == 0`][Self::len]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get the vector at `index` if there is one
    pub fn get(&self, index: usize) -> Option<&[T]> {
        if index + 1 < self.index.len() {
            Some(&self.data[self.index[index]..self.index[index + 1]])
        } else if index < self.index.len() {
            Some(&self.data[self.index[index]..])
        } else {
            None
        }
    }
    /// Get the vector at `index` if there is one
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        if index + 1 < self.index.len() {
            Some(&mut self.data[self.index[index]..self.index[index + 1]])
        } else if index < self.index.len() {
            Some(&mut self.data[self.index[index]..])
        } else {
            None
        }
    }
    /// Get the first vector if there is one
    pub fn first(&self) -> Option<&[T]> {
        if self.index.len() >= 2 {
            Some(&self.data[self.index[0]..self.index[1]])
        } else if self.index.len() == 1 {
            Some(&self.data[self.index[0]..])
        } else {
            None
        }
    }
    /// Get the first vector if there is one
    pub fn first_mut(&mut self) -> Option<&mut [T]> {
        if self.index.len() >= 2 {
            Some(&mut self.data[self.index[0]..self.index[1]])
        } else if self.index.len() == 1 {
            Some(&mut self.data[self.index[0]..])
        } else {
            None
        }
    }
    /// Get the last vector if there is one
    pub fn last(&self) -> Option<&[T]> {
        let start = *self.index.last()?;
        Some(&self.data[start..])
    }
    /// Get the last vector if there is one
    pub fn last_mut(&mut self) -> Option<&mut [T]> {
        let start = *self.index.last()?;
        Some(&mut self.data[start..])
    }

    /// Iterate over the inner vector
    #[inline(always)]
    pub fn iter(&self) -> Vec2dIter<T> {
        Vec2dIter {
            index: &self.index,
            end_index: self.data.len(),
            data: &self.data,
        }
    }

    /// Add an empty inner vector
    ///
    /// Elements can be added to that vector using [`Self::push_element()`]
    pub fn push_vec(&mut self) {
        self.index.push(self.data.len());
    }

    /// Remove the last vector (if there is one)
    ///
    /// Returns true iff the outer vector was non-empty before removal.
    pub fn pop_vec(&mut self) -> bool {
        match self.index.pop() {
            Some(i) => {
                self.data.truncate(i);
                true
            }
            None => false,
        }
    }

    /// Truncate the outer vector to `len` inner vectors
    ///
    /// If `len` is greater or equal to [`self.len()`][Self::len()], this is a
    /// no-op.
    pub fn truncate(&mut self, len: usize) {
        if len < self.index.len() {
            self.data.truncate(self.index[len]);
            self.index.truncate(len);
        }
    }

    /// Push an element to the last vector
    ///
    /// The vector list must be non-empty.
    pub fn push_element(&mut self, element: T) {
        debug_assert!(
            !self.is_empty(),
            "The outer vector is empty. Use push_vec() to create the first vector."
        );
        self.data.push(element);
    }

    /// Extend the last vector by the elements from `iter`
    ///
    /// There must be at least one inner vector (i.e.,
    /// [`!self.is_empty()`][Self::is_empty()]).
    pub fn push_elements(&mut self, iter: impl IntoIterator<Item = T>) {
        debug_assert!(
            !self.is_empty(),
            "The outer vector is empty. Use push_vec() to create the first vector."
        );
        self.data.extend(iter)
    }

    /// Get a slice containing all elements, ignoring the boundaries of the
    /// inner vectors
    #[inline(always)]
    pub fn all_elements(&self) -> &[T] {
        &self.data
    }
    /// Get a slice containing all elements, ignoring the boundaries of the
    /// inner vectors
    #[inline(always)]
    pub fn all_elements_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> std::ops::Index<usize> for Vec2d<T> {
    type Output = [T];

    #[track_caller]
    #[inline]
    fn index(&self, index: usize) -> &[T] {
        if index + 1 < self.index.len() {
            &self.data[self.index[index]..self.index[index + 1]]
        } else {
            assert!(index < self.index.len(), "index out of bounds");
            &self.data[self.index[index]..]
        }
    }
}
impl<T> std::ops::IndexMut<usize> for Vec2d<T> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut [T] {
        if index + 1 < self.index.len() {
            &mut self.data[self.index[index]..self.index[index + 1]]
        } else {
            assert!(index < self.index.len(), "index out of bounds");
            &mut self.data[self.index[index]..]
        }
    }
}

impl<'a, T: Copy> Extend<&'a [T]> for Vec2d<T> {
    fn extend<I: IntoIterator<Item = &'a [T]>>(&mut self, iter: I) {
        let it = iter.into_iter();
        let min_len = it.size_hint().0;
        self.index.reserve(min_len);
        self.data.reserve(min_len * 4);
        for v in it {
            self.index.push(self.data.len());
            self.data.extend(v);
        }
    }
}

impl<'a, T: Copy> FromIterator<&'a [T]> for Vec2d<T> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = &'a [T]>>(iter: I) -> Self {
        let mut vec = Vec2d::new();
        vec.extend(iter);
        vec
    }
}

impl<T: fmt::Debug> fmt::Debug for Vec2d<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Iterator over the inner vectors of a [`Vec2d`]
pub struct Vec2dIter<'a, T> {
    /// Remaining `index`
    index: &'a [usize],
    /// Index of the first vector not referenced by `index` (or `data.len()`)
    end_index: usize,
    /// Full `data`
    data: &'a [T],
}

impl<'a, T> Iterator for Vec2dIter<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        match self.index {
            [start, end, ..] => {
                self.index = &self.index[1..];
                Some(&self.data[*start..*end])
            }
            [start] => {
                self.index = &self.index[1..];
                Some(&self.data[*start..self.end_index])
            }
            [] => None,
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    // performance optimization

    #[inline(always)]
    fn count(self) -> usize {
        self.len()
    }
    #[inline(always)]
    fn last(self) -> Option<&'a [T]> {
        match self.index {
            [.., last] => Some(&self.data[*last..self.end_index]),
            [] => None,
        }
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a [T]> {
        let len = self.index.len();
        if n >= len {
            self.index = &self.index[len..];
            None
        } else {
            let start = self.index[n];
            self.index = &self.index[n + 1..];
            let end = self.index.first().copied().unwrap_or(self.end_index);
            Some(&self.data[start..end])
        }
    }
}

impl<'a, T> FusedIterator for Vec2dIter<'a, T> {}

impl<'a, T> ExactSizeIterator for Vec2dIter<'a, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.index.len()
    }
}

impl<'a, T> DoubleEndedIterator for Vec2dIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        match self.index {
            [rem @ .., last] => {
                self.index = rem;
                let res = &self.data[*last..self.end_index];
                self.end_index = *last;
                Some(res)
            }
            [] => None,
        }
    }

    // performance optimization

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<&'a [T]> {
        let len = self.index.len();
        if n >= len {
            self.index = &self.index[..0];
            // ignore `self.end_index`
            None
        } else {
            let end = if n == 0 {
                self.end_index
            } else {
                self.index[len - n]
            };
            let start = self.index[len - n - 1];
            self.end_index = start;
            let start = self.index[n];
            self.index = &self.index[..len - n - 1];
            Some(&self.data[start..end])
        }
    }
}
