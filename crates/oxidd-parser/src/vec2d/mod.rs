//! Compact representation of a `Vec<Vec<T>>`

use std::fmt;
use std::iter::FusedIterator;

pub(crate) mod with_metadata;

/// Compact representation of a `Vec<Vec<T>>`
#[derive(Clone, PartialEq, Eq)]
pub struct Vec2d<T>(with_metadata::Vec2d<T, 0>);

impl<T> Default for Vec2d<T> {
    #[inline(always)]
    fn default() -> Self {
        Self(Default::default())
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
        Self(with_metadata::Vec2d::with_capacity(vectors, elements_sum))
    }

    /// Reserve space for at least `additional` more inner vectors. Note that
    /// this will not reserve space for elements of the inner vectors. To that
    /// end, use [`Self::reserve_elements()`].
    ///
    /// This is essentially a wrapper around [`Vec::reserve()`], so the
    /// documentation there provides more details.
    #[inline(always)]
    pub fn reserve_vectors(&mut self, additional: usize) {
        self.0.reserve_vectors(additional);
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
        self.0.reserve_elements(additional);
    }

    /// Get the number of inner vectors
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// `true` iff there are no inner vectors
    ///
    /// Equivalent to [`self.len() == 0`][Self::len]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the vector at `index` if there is one
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[T]> {
        Some(self.0.get(index)?.1)
    }
    /// Get the vector at `index` if there is one
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.0.get_mut(index)
    }
    /// Get the first vector if there is one
    #[inline]
    pub fn first(&self) -> Option<&[T]> {
        Some(self.0.first()?.1)
    }
    /// Get the first vector if there is one
    #[inline(always)]
    pub fn first_mut(&mut self) -> Option<&mut [T]> {
        self.0.first_mut()
    }
    /// Get the last vector if there is one
    #[inline]
    pub fn last(&self) -> Option<&[T]> {
        Some(self.0.last()?.1)
    }
    /// Get the last vector if there is one
    #[inline(always)]
    pub fn last_mut(&mut self) -> Option<&mut [T]> {
        self.0.last_mut()
    }

    /// Iterate over the inner vector
    #[inline(always)]
    pub fn iter(&self) -> Vec2dIter<'_, T> {
        Vec2dIter(self.0.iter())
    }

    /// Add an empty inner vector
    ///
    /// Elements can be added to that vector using [`Self::push_element()`]
    #[inline(always)]
    pub fn push_vec(&mut self) {
        self.0.push_vec(0);
    }

    /// Remove the last vector (if there is one)
    ///
    /// Returns true iff the outer vector was non-empty before removal.
    #[inline(always)]
    pub fn pop_vec(&mut self) -> bool {
        self.0.pop_vec()
    }

    /// Truncate the outer vector to `len` inner vectors
    ///
    /// If `len` is greater or equal to [`self.len()`][Self::len()], this is a
    /// no-op.
    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }

    /// Push an element to the last vector
    ///
    /// The vector list must be non-empty.
    #[track_caller]
    #[inline(always)]
    pub fn push_element(&mut self, element: T) {
        self.0.push_element(element);
    }

    /// Extend the last vector by the elements from `iter`
    ///
    /// There must be at least one inner vector (i.e.,
    /// [`!self.is_empty()`][Self::is_empty()]).
    #[track_caller]
    #[inline(always)]
    pub fn push_elements(&mut self, iter: impl IntoIterator<Item = T>) {
        self.0.push_elements(iter);
    }

    /// Get a slice containing all elements, ignoring the boundaries of the
    /// inner vectors
    #[inline(always)]
    pub fn all_elements(&self) -> &[T] {
        self.0.all_elements()
    }
    /// Get a slice containing all elements, ignoring the boundaries of the
    /// inner vectors
    #[inline(always)]
    pub fn all_elements_mut(&mut self) -> &mut [T] {
        self.0.all_elements_mut()
    }
}

impl<T: Clone> Vec2d<T> {
    /// Resize the last vector in-place such that its length is equal to
    /// `new_len`
    ///
    /// If `new_len` is greater than the current length, the vector is extended
    /// by the difference, with each additional slot filled with `value`. If
    /// `new_len` is less than `len`, the `Vec` is simply truncated.
    #[track_caller]
    #[inline(always)]
    pub fn resize_last(&mut self, new_len: usize, value: T) {
        self.0.resize_last(new_len, value);
    }
}

#[cold]
#[inline(never)]
fn panic_bounds_check(len: usize, index: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}");
}

impl<T> std::ops::Index<usize> for Vec2d<T> {
    type Output = [T];

    #[track_caller]
    #[inline(always)]
    fn index(&self, index: usize) -> &[T] {
        if let Some(v) = self.get(index) {
            return v;
        }
        panic_bounds_check(self.len(), index)
    }
}
impl<T> std::ops::IndexMut<usize> for Vec2d<T> {
    #[track_caller]
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut [T] {
        let len = self.len(); // moved up here due to borrow checker issues
        if let Some(v) = self.get_mut(index) {
            return v;
        }
        panic_bounds_check(len, index)
    }
}

impl<'a, T: Copy> Extend<&'a [T]> for Vec2d<T> {
    fn extend<I: IntoIterator<Item = &'a [T]>>(&mut self, iter: I) {
        self.0.extend(iter.into_iter().map(|i| (0, i)));
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
#[derive(Clone)]
pub struct Vec2dIter<'a, T>(with_metadata::Vec2dIter<'a, T, 0>);

impl<'a, T> Iterator for Vec2dIter<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        Some(self.0.next()?.1)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    // performance optimization

    #[inline(always)]
    fn count(self) -> usize {
        self.0.count()
    }
    #[inline(always)]
    fn last(self) -> Option<&'a [T]> {
        Some(self.0.last()?.1)
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a [T]> {
        Some(self.0.nth(n)?.1)
    }
}

impl<T> FusedIterator for Vec2dIter<'_, T> {}

impl<T> ExactSizeIterator for Vec2dIter<'_, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, T> DoubleEndedIterator for Vec2dIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        Some(self.0.next_back()?.1)
    }

    // performance optimization

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<&'a [T]> {
        Some(self.0.nth_back(n)?.1)
    }
}
