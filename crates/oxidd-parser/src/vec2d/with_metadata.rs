//! Compact representation of a `Vec<Vec<T>>`, where each inner vector comes
//! with some bits of metadata

use std::fmt;
use std::iter::FusedIterator;

/// Compact representation of a `Vec<Vec<T>>`
#[derive(Clone, PartialEq, Eq)]
pub struct Vec2d<T, const METADATA_BITS: u32> {
    /// Start indices of the inner vectors in `data`
    ///
    /// If the 2d vector is non-empty, the first element should be 0, but this
    /// is not a requirement for correctness. With debug assertions disabled,
    /// the user could push elements before creating the first inner vector.
    index: Vec<usize>,
    data: Vec<T>,
}

impl<T, const METADATA_BITS: u32> Default for Vec2d<T, METADATA_BITS> {
    fn default() -> Self {
        Self {
            index: Default::default(),
            data: Default::default(),
        }
    }
}

impl<T, const METADATA_BITS: u32> Vec2d<T, METADATA_BITS> {
    const METADATA_MASK: usize = (1 << METADATA_BITS) - 1;

    /// Returns (metadata, index)
    #[inline(always)]
    fn decode(val: usize) -> (usize, usize) {
        (val & Self::METADATA_MASK, val >> METADATA_BITS)
    }

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

    /// Get the metadata and vector at `index`, if present
    pub fn get(&self, index: usize) -> Option<(usize, &[T])> {
        let end = if index + 1 < self.index.len() {
            self.index[index + 1] >> METADATA_BITS
        } else if index < self.index.len() {
            self.data.len()
        } else {
            return None;
        };
        let (metadata, start) = Self::decode(self.index[index]);
        Some((metadata, &self.data[start..end]))
    }
    /// Get the vector at `index`, if present
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        let end = if index + 1 < self.index.len() {
            self.index[index + 1] >> METADATA_BITS
        } else if index < self.index.len() {
            self.data.len()
        } else {
            return None;
        };
        let start = self.index[index] >> METADATA_BITS;
        Some(&mut self.data[start..end])
    }
    /// Get the first metadata-vector pair if there is one
    #[inline(always)]
    pub fn first(&self) -> Option<(usize, &[T])> {
        self.get(0)
    }
    /// Get the first vector if there is one
    #[inline(always)]
    pub fn first_mut(&mut self) -> Option<&mut [T]> {
        self.get_mut(0)
    }
    /// Get the last metadata-vector pair if there is one
    pub fn last(&self) -> Option<(usize, &[T])> {
        let (metadata, start) = Self::decode(*self.index.last()?);
        Some((metadata, &self.data[start..]))
    }
    /// Get the last vector if there is one
    pub fn last_mut(&mut self) -> Option<&mut [T]> {
        let start = *self.index.last()? >> METADATA_BITS;
        Some(&mut self.data[start..])
    }

    /// Set the metadata at `index`
    #[track_caller]
    #[inline]
    pub fn set_metadata(&mut self, index: usize, metadata: usize) {
        let val = &mut self.index[index];
        *val = (*val & !Self::METADATA_MASK) | (metadata & Self::METADATA_MASK);
    }

    /// Iterate over the inner vector
    #[inline(always)]
    pub fn iter(&self) -> Vec2dIter<T, METADATA_BITS> {
        Vec2dIter {
            index: &self.index,
            end_index: self.data.len(),
            data: &self.data,
        }
    }

    /// Add an empty inner vector
    ///
    /// Elements can be added to that vector using [`Self::push_element()`]
    #[inline]
    pub fn push_vec(&mut self, metadata: usize) {
        self.index
            .push((self.data.len() << METADATA_BITS) | (metadata & Self::METADATA_MASK));
    }

    /// Remove the last vector (if there is one)
    ///
    /// Returns true iff the outer vector was non-empty before removal.
    pub fn pop_vec(&mut self) -> bool {
        match self.index.pop() {
            Some(i) => {
                self.data.truncate(i >> METADATA_BITS);
                true
            }
            None => false,
        }
    }

    /// Clear the outer vector
    #[inline]
    pub fn clear(&mut self) {
        self.index.clear();
        self.data.clear();
    }

    /// Truncate the outer vector to `len` inner vectors
    ///
    /// If `len` is greater or equal to [`self.len()`][Self::len()], this is a
    /// no-op.
    pub fn truncate(&mut self, len: usize) {
        if len < self.index.len() {
            self.data.truncate(self.index[len] >> METADATA_BITS);
            self.index.truncate(len);
        }
    }

    /// Push an element to the last vector
    ///
    /// The vector list must be non-empty.
    #[track_caller]
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
    #[track_caller]
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

impl<T: Clone, const METADATA_BITS: u32> Vec2d<T, METADATA_BITS> {
    /// Resize the last vector in-place such that its length is equal to
    /// `new_len`
    ///
    /// If `new_len` is greater than the current length, the vector is extended
    /// by the difference, with each additional slot filled with `value`. If
    /// `new_len` is less than `len`, the `Vec` is simply truncated.
    #[track_caller]
    pub fn resize_last(&mut self, new_len: usize, value: T) {
        if let Some(last) = self.index.last().copied() {
            self.data.resize(last + new_len, value);
        }
    }
}

// We could implement `retain` without `T: Copy`, but not efficiently within
// Safe Rust
impl<T: Copy, const METADATA_BITS: u32> Vec2d<T, METADATA_BITS> {
    pub fn retain(&mut self, mut predicate: impl FnMut(usize, &mut [T]) -> bool) {
        let mut index_w = 0;
        let mut data_w = 0;
        for index_r in 0..self.index.len() {
            let (metadata, start) = Self::decode(self.index[index_r]);
            let end = match self.index.get(index_r + 1) {
                Some(i) => *i >> METADATA_BITS,
                None => self.data.len(),
            };

            if predicate(metadata, &mut self.data[start..end]) {
                if index_r == index_w {
                    data_w = end;
                } else {
                    self.data.copy_within(start..end, data_w);
                    self.index[index_w] = (data_w << METADATA_BITS) | metadata;
                    data_w += end - start;
                }
                index_w += 1;
            }
        }
        self.index.truncate(index_w);
        self.data.truncate(data_w);
    }
}

impl<'a, T: Copy, const METADATA_BITS: u32> Extend<(usize, &'a [T])> for Vec2d<T, METADATA_BITS> {
    fn extend<I: IntoIterator<Item = (usize, &'a [T])>>(&mut self, iter: I) {
        let it = iter.into_iter();
        let min_len = it.size_hint().0;
        self.index.reserve(min_len);
        self.data.reserve(min_len * 4);
        for (metadata, v) in it {
            self.push_vec(metadata);
            self.data.extend(v);
        }
    }
}

impl<'a, T: Copy, const METADATA_BITS: u32> FromIterator<(usize, &'a [T])>
    for Vec2d<T, METADATA_BITS>
{
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = (usize, &'a [T])>>(iter: I) -> Self {
        let mut vec = Vec2d::new();
        vec.extend(iter);
        vec
    }
}

impl<T: fmt::Debug, const METADATA_BITS: u32> fmt::Debug for Vec2d<T, METADATA_BITS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Iterator over the inner vectors of a [`Vec2d`]
#[derive(Clone)] // although technically possible, we do not implement Copy (just like
                 // std::slice::Iter) to avoid accidental copying
pub struct Vec2dIter<'a, T, const METADATA_BITS: u32> {
    /// Remaining `index`
    index: &'a [usize],
    /// Index of the first vector not referenced by `index` (or `data.len()`)
    end_index: usize,
    /// Full `data`
    data: &'a [T],
}

impl<T, const METADATA_BITS: u32> Vec2dIter<'_, T, METADATA_BITS> {
    const METADATA_MASK: usize = (1 << METADATA_BITS) - 1;
}

impl<'a, T, const METADATA_BITS: u32> Iterator for Vec2dIter<'a, T, METADATA_BITS> {
    type Item = (usize, &'a [T]);

    #[inline]
    fn next(&mut self) -> Option<(usize, &'a [T])> {
        let (start_raw, end) = match self.index {
            [start, end, ..] => (*start, *end >> METADATA_BITS),
            [start] => (*start, self.end_index),
            [] => return None,
        };
        self.index = &self.index[1..];
        let metadata = start_raw & Self::METADATA_MASK;
        let start = start_raw >> METADATA_BITS;
        Some((metadata, &self.data[start..end]))
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
    fn last(self) -> Option<(usize, &'a [T])> {
        match self.index {
            [.., last] => {
                let metadata = *last & Self::METADATA_MASK;
                let last = *last >> METADATA_BITS;
                Some((metadata, &self.data[last..self.end_index]))
            }
            [] => None,
        }
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<(usize, &'a [T])> {
        let len = self.index.len();
        if n >= len {
            self.index = &self.index[len..];
            None
        } else {
            let start_raw = self.index[n];
            self.index = &self.index[n + 1..];
            let end = if let [end, ..] = self.index {
                *end >> METADATA_BITS
            } else {
                self.end_index
            };
            let metadata = start_raw & Self::METADATA_MASK;
            let start = start_raw >> METADATA_BITS;
            Some((metadata, &self.data[start..end]))
        }
    }
}

impl<T, const METADATA_BITS: u32> FusedIterator for Vec2dIter<'_, T, METADATA_BITS> {}

impl<T, const METADATA_BITS: u32> ExactSizeIterator for Vec2dIter<'_, T, METADATA_BITS> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.index.len()
    }
}

impl<'a, T, const METADATA_BITS: u32> DoubleEndedIterator for Vec2dIter<'a, T, METADATA_BITS> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, &'a [T])> {
        match self.index {
            [rem @ .., last] => {
                self.index = rem;
                let metadata = *last & Self::METADATA_MASK;
                let res = &self.data[(*last >> METADATA_BITS)..self.end_index];
                self.end_index = *last;
                Some((metadata, res))
            }
            [] => None,
        }
    }

    // performance optimization

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<(usize, &'a [T])> {
        let len = self.index.len();
        if n >= len {
            self.index = &self.index[..0];
            // ignore `self.end_index`
            None
        } else {
            let end = if n == 0 {
                self.end_index
            } else {
                self.index[len - n] >> METADATA_BITS
            };
            let start_raw = self.index[len - n - 1];
            let start = start_raw >> METADATA_BITS;
            let metadata = start_raw & Self::METADATA_MASK;
            self.end_index = start;
            self.index = &self.index[..len - n - 1];
            Some((metadata, &self.data[start..end]))
        }
    }
}
