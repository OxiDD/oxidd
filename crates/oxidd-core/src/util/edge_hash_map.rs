//! [`HashMap`] mapping from edges to values of another type. Performs the
//! necessary management of [`Edge`][super::Edge]s.

use std::collections::hash_map;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::mem::ManuallyDrop;

use crate::Manager;

/// [`HashMap`] mapping from edges to values of type `V`
///
/// Internally, this map stores a [`Manager`] reference such that it can clone
/// or drop edges accordingly. There is no need to manually drop all contained
/// keys before dropping the map.
pub struct EdgeHashMap<'a, M: Manager, V, S> {
    manager: &'a M,
    map: ManuallyDrop<HashMap<ManuallyDrop<M::Edge>, V, S>>,
}

impl<'a, M: Manager, V, S: Default + BuildHasher> EdgeHashMap<'a, M, V, S> {
    /// Create a new edge map
    #[inline]
    pub fn new(manager: &'a M) -> Self {
        EdgeHashMap {
            manager,
            map: ManuallyDrop::new(HashMap::with_hasher(S::default())),
        }
    }

    /// Create a new edge map with `capacity`
    #[inline]
    pub fn with_capacity(manager: &'a M, capacity: usize) -> Self {
        EdgeHashMap {
            manager,
            map: ManuallyDrop::new(HashMap::with_capacity_and_hasher(capacity, S::default())),
        }
    }

    /// Returns the number of elements in the map
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` iff the map has no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Reserves capacity for at least `additional` more elements
    ///
    /// The collection may reserve more space to speculatively avoid frequent
    /// reallocations. After calling `reserve()`, the capacity will be greater
    /// than or equal to `self.len() + additional`. Does nothing if capacity is
    /// already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows [`usize`].
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
    }

    /// Get a reference to the value for `edge` (if present)
    #[inline]
    pub fn get(&self, key: &M::Edge) -> Option<&V> {
        // SAFETY: `ManuallyDrop<T>` has the same representation as `T`
        self.map
            .get(unsafe { std::mem::transmute::<_, &ManuallyDrop<M::Edge>>(key) })
    }

    /// Get a mutable reference to the value for `edge` (if present)
    #[inline]
    pub fn get_mut(&mut self, key: &M::Edge) -> Option<&mut V> {
        // SAFETY: `ManuallyDrop<T>` has the same representation as `T`
        self.map
            .get_mut(unsafe { std::mem::transmute::<_, &ManuallyDrop<M::Edge>>(key) })
    }

    /// Insert a key-value pair into the map
    ///
    /// If the map did not have this key present, the key is cloned, and `None`
    /// is returned. If the map did have this key present, the value is updated,
    /// and the old value is returned.
    pub fn insert(&mut self, key: &M::Edge, value: V) -> Option<V> {
        // SAFETY: `key` is valid for reads. If the edge is actually inserted
        // into the map, then we clone the edge (and forget the clone),
        // otherwise the map forgets it.
        // TODO: Do we need to add (safety) requirements to the `Edge`/`Manager` trait?
        let edge = ManuallyDrop::new(unsafe { std::ptr::read(key) });
        match self.map.insert(edge, value) {
            Some(old) => Some(old),
            None => {
                std::mem::forget(self.manager.clone_edge(key));
                None
            }
        }
    }

    /// Remove the entry for the given key (if present)
    ///
    /// Returns the value that was previously stored in the map, or `None`,
    /// respectively.
    pub fn remove(&mut self, key: &M::Edge) -> Option<V> {
        // SAFETY: `ManuallyDrop<T>` has the same representation as `T`
        match self
            .map
            .remove_entry(unsafe { std::mem::transmute::<_, &ManuallyDrop<M::Edge>>(key) })
        {
            Some((key, value)) => {
                self.manager.drop_edge(ManuallyDrop::into_inner(key));
                Some(value)
            }
            None => None,
        }
    }

    /// Iterator visiting all key-value pairs in arbitrary order
    ///
    /// The item type is `(&M::Edge, &V)`.
    pub fn iter(&self) -> Iter<'_, M, V> {
        Iter(self.map.iter())
    }

    /// Mutable iterator visiting all key-value pairs in arbitrary order
    ///
    /// The item type is `(&M::Edge, &mut V)`.
    pub fn iter_mut(&mut self) -> IterMut<'_, M, V> {
        IterMut(self.map.iter_mut())
    }
}

impl<'a, M: Manager, V: Clone, S: Default + BuildHasher> Clone for EdgeHashMap<'a, M, V, S> {
    fn clone(&self) -> Self {
        let mut map = HashMap::with_capacity_and_hasher(self.len(), S::default());
        for (k, v) in self.map.iter() {
            let _res = map.insert(ManuallyDrop::new(self.manager.clone_edge(k)), v.clone());
            debug_assert!(_res.is_none());
        }
        Self {
            manager: self.manager,
            map: ManuallyDrop::new(map),
        }
    }
}

impl<'a, M: Manager, V, S> Drop for EdgeHashMap<'a, M, V, S> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self.map` is never used again
        for (k, _) in unsafe { ManuallyDrop::take(&mut self.map) } {
            self.manager.drop_edge(ManuallyDrop::into_inner(k));
        }
    }
}

impl<'a, M: Manager, V, S> IntoIterator for EdgeHashMap<'a, M, V, S> {
    type Item = (M::Edge, V);

    type IntoIter = IntoIter<M, V>;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        // SAFETY: `self.map` is never used again (we forget `self`)
        let map = unsafe { ManuallyDrop::take(&mut self.map) };
        std::mem::forget(self);
        IntoIter(map.into_iter())
    }
}

/// Owning iterator over the entries of an [`EdgeHashMap`]
pub struct IntoIter<M: Manager, V>(hash_map::IntoIter<ManuallyDrop<M::Edge>, V>);

impl<M: Manager, V> Iterator for IntoIter<M, V> {
    type Item = (M::Edge, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some((key, value)) => Some((ManuallyDrop::into_inner(key), value)),
            None => None,
        }
    }
}

/// Iterator over the entries of an [`EdgeHashMap`]
///
/// Created by [`EdgeHashMap::iter()`], see its documentation for more details.
pub struct Iter<'a, M: Manager, V>(hash_map::Iter<'a, ManuallyDrop<M::Edge>, V>);

impl<'a, M: Manager, V> Iterator for Iter<'a, M, V> {
    type Item = (&'a M::Edge, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some((key, value)) => Some((key, value)),
            None => None,
        }
    }
}

impl<'a, 'b, M: Manager, V, S> IntoIterator for &'b EdgeHashMap<'a, M, V, S> {
    type Item = (&'b M::Edge, &'b V);

    type IntoIter = Iter<'b, M, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter(self.map.iter())
    }
}

/// Mutable iterator over the entries of an [`EdgeHashMap`]
///
/// Created by [`EdgeHashMap::iter_mut()`], see its documentation for more
/// details.
pub struct IterMut<'a, M: Manager, V>(hash_map::IterMut<'a, ManuallyDrop<M::Edge>, V>);

impl<'a, M: Manager, V> Iterator for IterMut<'a, M, V> {
    type Item = (&'a M::Edge, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some((key, value)) => Some((key, value)),
            None => None,
        }
    }
}

impl<'a, 'b, M: Manager, V, S> IntoIterator for &'b mut EdgeHashMap<'a, M, V, S> {
    type Item = (&'b M::Edge, &'b mut V);

    type IntoIter = IterMut<'b, M, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IterMut(self.map.iter_mut())
    }
}
