//! Efficient bi-directional mapping between variables and names

use std::collections::{hash_map::Entry, HashMap};
use std::fmt;
use std::iter::FusedIterator;
use std::ops::Range;

use crate::error::DuplicateVarName;
use crate::VarNo;

mod unowned {
    use std::{fmt, ptr::NonNull};

    /// A reference like `&'static T`, but (unsafely) convertible to a [`Box`].
    /// One may also think of it as an unowned [`Box`], or an [`Rc`] without a
    /// counter, i.e., external reasoning about the number of references and
    /// manual dropping.
    ///
    /// In context of the `VarNameMap` implementation, we would like to store
    /// the `str` content only once. We know that there are exactly two
    /// references for each stored name, so there is no need for a reference
    /// counter. However, we cannot use [`Box::leak`], and turn the resulting
    /// `&mut T` into a `&T` without loosing the ability to drop the allocation
    /// later on. This is because Rust's proposed aliasing model forbids turning
    /// a `&T` into a `&mut T` again (which would happen in the [`Drop`]
    /// implementation).
    // Type invariant: An `Unowned<T>` is generally
    // [convertible to a reference](std::ptr#pointer-to-reference-conversion).
    #[derive(Eq)]
    pub struct Unowned<T: ?Sized>(NonNull<T>);

    // SAFETY: `Unowned<T>` behaves like `&T`, and `&T` is `Send` iff `T` is
    // `Sync`
    unsafe impl<T: ?Sized + Sync> Send for Unowned<T> {}
    // SAFETY: `Unowned<T>` behaves like `&T`, and `&T` is `Sync` iff `T` is
    // `Sync`
    unsafe impl<T: ?Sized + Sync> Sync for Unowned<T> {}

    impl<T: ?Sized> Clone for Unowned<T> {
        #[inline(always)]
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T: ?Sized> Copy for Unowned<T> {}

    impl<T: ?Sized> std::ops::Deref for Unowned<T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &T {
            // SAFETY: follows from the type invariant
            unsafe { self.0.as_ref() }
        }
    }
    impl<T: ?Sized> std::borrow::Borrow<T> for Unowned<T> {
        #[inline(always)]
        fn borrow(&self) -> &T {
            self
        }
    }

    impl<T: ?Sized + PartialEq> PartialEq for Unowned<T> {
        #[inline(always)]
        fn eq(&self, other: &Self) -> bool {
            **self == **other
        }
    }
    impl<T: ?Sized + std::hash::Hash> std::hash::Hash for Unowned<T> {
        #[inline(always)]
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            (**self).hash(state);
        }
    }

    impl<T: ?Sized> From<Box<T>> for Unowned<T> {
        #[inline(always)]
        fn from(value: Box<T>) -> Self {
            Self(NonNull::from(Box::leak(value)))
        }
    }
    impl<T: ?Sized> From<&'static T> for Unowned<T> {
        #[inline(always)]
        fn from(value: &'static T) -> Self {
            Self(NonNull::from(value))
        }
    }
    impl<T: ?Sized> Unowned<T> {
        /// Convert an `Unowned<T>` back to a `Box<T>`
        ///
        /// # Safety
        ///
        /// 1. `this` must have been created from a [`Box`]
        /// 2. `this` must be the only existing reference
        /// 3. If `T` is not `Send`, then `into_box` must be called from the
        ///    thread that created the value of type `T`.
        #[inline(always)]
        pub unsafe fn into_box(this: Self) -> Box<T> {
            // SAFETY: ensured by caller, see above
            unsafe { Box::from_raw(this.0.as_ptr()) }
        }
    }

    impl<T: ?Sized + fmt::Debug> fmt::Debug for Unowned<T> {
        #[inline(always)]
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            (**self).fmt(f)
        }
    }
    impl<T: ?Sized + fmt::Display> fmt::Display for Unowned<T> {
        #[inline(always)]
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            (**self).fmt(f)
        }
    }
}
use unowned::Unowned;

/// Bi-directional mapping between variables and names, intended for the
/// implementation of a [`Manager`][crate::Manager].
///
/// The map requires variable names to be unique, but allows unnamed variables
/// (represented by empty strings).
// Type invariant: All `Unowned<str>` values in `index` are created from
// `Box<str>`. Unless the map is borrowed, there each `Unowned<str>` reference
// has an aliasing copy in `names`, and no other copies elsewhere. All
// `Unowned<str>` values in `names` are either a copy of a value in `index` or
// are the result of `Unowned::from("")`.
#[derive(Clone)]
pub struct VarNameMap {
    names: Vec<Unowned<str>>,
    index: HashMap<Unowned<str>, VarNo>,
}

impl Drop for VarNameMap {
    fn drop(&mut self) {
        self.names.clear();
        for (s, _) in self.index.drain() {
            // SAFETY:
            // 1. By the type invariant, `s` has been created from a `Box<str>`
            // 2. Since `self.names` is cleared now, `s` is the only reference
            // 3. `str` is `Send`
            drop(unsafe { Unowned::into_box(s) });
        }
    }
}

impl fmt::Debug for VarNameMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.names.iter().enumerate().filter(|(_, s)| !s.is_empty()))
            .finish()?;
        write!(f, " (length: {})", self.len())
    }
}

impl Default for VarNameMap {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl VarNameMap {
    /// Create an empty map
    #[inline]
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Get the number of variables (including unnamed ones)
    #[inline(always)]
    pub fn len(&self) -> VarNo {
        self.names.len() as VarNo
    }

    /// `true` if and only if [`self.len()`][Self::len()] is 0
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Reserve space for `additional` entries
    ///
    /// Adding the next `additional` entries will not cause an allocation in the
    /// underlying vector and map.
    #[inline]
    pub fn reserve(&mut self, additional: VarNo) {
        let additional = additional.try_into().unwrap_or(usize::MAX);
        self.names.reserve(additional);
        self.index.reserve(additional);
    }

    /// Get the number of named variables
    #[inline(always)]
    pub fn named_count(&self) -> VarNo {
        self.index.len() as VarNo
    }

    /// Add `additional` unnamed variables
    ///
    /// Panics if [`self.len()`][Self::len()] plus `additional` is greater than
    /// to [`VarNo::MAX`].
    #[inline]
    pub fn add_unnamed(&mut self, additional: VarNo) {
        let msg = "too many variables";
        let new_len = self.len().checked_add(additional).expect(msg);
        self.names.resize(new_len.try_into().expect(msg), "".into());
    }

    /// Add fresh variables with names from `names`
    ///
    /// Returns `Ok(range)` on success, where `range` contains the new variable
    /// numbers. If `names` is too long (there would be more than `VarNo::MAX`
    /// variables), the remaining elements are not consumed.
    ///
    /// If a variable name is not unique, this method returns a
    /// [`DuplicateVarName`] error.
    #[track_caller]
    pub fn add_named<T: IntoIterator<Item = S>, S: Into<String>>(
        &mut self,
        names: T,
    ) -> Result<Range<VarNo>, DuplicateVarName> {
        let len_pre = self.names.len() as VarNo;
        let it = names.into_iter();
        let size_hint = it.size_hint().0;
        self.index.reserve(size_hint);
        self.names.reserve(size_hint);

        for (name, v) in it.zip(len_pre..VarNo::MAX) {
            let name: String = name.into();
            if name.is_empty() {
                self.names.push("".into());
                continue;
            }
            let name = name.into_boxed_str().into();
            match self.index.entry(name) {
                Entry::Occupied(entry) => {
                    let present_var = *entry.get();
                    // SAFETY:
                    // 1. `name` has been created from a `Box<str>` above
                    // 2. `name` was not added to the map, so it is the only reference
                    // 3. `str` is `Send`
                    let name: Box<str> = unsafe { Unowned::into_box(name) };
                    return Err(DuplicateVarName {
                        name: name.into_string(),
                        present_var,
                        added_vars: len_pre..self.names.len() as VarNo,
                    });
                }
                Entry::Vacant(entry) => entry.insert(v),
            };
            self.names.push(name);
        }

        Ok(len_pre..self.names.len() as VarNo)
    }

    /// Get the variable number for the given name if present, or add a new
    /// variable
    ///
    /// Returns a pair `(var_no, found)`. If the provided variable name is
    /// empty, this method will always create a fresh variable.
    ///
    /// If a variable with the given name is not present yet, and there is no
    /// variable free in range `0..VarNo::MAX`, then the variable is not added.
    /// Still, the return value is `VarNo::MAX`.
    pub fn get_or_add(&mut self, name: impl Into<String>) -> (VarNo, bool) {
        let name: String = name.into();
        if name.is_empty() {
            let n = self.names.len() as VarNo;
            self.names.push("".into());
            return (n, false);
        }

        let name = name.into_boxed_str().into();
        match self.index.entry(name) {
            Entry::Occupied(entry) => {
                // SAFETY:
                // 1. `name` has been created from a `Box<str>` above
                // 2. `name` was not added to the map, so it is the only reference
                // 3. `str` is `Send`
                drop(unsafe { Unowned::into_box(name) });
                (*entry.get(), true)
            }
            Entry::Vacant(entry) => {
                let n = self.names.len() as VarNo;
                if n == VarNo::MAX {
                    // SAFETY: as above
                    drop(unsafe { Unowned::into_box(name) });
                    return (n, false);
                }
                entry.insert(n);
                self.names.push(name);
                (n, false)
            }
        }
    }

    /// Get the variable number for the given name, if present
    ///
    /// `self.name_to_var("")` always returns `None`.
    #[inline]
    pub fn name_to_var(&self, name: impl AsRef<str>) -> Option<VarNo> {
        self.index.get(name.as_ref()).copied()
    }

    /// Get `var`'s name
    ///
    /// For unnamed vars, this will return the empty string.
    ///
    /// Panics if `var` is greater or equal to [`self.len()`][Self::len()].
    #[inline(always)]
    #[track_caller]
    pub fn var_name(&self, var: VarNo) -> &str {
        &self.names[var as usize]
    }

    /// Label `var` as `name`
    ///
    /// An empty name means that the variable will become unnamed, and cannot be
    /// retrieved via [`Self::name_to_var()`] anymore.
    ///
    /// Returns `Err((name, other_var))` if `name` is not unique (and not `""`).
    ///
    /// Panics if `var` is greater or equal to the number of variables in this
    /// map.
    #[track_caller]
    pub fn set_var_name(
        &mut self,
        var: VarNo,
        name: impl Into<String>,
    ) -> Result<(), DuplicateVarName> {
        let name: String = name.into();
        if name.is_empty() {
            self.index
                .remove(&std::mem::replace(&mut self.names[var as usize], "".into()));
            return Ok(());
        }
        let name = name.into_boxed_str().into();
        match self.index.entry(name) {
            Entry::Occupied(entry) => {
                // SAFETY:
                // 1. `name` has been created from a `Box<str>` above
                // 2. `name` was not added to the map, so it is the only reference
                // 3. `str` is `Send`
                let name = unsafe { Unowned::into_box(name) };
                let present_var = *entry.get();
                if present_var != var {
                    let len = self.len() as VarNo;
                    return Err(DuplicateVarName {
                        name: name.into_string(),
                        present_var,
                        added_vars: len..len,
                    });
                }
            }
            Entry::Vacant(entry) => {
                let prev = std::mem::replace(&mut self.names[var as usize], name);
                entry.insert(var);
                if !prev.is_empty() {
                    // SAFETY:
                    // 1. By the type invariant and since `prev` is not empty, it `prev` has been
                    //    created from a `Box<str>`
                    // 2. `prev` was removed from `self.names`, and its copy in `index` has been
                    //    dropped since `entry.insert(var)`. By the type invariant, it follows that
                    //    `prev` is the only reference.
                    // 3. `str` is `Send`
                    drop(unsafe { Unowned::into_box(prev) });
                }
            }
        };
        Ok(())
    }

    /// Iterate over the variable names (including all unnamed variables)
    pub fn into_names_iter(mut self) -> IntoNamesIter {
        self.index.clear();
        IntoNamesIter(std::mem::take(&mut self.names).into_iter())
    }
}

/// Owning iterator over variable names
// Type invariant: All non-empty `Unowned<str>` values are created from
// `Box<str>`. There are no aliasing copies of these values.
#[derive(Debug)]
pub struct IntoNamesIter(std::vec::IntoIter<Unowned<str>>);

/// Convert an [`Unowned<str>`] back to a [`String`]
///
/// # Safety
///
/// If `s` is non-empty, then the SAFETY conditions for [`Unowned::into_box()`]
/// must be fulfilled.
#[inline(always)]
unsafe fn unowned_to_string(s: Unowned<str>) -> String {
    if s.is_empty() {
        String::new()
    } else {
        // SAFETY: upheld by caller
        unsafe { Unowned::into_box(s) }.into_string()
    }
}

impl Drop for IntoNamesIter {
    fn drop(&mut self) {
        for &name in self.0.as_slice() {
            // SAFETY follows from type invariant
            drop(unsafe { unowned_to_string(name) });
        }
    }
}

impl Iterator for IntoNamesIter {
    type Item = String;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let name = self.0.next()?;
        // SAFETY follows from type invariant
        Some(unsafe { unowned_to_string(name) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let name = self.0.nth(n)?;
        // SAFETY follows from type invariant
        Some(unsafe { unowned_to_string(name) })
    }
}

impl ExactSizeIterator for IntoNamesIter {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}
impl FusedIterator for IntoNamesIter where std::vec::IntoIter<Unowned<str>>: FusedIterator {}

impl DoubleEndedIterator for IntoNamesIter {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let name = self.0.next_back()?;
        // SAFETY follows from type invariant
        Some(unsafe { unowned_to_string(name) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let name = self.0.nth_back(n)?;
        // SAFETY follows from type invariant
        Some(unsafe { unowned_to_string(name) })
    }
}

// TODO: replace String by Box<str>?
