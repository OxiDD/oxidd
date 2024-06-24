use std::marker::PhantomData;
use std::sync::atomic;
use std::sync::atomic::AtomicU64;

/// Generate a new, globally unique substitution ID
pub fn new_substitution_id() -> u32 {
    static ID: AtomicU64 = AtomicU64::new(0);
    let id = ID.fetch_add(1, atomic::Ordering::Relaxed);
    if id > u32::MAX as u64 {
        panic!(
            "Too many `Substitution` structs\n\
            \n\
            To allow caching across multiple results, we assign each \
            substitution a unique 32 bit integer. Now, so many substitutions \
            have been created that we cannot guarantee uniqueness anymore."
        );
    }

    id as u32
}

/// Substitution mapping variables to replacement functions
///
/// The intent behind substitution structs is to optimize the case where the
/// same substitution is applied multiple times. We would like to re-use apply
/// cache entries across these operations, and therefore, we need a compact
/// identifier for the substitution (provided by [`Self::id()`] here).
///
/// To create a substitution, you'll probably want to use [`Subst::new()`].
pub trait Substitution {
    /// Variable type
    type Var;
    /// Replacement type
    type Replacement;

    /// Get the ID of this substitution
    ///
    /// This unique identifier may safely be used as part of a cache key, i.e.,
    /// two different substitutions to be used with one manager must not have
    /// the same ID. (That two equal substitutions have the same ID would be
    /// ideal but is not required for correctness.)
    fn id(&self) -> u32;

    /// Iterate over pairs of variable and replacement
    fn pairs(&self) -> impl ExactSizeIterator<Item = (Self::Var, Self::Replacement)>;

    /// Map the substitution, e.g., to use different variable and replacement
    /// types
    ///
    /// `f` should be injective with respect to variables (the first component),
    /// i.e., two different variables should not be mapped to one. This is
    /// required to preserve that the substitution is a mapping from variables
    /// to replacement functions.
    #[inline]
    fn map<V, R, F>(&self, f: F) -> MapSubst<Self, F>
    where
        F: Fn((Self::Var, Self::Replacement)) -> (V, R),
    {
        MapSubst { inner: self, f }
    }
}

impl<T: Substitution> Substitution for &T {
    type Var = T::Var;
    type Replacement = T::Replacement;

    #[inline]
    fn id(&self) -> u32 {
        (*self).id()
    }
    #[inline]
    fn pairs(&self) -> impl ExactSizeIterator<Item = (Self::Var, Self::Replacement)> {
        (*self).pairs()
    }
}

/// Substitution mapping variables to replacement functions, created from slices
/// of functions
///
/// `S` is the storage type, and can, e.g., be `Vec<F>` or `&[F]`.
#[derive(Debug)]
pub struct Subst<F, S = Vec<F>> {
    id: u32,
    vars: S,
    replacements: S,
    phantom: PhantomData<fn(&F)>,
}

impl<F, S: Copy> Copy for Subst<F, S> {}
impl<F, S: Clone> Clone for Subst<F, S> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            vars: self.vars.clone(),
            replacements: self.replacements.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'a, F, S: AsRef<[F]>> Substitution for &'a Subst<F, S> {
    type Var = &'a F;
    type Replacement = &'a F;

    #[inline]
    fn id(&self) -> u32 {
        self.id
    }

    #[inline]
    fn pairs(&self) -> impl ExactSizeIterator<Item = (Self::Var, Self::Replacement)> {
        self.vars.as_ref().iter().zip(self.replacements.as_ref())
    }
}

impl<F, S: AsRef<[F]>> Subst<F, S> {
    /// Create a new substitution to replace the i-th variable of `vars` by the
    /// i-th function in replacement
    ///
    /// All variables of `vars` should be distinct. Furthermore, variables must
    /// be handles for the respective decision diagram levels, e.g., the
    /// respective Boolean function for B(C)DDs, and a singleton set for ZBDDs.
    ///
    /// Panics if `vars` and `replacements` have different length
    #[track_caller]
    pub fn new(vars: S, replacements: S) -> Self {
        assert_eq!(
            vars.as_ref().len(),
            replacements.as_ref().len(),
            "`vars` and `replacements` must have the same length"
        );

        Self {
            id: new_substitution_id(),
            vars,
            replacements,
            phantom: PhantomData,
        }
    }
}

/// Substitution mapping variables to replacement functions, created via
/// [`Substitution::map()`]
#[derive(Clone, Copy, Debug)]
pub struct MapSubst<'a, S: ?Sized, F> {
    inner: &'a S,
    f: F,
}

impl<V, R, S: Substitution + ?Sized, F: Fn((S::Var, S::Replacement)) -> (V, R)> Substitution
    for MapSubst<'_, S, F>
{
    type Var = V;
    type Replacement = R;

    #[inline]
    fn id(&self) -> u32 {
        self.inner.id()
    }

    #[inline]
    fn pairs(&self) -> impl ExactSizeIterator<Item = (Self::Var, Self::Replacement)> {
        self.inner.pairs().map(&self.f)
    }
}
