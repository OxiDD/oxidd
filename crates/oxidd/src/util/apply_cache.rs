use std::hash::Hash;

use oxidd_core::util::Borrowed;
use oxidd_core::Manager;

cfg_if::cfg_if! {
    if #[cfg(feature = "apply-cache-direct-mapped")] {
        pub(crate) type ApplyCache<M, O, const ARITY: usize> =
            oxidd_cache::direct::DMApplyCache<M, O, rustc_hash::FxHasher, ARITY>;
    } else {
        pub(crate) type ApplyCache<M, O, const ARITY: usize> = NoApplyCache<M, O, ARITY>;
    }
}

/// Create a new apply cache
///
/// SAFETY: The apply cache must only be used inside a manager that guarantees
/// all node deletions to be wrapped inside a
/// [`oxidd_core::ApplyCache::pre_gc()`] / [`oxidd_core::ApplyCache::post_gc()`]
/// pair.
pub(crate) unsafe fn new_apply_cache<M: Manager, O: Copy + Ord + Hash, const ARITY: usize>(
    capacity: usize,
) -> ApplyCache<M, O, ARITY> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "apply-cache-direct-mapped")] {
            // SAFETY: see above
            unsafe { ApplyCache::with_capacity(capacity) }
        } else {
            let _ = capacity;
            NoApplyCache(std::marker::PhantomData)
        }
    }
}

pub struct NoApplyCache<M, O, const ARITY: usize>(pub std::marker::PhantomData<(M, O)>);

impl<M: Manager, O: Copy, const ARITY: usize> oxidd_core::util::DropWith<M::Edge>
    for NoApplyCache<M, O, ARITY>
{
    #[inline(always)]
    fn drop_with(self, _drop_edge: impl Fn(M::Edge)) {
        // Nothing to do
    }
}

impl<M: Manager, O: Copy, const ARITY: usize> oxidd_core::ApplyCache<M, O>
    for NoApplyCache<M, O, ARITY>
{
    #[inline(always)]
    fn get(&self, _manager: &M, _operator: O, _operands: &[Borrowed<M::Edge>]) -> Option<M::Edge> {
        None
    }

    #[inline(always)]
    fn add(
        &self,
        _manager: &M,
        _operator: O,
        _operands: &[Borrowed<M::Edge>],
        _value: Borrowed<M::Edge>,
    ) {
        // Just forget about it
    }

    #[inline(always)]
    fn clear(&self, _manager: &M) {
        // Nothing to do
    }
}
