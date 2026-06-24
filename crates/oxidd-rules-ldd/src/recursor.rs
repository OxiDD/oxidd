//! Abstraction over sequential and parallel recursion for the apply algorithms
//!
//! This is a direct adaptation of the recursor used by the `oxidd-rules-bdd`
//! crate. The LDD apply algorithms recurse with operators taking up to four
//! edges (the relational predecessor takes `set`, `rel`, `meta`, and
//! `universe`), hence this version exposes [`Recursor::binary`],
//! [`Recursor::ternary`], and [`Recursor::quaternary`] instead of the
//! BDD-specific `unary`/`subst` operators.

use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::Manager;

type BinaryOp<M, R> = fn(
    &M,
    R,
    Borrowed<<M as Manager>::Edge>,
    Borrowed<<M as Manager>::Edge>,
) -> AllocResult<<M as Manager>::Edge>;

type TernaryOp<M, R> = fn(
    &M,
    R,
    Borrowed<<M as Manager>::Edge>,
    Borrowed<<M as Manager>::Edge>,
    Borrowed<<M as Manager>::Edge>,
) -> AllocResult<<M as Manager>::Edge>;

type QuaternaryOp<M, R> = fn(
    &M,
    R,
    Borrowed<<M as Manager>::Edge>,
    Borrowed<<M as Manager>::Edge>,
    Borrowed<<M as Manager>::Edge>,
    Borrowed<<M as Manager>::Edge>,
) -> AllocResult<<M as Manager>::Edge>;

/// Abstraction over the recursion strategy used by the apply algorithms
///
/// Each method runs the given operator on the two argument tuples and returns
/// the two results. A [`SequentialRecursor`] simply evaluates them one after
/// the other, while a [parallel recursor][mt::ParallelRecursor] may spawn the
/// two evaluations onto a worker pool.
pub trait Recursor<M: Manager>: Copy {
    fn binary<'a>(
        self,
        op: BinaryOp<M, Self>,
        manager: &'a M,
        a: (Borrowed<M::Edge>, Borrowed<M::Edge>),
        b: (Borrowed<M::Edge>, Borrowed<M::Edge>),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)>;

    #[allow(clippy::type_complexity)]
    fn ternary<'a>(
        self,
        op: TernaryOp<M, Self>,
        manager: &'a M,
        a: (Borrowed<M::Edge>, Borrowed<M::Edge>, Borrowed<M::Edge>),
        b: (Borrowed<M::Edge>, Borrowed<M::Edge>, Borrowed<M::Edge>),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)>;

    #[allow(clippy::type_complexity)]
    fn quaternary<'a>(
        self,
        op: QuaternaryOp<M, Self>,
        manager: &'a M,
        a: (
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
        ),
        b: (
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
        ),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)>;

    /// Returns true if the algorithm should switch to a sequential recursor
    ///
    /// With the current [`join()`][oxidd_core::WorkerPool::join]
    /// implementations, we observe a significant performance overhead
    /// compared to sequentially calling the functions. Therefore, it may
    /// make sense to switch to the sequential version after, e.g., a
    /// certain recursion depth.
    fn should_switch_to_sequential(self) -> bool;
}

#[derive(Clone, Copy)]
pub struct SequentialRecursor;

impl<M: Manager> Recursor<M> for SequentialRecursor {
    #[inline(always)]
    fn binary<'a>(
        self,
        op: BinaryOp<M, Self>,
        manager: &'a M,
        a: (Borrowed<M::Edge>, Borrowed<M::Edge>),
        b: (Borrowed<M::Edge>, Borrowed<M::Edge>),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
        let ra = EdgeDropGuard::new(manager, op(manager, self, a.0, a.1)?);
        let rb = EdgeDropGuard::new(manager, op(manager, self, b.0, b.1)?);
        Ok((ra, rb))
    }

    #[inline(always)]
    fn ternary<'a>(
        self,
        op: TernaryOp<M, Self>,
        manager: &'a M,
        a: (Borrowed<M::Edge>, Borrowed<M::Edge>, Borrowed<M::Edge>),
        b: (Borrowed<M::Edge>, Borrowed<M::Edge>, Borrowed<M::Edge>),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
        let ra = EdgeDropGuard::new(manager, op(manager, self, a.0, a.1, a.2)?);
        let rb = EdgeDropGuard::new(manager, op(manager, self, b.0, b.1, b.2)?);
        Ok((ra, rb))
    }

    #[inline(always)]
    fn quaternary<'a>(
        self,
        op: QuaternaryOp<M, Self>,
        manager: &'a M,
        a: (
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
        ),
        b: (
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
            Borrowed<M::Edge>,
        ),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
        let ra = EdgeDropGuard::new(manager, op(manager, self, a.0, a.1, a.2, a.3)?);
        let rb = EdgeDropGuard::new(manager, op(manager, self, b.0, b.1, b.2, b.3)?);
        Ok((ra, rb))
    }

    #[inline(always)]
    fn should_switch_to_sequential(self) -> bool {
        false // returning true would make the algorithms diverge
    }
}

#[cfg(feature = "multi-threading")]
pub mod mt {
    use super::*;
    use oxidd_core::WorkerPool;

    #[derive(Clone, Copy)]
    pub struct ParallelRecursor {
        remaining_depth: u32,
    }

    impl ParallelRecursor {
        pub fn new<M: oxidd_core::HasWorkers>(manager: &M) -> Self {
            Self {
                remaining_depth: manager.workers().split_depth(),
            }
        }
    }

    impl<M> Recursor<M> for ParallelRecursor
    where
        M: Manager + oxidd_core::HasWorkers,
        M::Edge: Send + Sync,
    {
        fn binary<'a>(
            mut self,
            op: BinaryOp<M, Self>,
            manager: &'a M,
            a: (Borrowed<M::Edge>, Borrowed<M::Edge>),
            b: (Borrowed<M::Edge>, Borrowed<M::Edge>),
        ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
            self.remaining_depth -= 1;
            let (ra, rb) = manager.workers().join(
                move || {
                    let edge = op(manager, self, a.0, a.1)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
                move || {
                    let edge = op(manager, self, b.0, b.1)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
            );
            Ok((ra?, rb?))
        }

        fn ternary<'a>(
            mut self,
            op: TernaryOp<M, Self>,
            manager: &'a M,
            a: (Borrowed<M::Edge>, Borrowed<M::Edge>, Borrowed<M::Edge>),
            b: (Borrowed<M::Edge>, Borrowed<M::Edge>, Borrowed<M::Edge>),
        ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
            self.remaining_depth -= 1;
            let (ra, rb) = manager.workers().join(
                move || {
                    let edge = op(manager, self, a.0, a.1, a.2)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
                move || {
                    let edge = op(manager, self, b.0, b.1, b.2)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
            );
            Ok((ra?, rb?))
        }

        fn quaternary<'a>(
            mut self,
            op: QuaternaryOp<M, Self>,
            manager: &'a M,
            a: (
                Borrowed<M::Edge>,
                Borrowed<M::Edge>,
                Borrowed<M::Edge>,
                Borrowed<M::Edge>,
            ),
            b: (
                Borrowed<M::Edge>,
                Borrowed<M::Edge>,
                Borrowed<M::Edge>,
                Borrowed<M::Edge>,
            ),
        ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
            self.remaining_depth -= 1;
            let (ra, rb) = manager.workers().join(
                move || {
                    let edge = op(manager, self, a.0, a.1, a.2, a.3)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
                move || {
                    let edge = op(manager, self, b.0, b.1, b.2, b.3)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
            );
            Ok((ra?, rb?))
        }

        #[inline(always)]
        fn should_switch_to_sequential(self) -> bool {
            self.remaining_depth == 0
        }
    }
}
