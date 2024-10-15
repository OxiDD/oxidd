use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::Manager;

type UnaryOp<M, R> = fn(&M, R, Borrowed<<M as Manager>::Edge>) -> AllocResult<<M as Manager>::Edge>;

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

type SubstOp<M, R> = fn(
    &M,
    R,
    Borrowed<<M as Manager>::Edge>,
    &[<M as Manager>::Edge],
    u32,
) -> AllocResult<<M as Manager>::Edge>;

pub trait Recursor<M: Manager>: Copy {
    fn unary<'a>(
        self,
        op: UnaryOp<M, Self>,
        manager: &'a M,
        a: Borrowed<M::Edge>,
        b: Borrowed<M::Edge>,
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)>;

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

    fn subst<'a>(
        self,
        op: SubstOp<M, Self>,
        manager: &'a M,
        a: (Borrowed<M::Edge>, &[M::Edge], u32),
        b: (Borrowed<M::Edge>, &[M::Edge], u32),
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
    fn unary<'a>(
        self,
        op: UnaryOp<M, Self>,
        manager: &'a M,
        a: Borrowed<M::Edge>,
        b: Borrowed<M::Edge>,
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
        let ra = EdgeDropGuard::new(manager, op(manager, self, a)?);
        let rb = EdgeDropGuard::new(manager, op(manager, self, b)?);
        Ok((ra, rb))
    }

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
    fn subst<'a>(
        self,
        op: SubstOp<M, Self>,
        manager: &'a M,
        a: (Borrowed<M::Edge>, &[M::Edge], u32),
        b: (Borrowed<M::Edge>, &[M::Edge], u32),
    ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
        let ra = EdgeDropGuard::new(manager, op(manager, self, a.0, a.1, a.2)?);
        let rb = EdgeDropGuard::new(manager, op(manager, self, b.0, b.1, a.2)?);
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
        fn unary<'a>(
            mut self,
            op: UnaryOp<M, Self>,
            manager: &'a M,
            a: Borrowed<M::Edge>,
            b: Borrowed<M::Edge>,
        ) -> AllocResult<(EdgeDropGuard<'a, M>, EdgeDropGuard<'a, M>)> {
            self.remaining_depth -= 1;
            let (ra, rb) = manager.workers().join(
                move || {
                    let edge = op(manager, self, a)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
                move || {
                    let edge = op(manager, self, b)?;
                    Ok(EdgeDropGuard::new(manager, edge))
                },
            );
            Ok((ra?, rb?))
        }

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

        fn subst<'a>(
            mut self,
            op: SubstOp<M, Self>,
            manager: &'a M,
            a: (Borrowed<M::Edge>, &[M::Edge], u32),
            b: (Borrowed<M::Edge>, &[M::Edge], u32),
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

        #[inline(always)]
        fn should_switch_to_sequential(self) -> bool {
            self.remaining_depth == 0
        }
    }
}
