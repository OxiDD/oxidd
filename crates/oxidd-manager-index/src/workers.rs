use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Worker thread pool
pub struct Workers {
    pub(crate) pool: rayon::ThreadPool,
    split_depth: AtomicU32,
}

impl Workers {
    pub(crate) fn new(threads: u32) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .thread_name(|i| format!("oxidd mi {i}")) // "mi" for "manager index"
            .build()
            .expect("could not build thread pool");
        let split_depth = AtomicU32::new(Workers::auto_split_depth(&pool));
        Self { pool, split_depth }
    }

    fn auto_split_depth(pool: &rayon::ThreadPool) -> u32 {
        let threads = pool.current_num_threads();
        if threads > 1 {
            (4096 * threads).ilog2()
        } else {
            0
        }
    }
}

impl oxidd_core::WorkerPool for Workers {
    #[inline]
    fn current_num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    #[inline(always)]
    fn split_depth(&self) -> u32 {
        self.split_depth.load(Relaxed)
    }

    fn set_split_depth(&self, depth: Option<u32>) {
        let depth = match depth {
            Some(d) => d,
            None => Self::auto_split_depth(&self.pool),
        };
        self.split_depth.store(depth, Relaxed);
    }

    #[inline]
    fn install<RA: Send>(&self, op: impl FnOnce() -> RA + Send) -> RA {
        self.pool.install(op)
    }

    #[inline]
    fn join<RA: Send, RB: Send>(
        &self,
        op_a: impl FnOnce() -> RA + Send,
        op_b: impl FnOnce() -> RB + Send,
    ) -> (RA, RB) {
        self.pool.join(op_a, op_b)
    }

    #[inline]
    fn broadcast<RA: Send>(
        &self,
        op: impl Fn(oxidd_core::BroadcastContext) -> RA + Sync,
    ) -> Vec<RA> {
        self.pool.broadcast(|ctx| {
            op(oxidd_core::BroadcastContext {
                index: ctx.index() as u32,
                num_threads: ctx.num_threads() as u32,
            })
        })
    }
}
