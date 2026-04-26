use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Worker thread pool.
///
/// On native targets, owns a private `rayon::ThreadPool`. On `wasm32`,
/// there is no pool at all: every `WorkerPool` trait method executes
/// inline on the calling thread. This is because `std::thread::spawn`
/// is unsupported on `wasm32`, so rayon cannot build a pool; and the
/// one prior workaround (`wasm-bindgen-rayon`'s `initThreadPool`)
/// required `SharedArrayBuffer` and shipped parallel task coordination
/// cost that empirically didn't translate into speedup for BDD apply
/// at browser workload sizes. Keeping wasm32 strictly serial removes
/// `SharedArrayBuffer` / `crossOriginIsolated` / nightly Rust from the
/// build requirements.
pub struct Workers {
    #[cfg(not(target_arch = "wasm32"))]
    pool: rayon::ThreadPool,
    split_depth: AtomicU32,
}

impl Workers {
    pub(crate) fn new(threads: u32) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let stack_size = std::env::var("OXIDD_STACK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1024 * 1024 * 1024); // default: 1 GiB

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads as usize)
                .thread_name(|i| format!("oxidd mi {i}")) // "mi" for "manager index"
                .stack_size(stack_size)
                .build()
                .expect("could not build thread pool");
            let num_threads = pool.current_num_threads();
            let split_depth = AtomicU32::new(if num_threads > 1 {
                (4096 * num_threads).ilog2()
            } else {
                0
            });
            return Self { pool, split_depth };
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = threads; // wasm32 is always single-threaded
            Self {
                split_depth: AtomicU32::new(0),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn spawn_broadcast(&self, op: impl Fn(rayon::BroadcastContext) + Sync + Send + 'static) {
        self.pool.spawn_broadcast(op);
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn spawn_broadcast<F>(&self, op: F)
    where
        F: Fn(WasmBroadcastContext) + Send + 'static,
    {
        // On wasm32 we're strictly single-threaded; broadcast of N is
        // just execute-once synchronously on this thread.
        op(WasmBroadcastContext);
    }
}

/// Zero-sized argument stand-in for rayon's broadcast context on wasm32.
/// We never read from it (we're single-threaded), so it carries no data.
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub struct WasmBroadcastContext;

impl oxidd_core::WorkerPool for Workers {
    #[inline]
    fn current_num_threads(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        { self.pool.current_num_threads() }
        #[cfg(target_arch = "wasm32")]
        { 1 }
    }

    #[inline(always)]
    fn split_depth(&self) -> u32 {
        self.split_depth.load(Relaxed)
    }

    fn set_split_depth(&self, depth: Option<u32>) {
        let depth = match depth {
            Some(d) => d,
            None => {
                let n = self.current_num_threads();
                if n > 1 { (4096 * n).ilog2() } else { 0 }
            }
        };
        self.split_depth.store(depth, Relaxed);
    }

    #[inline]
    fn install<RA: Send>(&self, op: impl FnOnce() -> RA + Send) -> RA {
        #[cfg(not(target_arch = "wasm32"))]
        { self.pool.install(op) }
        #[cfg(target_arch = "wasm32")]
        { op() }
    }

    #[inline]
    fn join<RA: Send, RB: Send>(
        &self,
        op_a: impl FnOnce() -> RA + Send,
        op_b: impl FnOnce() -> RB + Send,
    ) -> (RA, RB) {
        #[cfg(not(target_arch = "wasm32"))]
        { self.pool.join(op_a, op_b) }
        #[cfg(target_arch = "wasm32")]
        { (op_a(), op_b()) }
    }

    #[inline]
    fn broadcast<RA: Send>(
        &self,
        op: impl Fn(oxidd_core::BroadcastContext) -> RA + Sync,
    ) -> Vec<RA> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pool.broadcast(|ctx| {
                op(oxidd_core::BroadcastContext {
                    index: ctx.index() as u32,
                    num_threads: ctx.num_threads() as u32,
                })
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            vec![op(oxidd_core::BroadcastContext {
                index: 0,
                num_threads: 1,
            })]
        }
    }
}
