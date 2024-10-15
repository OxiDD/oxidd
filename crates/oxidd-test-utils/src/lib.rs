//! Mock implementations of core traits and other testing utilities

#![warn(missing_docs)]

use oxidd_core::{BroadcastContext, WorkerPool};

pub mod edge;

/// Worker thread pool (implemented via rayon)
pub struct Workers;

impl WorkerPool for Workers {
    fn current_num_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    fn split_depth(&self) -> u32 {
        42
    }

    fn set_split_depth(&self, _depth: Option<u32>) {}

    fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        op()
    }

    fn join<RA: Send, RB: Send>(
        &self,
        op_a: impl FnOnce() -> RA + Send,
        op_b: impl FnOnce() -> RB + Send,
    ) -> (RA, RB) {
        rayon::join(op_a, op_b)
    }

    fn broadcast<R: Send>(&self, op: impl Fn(oxidd_core::BroadcastContext) -> R + Sync) -> Vec<R> {
        rayon::broadcast(|ctx| {
            op(BroadcastContext {
                index: ctx.index() as u32,
                num_threads: ctx.num_threads() as u32,
            })
        })
    }
}
