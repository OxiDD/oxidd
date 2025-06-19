//! Index-Based Manager Implementation

//#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::type_complexity)]

pub mod manager;
pub mod node;
pub mod terminal_manager;
pub mod workers;

mod util;

#[cfg(target_pointer_width = "16")]
compile_error!("oxidd-manager-index assumes that for all `x: u32`, `x as usize` does not truncate");
