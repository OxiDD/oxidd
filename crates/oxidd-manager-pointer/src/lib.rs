//! Pointer-based manager implementation

//#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::type_complexity)]
// We use const assertions for checking configurations and need to make sure
// that they are evaluated
#![allow(clippy::let_unit_value)]

pub mod manager;
pub mod node;
pub mod terminal_manager;
pub mod workers;

mod util;
