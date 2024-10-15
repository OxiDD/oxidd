//! Index-Based Manager Implementation

//#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::type_complexity)]
// We use const assertions for checking configurations and reporting errors in
// case of obscure targets. To achieve this, we use assertions that evaluate to
// `true` on usual targets as well as unit let bindings.
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::let_unit_value)]

pub mod manager;
pub mod node;
pub mod terminal_manager;
pub mod workers;

mod util;
