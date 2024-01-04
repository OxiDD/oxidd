//! Hash table with open addressing and linear probing

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod raw;

#[cfg(not(feature = "allocator-api2"))]
mod __alloc {
    pub trait Allocator {}

    #[derive(Clone, Copy, Default, Debug)]
    pub struct Global;
    impl Allocator for Global {}
}
