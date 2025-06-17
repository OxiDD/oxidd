//! Hash table with open addressing and linear probing

#![no_std]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod raw;

#[cfg(not(any(feature = "allocator-api2", feature = "nightly")))]
mod __alloc {
    pub trait Allocator {}

    #[derive(Clone, Copy, Default, Debug)]
    pub struct Global;
    impl Allocator for Global {}
}
