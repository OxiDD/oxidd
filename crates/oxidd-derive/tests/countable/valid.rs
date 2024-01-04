use oxidd_derive::Countable;

#[derive(Clone, Copy, Countable)]
#[repr(u8)]
pub enum EdgeTag {
    None,
    Complemented,
}

#[derive(Clone, Copy, Countable)]
pub struct NoTag1;

#[derive(Clone, Copy, Countable)]
pub struct NoTag2();

#[derive(Clone, Copy, Countable)]
pub struct NoTag3 {}

fn main() {}
