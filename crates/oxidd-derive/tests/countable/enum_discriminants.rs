use oxidd_derive::Countable;

#[derive(Countable)]
#[repr(u8)]
enum EdgeTag {
    None = 0,
    Complemented = 1,
}

fn main() {}
