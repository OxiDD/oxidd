use oxidd_derive::Countable;

#[derive(Countable)]
union Tag {
    f1: u8,
    f2: u32,
}

fn main() {}
