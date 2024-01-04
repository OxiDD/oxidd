use oxidd_derive::Countable;

#[derive(Countable)]
enum Tag {
    X(u8),
    Y(u32),
}

fn main() {}
