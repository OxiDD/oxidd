use oxidd_derive::Countable;

#[derive(Countable)]
struct Tag1(u8);

#[derive(Countable)]
struct Tag2 {
    field: u8,
}

fn main() {}
