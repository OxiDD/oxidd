use oxidd_derive::Function;

#[derive(Function)]
enum Foo {}

#[derive(Function)]
union Bar {
    x: u32,
    y: bool,
}

fn main() {}
