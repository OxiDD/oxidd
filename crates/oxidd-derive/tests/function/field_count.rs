use oxidd_derive::Function;

#[derive(Function)]
struct Fieldless;

#[derive(Function)]
struct Unnamed0();

#[derive(Function)]
struct Unnamed2<F>(F, bool);

#[derive(Function)]
struct Named0 {}

#[derive(Function)]
struct Named2 {
    x: bool,
    y: u32,
}

fn main() {}
