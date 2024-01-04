use oxidd_derive::Function;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function)]
pub struct MyFunc<F: oxidd_core::function::Function>(F);

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function)]
pub struct MyFuncNamed<F>
where
    F: oxidd_core::function::Function,
{
    field: F,
}

fn main() {}
