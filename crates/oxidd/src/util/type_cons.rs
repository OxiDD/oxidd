// Do not spam the user with warnings in case both or no manager implementations
// are enabled, or no DD kind is enabled.
#![allow(unused)]

use std::marker::PhantomData;

pub trait DD {
    type Function: oxidd_core::function::Function;
    type ManagerRef: oxidd_core::ManagerRef;
    type Manager<'id>: oxidd_core::Manager;
    type Edge<'id>: oxidd_core::Edge;
}

#[cfg(feature = "manager-index")]
pub struct IndexDD<DI: oxidd_manager_index::DiagramInfo>(PhantomData<DI>);
#[cfg(feature = "manager-index")]
impl<DI: oxidd_manager_index::DiagramInfo> DD for IndexDD<DI> {
    type Function = oxidd_manager_index::manager::Function<DI>;
    type ManagerRef = oxidd_manager_index::manager::ManagerRef<DI>;
    type Manager<'id> = oxidd_manager_index::manager::Manager<'id, DI>;
    type Edge<'id> = oxidd_manager_index::manager::Edge<'id, DI>;
}

#[cfg(feature = "manager-pointer")]
pub struct PointerDD<DI: oxidd_manager_pointer::DiagramInfo>(PhantomData<DI>);
#[cfg(feature = "manager-pointer")]
impl<DI: oxidd_manager_pointer::DiagramInfo> DD for PointerDD<DI> {
    type Function = oxidd_manager_pointer::manager::Function<DI>;
    type ManagerRef = oxidd_manager_pointer::manager::ManagerRef<DI>;
    type Manager<'id> = oxidd_manager_pointer::manager::Manager<'id, DI>;
    type Edge<'id> = oxidd_manager_pointer::manager::Edge<'id, DI>;
}
