use std::marker::PhantomData;

pub trait DD {
    type Function: oxidd_core::function::Function;
    type ManagerRef: oxidd_core::ManagerRef;
    type Manager<'id>: oxidd_core::Manager;
    type Edge<'id>: oxidd_core::Edge;
}

#[cfg(feature = "manager-index")]
pub struct IndexDD<DI: oxidd_manager_index::DiagramInfo<TERMINALS>, const TERMINALS: usize>(
    PhantomData<DI>,
);
#[cfg(feature = "manager-index")]
impl<DI: oxidd_manager_index::DiagramInfo<TERMINALS>, const TERMINALS: usize> DD
    for IndexDD<DI, TERMINALS>
{
    type Function = oxidd_manager_index::manager::Function<DI, TERMINALS>;
    type ManagerRef = oxidd_manager_index::manager::ManagerRef<DI, TERMINALS>;
    type Manager<'id> = oxidd_manager_index::manager::Manager<'id, DI, TERMINALS>;
    type Edge<'id> = oxidd_manager_index::manager::Edge<'id, DI, TERMINALS>;
}

#[cfg(feature = "manager-pointer")]
pub struct PointerDD<
    DI: oxidd_manager_pointer::DiagramInfo<PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>(PhantomData<DI>);
#[cfg(feature = "manager-pointer")]
impl<
    DI: oxidd_manager_pointer::DiagramInfo<PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
> DD for PointerDD<DI, PAGE_SIZE, TAG_BITS>
{
    type Function = oxidd_manager_pointer::manager::Function<DI, PAGE_SIZE, TAG_BITS>;
    type ManagerRef = oxidd_manager_pointer::manager::ManagerRef<DI, PAGE_SIZE, TAG_BITS>;
    type Manager<'id> = oxidd_manager_pointer::manager::Manager<'id, DI, PAGE_SIZE, TAG_BITS>;
    type Edge<'id> = oxidd_manager_pointer::manager::Edge<'id, DI, PAGE_SIZE, TAG_BITS>;
}
