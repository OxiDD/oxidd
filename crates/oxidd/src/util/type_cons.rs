use std::marker::PhantomData;

pub trait DD {
    type Function: oxidd_core::function::Function;
    type ManagerRef: oxidd_core::ManagerRef;
    type Manager<'id>: oxidd_core::Manager;
    type Edge<'id>: oxidd_core::Edge;
}

#[cfg(feature = "manager-index")]
pub struct IndexDD<
    NC: oxidd_manager_index::manager::InnerNodeCons<ET>,
    ET: oxidd_core::Tag,
    TMC: oxidd_manager_index::manager::TerminalManagerCons<NC, ET, TERMINALS>,
    RC: oxidd_manager_index::manager::DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
    MDC: oxidd_manager_index::manager::ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
    const TERMINALS: usize,
>(PhantomData<(NC, ET, TMC, RC, MDC)>);
#[cfg(feature = "manager-index")]
impl<
        NC: oxidd_manager_index::manager::InnerNodeCons<ET>,
        ET: oxidd_core::Tag,
        TMC: oxidd_manager_index::manager::TerminalManagerCons<NC, ET, TERMINALS>,
        RC: oxidd_manager_index::manager::DiagramRulesCons<NC, ET, TMC, MDC, TERMINALS>,
        MDC: oxidd_manager_index::manager::ManagerDataCons<NC, ET, TMC, RC, TERMINALS>,
        const TERMINALS: usize,
    > DD for IndexDD<NC, ET, TMC, RC, MDC, TERMINALS>
{
    type Function = oxidd_manager_index::manager::Function<NC, ET, TMC, RC, MDC, TERMINALS>;
    type ManagerRef = oxidd_manager_index::manager::ManagerRef<NC, ET, TMC, RC, MDC, TERMINALS>;
    type Manager<'id> = oxidd_manager_index::manager::Manager<
        'id,
        NC::T<'id>,
        ET,
        TMC::T<'id>,
        RC::T<'id>,
        MDC::T<'id>,
        TERMINALS,
    >;
    type Edge<'id> = oxidd_manager_index::manager::Edge<'id, NC::T<'id>, ET>;
}

#[cfg(feature = "manager-pointer")]
pub struct PointerDD<
    NC: oxidd_manager_pointer::manager::InnerNodeCons<ET, TAG_BITS>,
    ET: oxidd_core::Tag,
    TMC: oxidd_manager_pointer::manager::TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
    RC: oxidd_manager_pointer::manager::DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
    MDC: oxidd_manager_pointer::manager::ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
    const PAGE_SIZE: usize,
    const TAG_BITS: u32,
>(PhantomData<(NC, ET, TMC, RC, MDC)>);
#[cfg(feature = "manager-pointer")]
impl<
        NC: oxidd_manager_pointer::manager::InnerNodeCons<ET, TAG_BITS>,
        ET: oxidd_core::Tag,
        TMC: oxidd_manager_pointer::manager::TerminalManagerCons<NC, ET, RC, MDC, PAGE_SIZE, TAG_BITS>,
        RC: oxidd_manager_pointer::manager::DiagramRulesCons<NC, ET, TMC, MDC, PAGE_SIZE, TAG_BITS>,
        MDC: oxidd_manager_pointer::manager::ManagerDataCons<NC, ET, TMC, RC, PAGE_SIZE, TAG_BITS>,
        const PAGE_SIZE: usize,
        const TAG_BITS: u32,
    > DD for PointerDD<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>
{
    type Function =
        oxidd_manager_pointer::manager::Function<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>;
    type ManagerRef =
        oxidd_manager_pointer::manager::ManagerRef<NC, ET, TMC, RC, MDC, PAGE_SIZE, TAG_BITS>;
    type Manager<'id> = oxidd_manager_pointer::manager::Manager<
        'id,
        NC::T<'id>,
        ET,
        TMC::T<'id>,
        RC::T<'id>,
        MDC::T<'id>,
        PAGE_SIZE,
        TAG_BITS,
    >;
    type Edge<'id> = oxidd_manager_pointer::manager::Edge<'id, NC::T<'id>, ET, TAG_BITS>;
}
