//! Binary decision diagrams with complemented edges (BCDDs)

cfg_if::cfg_if! {
    if #[cfg(feature = "manager-pointer")] {
        pub use pointer::{BCDDFunction, BCDDManagerRef};
    } else if #[cfg(feature = "manager-index")] {
        pub use index::{BCDDFunction, BCDDManagerRef};
    } else {
        pub type BCDDFunction = ();
        pub type BCDDManagerRef = ();
    }
}

/// Print statistics to stderr
pub fn print_stats() {
    #[cfg(not(feature = "statistics"))]
    eprintln!("[statistics feature disabled]");

    #[cfg(feature = "statistics")]
    oxidd_rules_bdd::complement_edge::print_stats();
}

/// Create a new manager for a binary decision diagram with complement edges
#[allow(unused_variables)]
pub fn new_manager(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> BCDDManagerRef {
    cfg_if::cfg_if! {
        if #[cfg(feature = "manager-pointer")] {
            pointer::BCDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
        } else if #[cfg(feature = "manager-index")] {
            index::BCDDManagerRef::new_manager(inner_node_capacity, 1, apply_cache_capacity, threads)
        } else {
            unreachable!()
        }
    }
}

#[cfg(all(feature = "manager-index", not(feature = "manager-pointer")))]
mod index {
    use oxidd_core::AtomicLevelNo;
    use oxidd_rules_bdd::complement_edge::{BCDDOp, BCDDRules, BCDDTerminal, EdgeTag};

    use crate::util::type_cons::{DD, IndexDD};

    pub struct BCDDInfo;
    impl oxidd_manager_index::DiagramInfo for BCDDInfo {
        type EdgeTag = EdgeTag;
        type InnerNode<'id> =
            oxidd_manager_index::node::fixed_arity::Node<'id, Self, AtomicLevelNo, 2>;
        type InnerNodeValue = ();
        const TERMINALS: u32 = 1;
        type TerminalNode = BCDDTerminal;
        type TerminalManager<'id> =
            oxidd_manager_index::terminal_manager::StaticTerminalManager<'id, Self>;
        type Rules = BCDDRules;
        type ManagerData<'id> = BCDDManagerData<'id>;
    }
    type BCDD = IndexDD<BCDDInfo>;

    crate::util::manager_data!(BCDDManagerData for BCDD, operator: BCDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_index_based!(pub struct BCDDManagerRef(<BCDD as DD>::ManagerRef) with BCDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_bdd::complement_edge::BCDDFunction<<BCDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_bdd::complement_edge::BCDDFunctionMT<<BCDD as DD>::Function>;

    /// Boolean function represented as BCDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::FunctionSubst,
        oxidd_derive::BooleanFunction,
        oxidd_derive::BooleanFunctionQuant,
        oxidd_derive::DumpDeserialize,
    )]
    #[use_manager_ref(BCDDManagerRef, BCDDManagerRef(inner))]
    pub struct BCDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: BCDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<EdgeTag> for BCDDFunction {}
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::Node;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManager;
    use oxidd_rules_bdd::complement_edge::{BCDDOp, BCDDRules, BCDDTerminal, EdgeTag};

    use crate::util::type_cons::{DD, PointerDD};

    pub struct BCDDInfo;
    impl oxidd_manager_pointer::DiagramInfo for BCDDInfo {
        type PageSizeInfo = crate::PageSizeInfo;
        type EdgeTag = EdgeTag;
        const TAG_BITS: u32 = 1;
        type InnerNode<'id> = Node<'id, Self, oxidd_core::AtomicLevelNo, 2>;
        type InnerNodeValue = ();
        type TerminalNode = BCDDTerminal;
        type TerminalManager<'id> = StaticTerminalManager<'id, Self>;
        type Rules = BCDDRules;
        type ManagerData<'id> = BCDDManagerData<'id>;
    }
    type BCDD = PointerDD<BCDDInfo>;

    crate::util::manager_data!(BCDDManagerData for BCDD, operator: BCDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_pointer_based!(pub struct BCDDManagerRef(<BCDD as DD>::ManagerRef) with BCDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_bdd::complement_edge::BCDDFunction<<BCDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_bdd::complement_edge::BCDDFunctionMT<<BCDD as DD>::Function>;

    /// Boolean function represented as BCDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::FunctionSubst,
        oxidd_derive::BooleanFunction,
        oxidd_derive::BooleanFunctionQuant,
        oxidd_derive::DumpDeserialize,
    )]
    #[use_manager_ref(BCDDManagerRef, BCDDManagerRef(inner))]
    pub struct BCDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: BCDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<EdgeTag> for BCDDFunction {}
}
