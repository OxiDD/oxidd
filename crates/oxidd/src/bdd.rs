//! Binary decision diagrams (BDDs)

cfg_if::cfg_if! {
    if #[cfg(feature = "manager-pointer")] {
        pub use pointer::{BDDFunction, BDDManagerRef};
    } else if #[cfg(feature = "manager-index")] {
        pub use index::{BDDFunction, BDDManagerRef};
    } else {
        pub type BDDFunction = ();
        pub type BDDManagerRef = ();
    }
}

/// Create a new manager for a simple binary decision diagram
#[allow(unused_variables)]
pub fn new_manager(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> BDDManagerRef {
    cfg_if::cfg_if! {
        if #[cfg(feature = "manager-pointer")] {
            pointer::BDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
        } else if #[cfg(feature = "manager-index")] {
            index::BDDManagerRef::new_manager(inner_node_capacity, 2, apply_cache_capacity, threads)
        } else {
            unreachable!()
        }
    }
}

/// Print statistics to stderr
pub fn print_stats() {
    #[cfg(not(feature = "statistics"))]
    eprintln!("[statistics feature disabled]");

    #[cfg(feature = "statistics")]
    oxidd_rules_bdd::simple::print_stats();
}

#[cfg(all(feature = "manager-index", not(feature = "manager-pointer")))]
mod index {
    use oxidd_rules_bdd::simple::{BDDOp, BDDRules, BDDTerminal};

    use crate::util::type_cons::{DD, IndexDD};

    pub struct BDDInfo;
    impl oxidd_manager_index::DiagramInfo for BDDInfo {
        type EdgeTag = ();
        type InnerNode<'id> = oxidd_manager_index::node::fixed_arity::NodeWithLevel<'id, Self, 2>;
        const TERMINALS: u32 = 2;
        type TerminalNode = BDDTerminal;
        type TerminalManager<'id> =
            oxidd_manager_index::terminal_manager::StaticTerminalManager<'id, Self>;
        type Rules = BDDRules;
        type ManagerData<'id> = BDDManagerData<'id>;
    }
    type BDD = IndexDD<BDDInfo>;

    crate::util::manager_data!(BDDManagerData for BDD, operator: BDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_index_based!(pub struct BDDManagerRef(<BDD as DD>::ManagerRef) with BDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_bdd::simple::BDDFunction<<BDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_bdd::simple::BDDFunctionMT<<BDD as DD>::Function>;

    /// Boolean function represented as BDD
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
    #[use_manager_ref(BDDManagerRef, BDDManagerRef(inner))]
    pub struct BDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: BDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for BDDFunction {}
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::NodeWithLevel;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManager;
    use oxidd_rules_bdd::simple::{BDDOp, BDDRules, BDDTerminal};

    use crate::util::type_cons::{DD, PointerDD};

    pub struct BDDInfo;
    impl oxidd_manager_pointer::DiagramInfo for BDDInfo {
        type PageSizeInfo = crate::PageSizeInfo;
        type EdgeTag = ();
        const TAG_BITS: u32 = 0;
        type InnerNode<'id> = NodeWithLevel<'id, Self, 2>;
        type TerminalNode = BDDTerminal;
        type TerminalManager<'id> = StaticTerminalManager<'id, Self>;
        type Rules = BDDRules;
        type ManagerData<'id> = BDDManagerData<'id>;
    }
    type BDD = PointerDD<BDDInfo>;

    crate::util::manager_data!(BDDManagerData for BDD, operator: BDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_pointer_based!(pub struct BDDManagerRef(<BDD as DD>::ManagerRef) with BDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_bdd::simple::BDDFunction<<BDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_bdd::simple::BDDFunctionMT<<BDD as DD>::Function>;

    /// Boolean function represented as BDD
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
    #[use_manager_ref(BDDManagerRef, BDDManagerRef(inner))]
    pub struct BDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: BDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for BDDFunction {}
}
