cfg_if::cfg_if! {
    if #[cfg(feature = "manager-pointer")] {
        pub use pointer::{CBDDFunction, CBDDManagerRef};
    } else if #[cfg(feature = "manager-index")] {
        pub use index::{CBDDFunction, CBDDManagerRef};
    } else {
        pub type CBDDFunction = ();
        pub type CBDDManagerRef = ();
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
) -> CBDDManagerRef {
    cfg_if::cfg_if! {
        if #[cfg(feature = "manager-pointer")] {
            pointer::CBDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
        } else if #[cfg(feature = "manager-index")] {
            index::CBDDManagerRef::new_manager(inner_node_capacity, 2, apply_cache_capacity, threads)
        } else {
            unreachable!()
        }
    }
}

#[cfg(feature = "manager-index")]
mod index {
    use oxidd_manager_index::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_index::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_bdd::complement_edge::CBDDOp;
    use oxidd_rules_bdd::complement_edge::CBDDRules;
    use oxidd_rules_bdd::complement_edge::CBDDTerminal;
    use oxidd_rules_bdd::complement_edge::EdgeTag;

    use crate::util::type_cons::DD;

    crate::util::dd_index_based!(CBDD {
        node: NodeWithLevelCons<2>,
        edge_tag: EdgeTag,
        terminal_manager: StaticTerminalManagerCons<CBDDTerminal>,
        rules: CBDDRulesCons for CBDDRules,
        manager_data: CBDDManagerDataCons for CBDDManagerData,
        terminals: 1,
    });

    crate::util::manager_data!(CBDDManagerData for CBDD, operator: CBDDOp, cache_max_arity: 2);

    crate::util::manager_ref_index_based!(pub struct CBDDManagerRef(<CBDD as DD>::ManagerRef) with CBDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_bdd::complement_edge::CBDDFunction<<CBDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_bdd::complement_edge::CBDDFunctionMT<<CBDD as DD>::Function>;

    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::BooleanFunction,
        oxidd_derive::BooleanFunctionQuant,
    )]
    #[use_manager_ref(CBDDManagerRef)]
    pub struct CBDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: CBDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<EdgeTag> for CBDDFunction {}
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_bdd::complement_edge::CBDDOp;
    use oxidd_rules_bdd::complement_edge::CBDDRules;
    use oxidd_rules_bdd::complement_edge::CBDDTerminal;
    use oxidd_rules_bdd::complement_edge::EdgeTag;

    use crate::util::type_cons::DD;

    crate::util::dd_pointer_based!(CBDD {
        node: NodeWithLevelCons<2>,
        edge_tag: EdgeTag,
        terminal_manager: StaticTerminalManagerCons<CBDDTerminal>,
        rules: CBDDRulesCons for CBDDRules,
        manager_data: CBDDManagerDataCons for CBDDManagerData,
        tag_bits: 2,
    });

    crate::util::manager_data!(CBDDManagerData for CBDD, operator: CBDDOp, cache_max_arity: 2);

    crate::util::manager_ref_pointer_based!(pub struct CBDDManagerRef(<CBDD as DD>::ManagerRef) with CBDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_bdd::complement_edge::CBDDFunction<<CBDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_bdd::complement_edge::CBDDFunctionMT<<CBDD as DD>::Function>;

    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::BooleanFunction,
        oxidd_derive::BooleanFunctionQuant,
    )]
    #[use_manager_ref(CBDDManagerRef)]
    pub struct CBDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: CBDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<EdgeTag> for CBDDFunction {}
}
