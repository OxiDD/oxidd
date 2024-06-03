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
    use oxidd_manager_index::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_index::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_bdd::simple::BDDOp;
    use oxidd_rules_bdd::simple::BDDRules;
    use oxidd_rules_bdd::simple::BDDTerminal;

    use crate::util::type_cons::DD;

    crate::util::dd_index_based!(BDD {
        node: NodeWithLevelCons<2>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<BDDTerminal>,
        rules: BDDRulesCons for BDDRules,
        manager_data: BDDManagerDataCons for BDDManagerData,
        terminals: 2,
    });

    crate::util::manager_data!(BDDManagerData for BDD, operator: BDDOp, cache_max_arity: 3);

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
    )]
    #[use_manager_ref(BDDManagerRef, BDDManagerRef(inner))]
    pub struct BDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: BDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for BDDFunction {}
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_bdd::simple::BDDOp;
    use oxidd_rules_bdd::simple::BDDRules;
    use oxidd_rules_bdd::simple::BDDTerminal;

    use crate::util::type_cons::DD;

    crate::util::dd_pointer_based!(BDD {
        node: NodeWithLevelCons<2>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<BDDTerminal>,
        rules: BDDRulesCons for BDDRules,
        manager_data: BDDManagerDataCons for BDDManagerData,
        tag_bits: 2,
    });

    crate::util::manager_data!(BDDManagerData for BDD, operator: BDDOp, cache_max_arity: 3);

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
    )]
    #[use_manager_ref(BDDManagerRef, BDDManagerRef(inner))]
    pub struct BDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: BDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for BDDFunction {}
}
