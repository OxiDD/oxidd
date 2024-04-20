//! Ternary decision diagrams (TDDs)

cfg_if::cfg_if! {
    if #[cfg(feature = "manager-pointer")] {
        pub use pointer::{TDDFunction, TDDManagerRef};
    } else if #[cfg(feature = "manager-index")] {
        pub use index::{TDDFunction, TDDManagerRef};
    } else {
        pub type TDDFunction = ();
        pub type TDDManagerRef = ();
    }
}

/// Create a new manager for a simple binary decision diagram
#[allow(unused_variables)]
pub fn new_manager(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> TDDManagerRef {
    cfg_if::cfg_if! {
        if #[cfg(feature = "manager-pointer")] {
            pointer::TDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
        } else if #[cfg(feature = "manager-index")] {
            index::TDDManagerRef::new_manager(inner_node_capacity, 3, apply_cache_capacity, threads)
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
    oxidd_rules_tdd::print_stats();
}

#[cfg(all(feature = "manager-index", not(feature = "manager-pointer")))]
mod index {
    use oxidd_manager_index::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_index::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_tdd::TDDOp;
    use oxidd_rules_tdd::TDDRules;
    use oxidd_rules_tdd::TDDTerminal;

    use crate::util::type_cons::DD;

    crate::util::dd_index_based!(TDD {
        node: NodeWithLevelCons<3>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<TDDTerminal>,
        rules: TDDRulesCons for TDDRules,
        manager_data: TDDManagerDataCons for TDDManagerData,
        terminals: 3,
    });

    crate::util::manager_data!(TDDManagerData for TDD, operator: TDDOp, cache_max_arity: 2);

    crate::util::manager_ref_index_based!(pub struct TDDManagerRef(<TDD as DD>::ManagerRef) with TDDManagerData);

    //#[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_tdd::TDDFunction<<TDD as DD>::Function>;
    //#[cfg(feature = "multi-threading")]
    //type FunctionInner = oxidd_rules_tdd::TDDFunctionMT<<TDD as DD>::Function>;

    /// Function of three-valued logic represented as TDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::TVLFunction,
    )]
    #[use_manager_ref(TDDManagerRef, TDDManagerRef(inner))]
    pub struct TDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: TDDFunction, inner: FunctionInner);

    impl oxidd_dump::dot::DotStyle<()> for TDDFunction {
        fn edge_style(
            no: usize,
            tag: (),
        ) -> (oxidd_dump::dot::EdgeStyle, bool, oxidd_dump::dot::Color) {
            FunctionInner::edge_style(no, tag)
        }
    }
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_tdd::TDDOp;
    use oxidd_rules_tdd::TDDRules;
    use oxidd_rules_tdd::TDDTerminal;

    use crate::util::type_cons::DD;

    crate::util::dd_pointer_based!(TDD {
        node: NodeWithLevelCons<3>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<TDDTerminal>,
        rules: TDDRulesCons for TDDRules,
        manager_data: TDDManagerDataCons for TDDManagerData,
        tag_bits: 2,
    });

    crate::util::manager_data!(TDDManagerData for TDD, operator: TDDOp, cache_max_arity: 2);

    crate::util::manager_ref_pointer_based!(pub struct TDDManagerRef(<TDD as DD>::ManagerRef) with TDDManagerData);

    //#[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_tdd::TDDFunction<<TDD as DD>::Function>;
    //#[cfg(feature = "multi-threading")]
    //type FunctionInner = oxidd_rules_tdd::TDDFunctionMT<<TDD as DD>::Function>;

    /// Function of three-valued logic represented as TDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::TVLFunction,
    )]
    #[use_manager_ref(TDDManagerRef, TDDManagerRef(inner))]
    pub struct TDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: TDDFunction, inner: FunctionInner);

    impl oxidd_dump::dot::DotStyle<()> for TDDFunction {
        fn edge_style(
            no: usize,
            tag: (),
        ) -> (oxidd_dump::dot::EdgeStyle, bool, oxidd_dump::dot::Color) {
            FunctionInner::edge_style(no, tag)
        }
    }
}
