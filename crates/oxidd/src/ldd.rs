//! List decision diagrams (LDDs)

cfg_if::cfg_if! {
    if #[cfg(feature = "manager-pointer")] {
        pub use pointer::{LDDFunction, LDDManagerRef};
    } else if #[cfg(feature = "manager-index")] {
        pub use index::{LDDFunction, LDDManagerRef};
    } else {
        pub type LDDFunction = ();
        pub type LDDManagerRef = ();
    }
}

/// Create a new manager for a simple binary decision diagram
#[allow(unused_variables)]
pub fn new_manager(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> LDDManagerRef {
    cfg_if::cfg_if! {
        if #[cfg(feature = "manager-pointer")] {
            pointer::LDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
        } else if #[cfg(feature = "manager-index")] {
            index::LDDManagerRef::new_manager(inner_node_capacity, 2, apply_cache_capacity, threads)
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

/// We only expose a hard coded u32 valued LDDManager.
pub type Value = u32;

#[cfg(all(feature = "manager-index", not(feature = "manager-pointer")))]
mod index {
    use oxidd_manager_index::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_index::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_ldd::LDDOp;
    use oxidd_rules_ldd::LDDRules;
    use oxidd_rules_ldd::LDDTerminal;

    use crate::ldd::Value;
    use crate::util::type_cons::DD;

    crate::util::dd_index_based!(LDD {
        node: NodeWithLevelCons<Value, 2>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<LDDTerminal>,
        rules: LDDRulesCons for LDDRules,
        manager_data: LDDManagerDataCons for LDDManagerData,
        terminals: 2,
    });

    crate::util::manager_data!(LDDManagerData for LDD, operator: LDDOp, cache_max_arity: 3);

    crate::util::manager_ref_index_based!(pub struct LDDManagerRef(<LDD as DD>::ManagerRef) with LDDManagerData);

    type FunctionInner = oxidd_rules_ldd::LDDFunction<<LDD as DD>::Function>;

    /// Boolean function represented as BDD
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, oxidd_derive::Function)]
    #[use_manager_ref(LDDManagerRef, LDDManagerRef(inner))]
    pub struct LDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: LDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for LDDFunction {}
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_ldd::LDDOp;
    use oxidd_rules_ldd::LDDRules;
    use oxidd_rules_ldd::LDDTerminal;

    use crate::ldd::Value;
    use crate::util::type_cons::DD;

    crate::util::dd_pointer_based!(LDD {
        node: NodeWithLevelCons<Value, 2>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<LDDTerminal>,
        rules: LDDRulesCons for LDDRules,
        manager_data: LDDManagerDataCons for LDDManagerData,
        tag_bits: 2,
    });

    crate::util::manager_data!(LDDManagerData for LDD, operator: LDDOp, cache_max_arity: 3);

    crate::util::manager_ref_pointer_based!(pub struct LDDManagerRef(<LDD as DD>::ManagerRef) with LDDManagerData);

    type FunctionInner = oxidd_rules_ldd::LDDFunction<<LDD as DD>::Function>;

    /// Boolean function represented as BDD
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, oxidd_derive::Function)]
    #[use_manager_ref(LDDManagerRef, LDDManagerRef(inner))]
    pub struct LDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: LDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for LDDFunction {}
}
