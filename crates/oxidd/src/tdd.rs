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
    use oxidd_core::AtomicLevelNo;
    use oxidd_rules_tdd::{TDDOp, TDDRules, TDDTerminal};

    use crate::util::type_cons::{DD, IndexDD};

    pub struct TDDInfo;
    impl oxidd_manager_index::DiagramInfo for TDDInfo {
        type EdgeTag = ();
        type InnerNode<'id> =
            oxidd_manager_index::node::fixed_arity::Node<'id, Self, AtomicLevelNo, 3>;
        type InnerNodeValue = ();
        const TERMINALS: u32 = 3;
        type TerminalNode = TDDTerminal;
        type TerminalManager<'id> =
            oxidd_manager_index::terminal_manager::StaticTerminalManager<'id, Self>;
        type Rules = TDDRules;
        type ManagerData<'id> = TDDManagerData<'id>;
    }
    type TDD = IndexDD<TDDInfo>;

    crate::util::manager_data!(TDDManagerData for TDD, operator: TDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_index_based!(pub struct TDDManagerRef(<TDD as DD>::ManagerRef) with TDDManagerData);

    //#[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_tdd::TDDFunction<<TDD as DD>::Function>;
    //#[cfg(feature = "multi-threading")]
    //type FunctionInner
    //    = oxidd_rules_tdd::TDDFunctionMT<<TDD as DD>::Function>;

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
        oxidd_derive::DumpDeserialize,
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
    use oxidd_manager_pointer::node::fixed_arity::Node;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManager;
    use oxidd_rules_tdd::{TDDOp, TDDRules, TDDTerminal};

    use crate::util::type_cons::{DD, PointerDD};

    pub struct TDDInfo;
    impl oxidd_manager_pointer::DiagramInfo for TDDInfo {
        type PageSizeInfo = crate::PageSizeInfo;
        type EdgeTag = ();
        const TAG_BITS: u32 = 0;
        type InnerNode<'id> = Node<'id, Self, oxidd_core::AtomicLevelNo, 3>;
        type InnerNodeValue = ();
        type TerminalNode = TDDTerminal;
        type TerminalManager<'id> = StaticTerminalManager<'id, Self>;
        type Rules = TDDRules;
        type ManagerData<'id> = TDDManagerData<'id>;
    }
    type TDD = PointerDD<TDDInfo>;

    crate::util::manager_data!(TDDManagerData for TDD, operator: TDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_pointer_based!(pub struct TDDManagerRef(<TDD as DD>::ManagerRef) with TDDManagerData);

    //#[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_tdd::TDDFunction<<TDD as DD>::Function>;
    //#[cfg(feature = "multi-threading")]
    //type FunctionInner
    //    = oxidd_rules_tdd::TDDFunctionMT<<TDD as DD>::Function>;

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
        oxidd_derive::DumpDeserialize,
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
