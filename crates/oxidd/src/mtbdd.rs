//! Multi-terminal binary decision diagrams (MTBDDs)

pub use oxidd_rules_mtbdd::terminal;

cfg_if::cfg_if! {
    // if #[cfg(feature = "manager-pointer")] {
    //     pub use pointer::{MTBDDFunction, MTBDDManagerRef};
    // } else
    if #[cfg(feature = "manager-index")] {
        pub use index::{MTBDDFunction, MTBDDManagerRef};
    } else {
        pub type MTBDDFunction<T> = std::marker::PhantomData<T>;
        pub type MTBDDManagerRef<T> = std::marker::PhantomData<T>;
    }
}

/// Create a new manager for a simple binary decision diagram
#[allow(unused_variables)]
pub fn new_manager<T: Eq + std::hash::Hash + Send + Sync>(
    inner_node_capacity: usize,
    terminal_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> MTBDDManagerRef<T> {
    cfg_if::cfg_if! {
        // if #[cfg(feature = "manager-pointer")] {
        //     // TODO
        // } else
        if #[cfg(feature = "manager-index")] {
            index::MTBDDManagerRef::new_manager(inner_node_capacity, terminal_node_capacity, apply_cache_capacity, threads)
        } else if #[cfg(feature = "manager-pointer")] {
            todo!("There is no DynamicTerminalManager for the pointer-based manager yet")
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

//#[cfg(all(feature = "manager-index", not(feature = "manager-pointer")))]
#[cfg(feature = "manager-index")]
mod index {
    use std::{hash::Hash, marker::PhantomData};

    use oxidd_core::function::NumberBase;
    use oxidd_rules_mtbdd::{MTBDDOp, MTBDDRules};

    use crate::util::type_cons::{DD, IndexDD};

    pub struct MTBDDInfo<T>(PhantomData<T>);
    impl<T: Eq + Hash + Send + Sync> oxidd_manager_index::DiagramInfo for MTBDDInfo<T> {
        type EdgeTag = ();
        type InnerNode<'id> = oxidd_manager_index::node::fixed_arity::NodeWithLevel<'id, Self, 2>;
        type InnerNodeValue = ();
        const TERMINALS: u32 = 0x2000000; // up to 0.78125 % terminal nodes
        type TerminalNode = T;
        type TerminalManager<'id> =
            oxidd_manager_index::terminal_manager::DynamicTerminalManager<'id, Self>;
        type Rules = MTBDDRules;
        type ManagerData<'id> = MTBDDManagerData<'id, T>;
    }
    type MTBDD<T> = IndexDD<MTBDDInfo<T>>;

    crate::util::manager_data!(MTBDDManagerData<T> for MTBDD<T>, operator: MTBDDOp, cache_entry_capacity: 3, where T: Eq + Hash + Send + Sync);

    crate::util::manager_ref_index_based!(pub struct MTBDDManagerRef<T>(<MTBDD<T> as DD>::ManagerRef) with MTBDDManagerData<T> where T: 'static + Eq + Hash + Send + Sync);

    //#[cfg(not(feature = "multi-threading"))]
    type FunctionInner<T> = oxidd_rules_mtbdd::MTBDDFunction<<MTBDD<T> as DD>::Function>;
    //#[cfg(feature = "multi-threading")]
    //type FunctionInner<T> = oxidd_rules_mtbdd::MTBDDFunctionMT<<MTBDD<T> as
    // DD>::Function>;

    /// Pseudo-Boolean function represented as MTBDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::PseudoBooleanFunction,
        oxidd_derive::DumpDeserialize,
    )]
    #[use_manager_ref(MTBDDManagerRef<T>, MTBDDManagerRef::<T>(inner))]
    #[derive_where::derive_where(PartialOrd, Ord; T: NumberBase + Send + Sync)]
    pub struct MTBDDFunction<T: 'static + NumberBase + Send + Sync>(FunctionInner<T>);

    crate::util::derive_raw_function_index_based!(for: MTBDDFunction<T>, inner: FunctionInner<T>, where T: NumberBase + Send + Sync);

    // Default implementation suffices
    impl<T: NumberBase + Send + Sync> oxidd_dump::dot::DotStyle<()> for MTBDDFunction<T> {}
}
