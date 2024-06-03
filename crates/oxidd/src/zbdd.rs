//! Zero-suppressed binary decision diagrams (ZBDDs)

#[allow(unused)] // unused in case no manager impl is selected
macro_rules! manager_data {
    ($name:ident for $dd:ident, operator: $op:ty, cache_max_arity: $arity:expr) => {
        pub struct $name<'id> {
            apply_cache: $crate::util::apply_cache::ApplyCache<
                <$dd as $crate::util::type_cons::DD>::Manager<'id>,
                $op,
                $arity,
            >,
            zbdd_cache:
                ::oxidd_rules_zbdd::ZBDDCache<<$dd as $crate::util::type_cons::DD>::Edge<'id>>,
        }

        impl<'id> $name<'id> {
            /// SAFETY: The manager data must only be used inside a manager that
            /// guarantees all node deletions to be wrapped inside a
            /// [`oxidd_core::ApplyCache::pre_gc()`] /
            /// [`oxidd_core::ApplyCache::post_gc()`]
            /// pair for the contained apply cache.
            unsafe fn new(apply_cache_capacity: usize) -> Self {
                Self {
                    // SAFETY: see above
                    apply_cache: unsafe {
                        $crate::util::apply_cache::new_apply_cache(apply_cache_capacity)
                    },
                    zbdd_cache: ::oxidd_rules_zbdd::ZBDDCache::new(),
                }
            }
        }

        impl<'id> ::oxidd_core::util::DropWith<<$dd as $crate::util::type_cons::DD>::Edge<'id>>
            for $name<'id>
        {
            fn drop_with(
                self,
                drop_edge: impl Fn(<$dd as $crate::util::type_cons::DD>::Edge<'id>),
            ) {
                self.apply_cache.drop_with(&drop_edge);
                self.zbdd_cache.drop_with(drop_edge);
            }
        }

        impl<'id>
            ::oxidd_core::util::GCContainer<<$dd as $crate::util::type_cons::DD>::Manager<'id>>
            for $name<'id>
        {
            #[inline]
            fn pre_gc(&self, manager: &<$dd as $crate::util::type_cons::DD>::Manager<'id>) {
                self.apply_cache.pre_gc(manager)
            }
            #[inline]
            unsafe fn post_gc(&self, manager: &<$dd as $crate::util::type_cons::DD>::Manager<'id>) {
                // SAFETY: inherited from outer
                unsafe { self.apply_cache.post_gc(manager) }
            }
        }
        impl<'id>
            ::oxidd_core::HasApplyCache<<$dd as $crate::util::type_cons::DD>::Manager<'id>, $op>
            for $name<'id>
        {
            type ApplyCache = $crate::util::apply_cache::ApplyCache<
                <$dd as $crate::util::type_cons::DD>::Manager<'id>,
                $op,
                $arity,
            >;

            #[inline]
            fn apply_cache(&self) -> &Self::ApplyCache {
                &self.apply_cache
            }

            #[inline]
            fn apply_cache_mut(&mut self) -> &mut Self::ApplyCache {
                &mut self.apply_cache
            }
        }

        impl<'id>
            AsRef<::oxidd_rules_zbdd::ZBDDCache<<$dd as $crate::util::type_cons::DD>::Edge<'id>>>
            for $name<'id>
        {
            #[inline(always)]
            fn as_ref(
                &self,
            ) -> &::oxidd_rules_zbdd::ZBDDCache<<$dd as $crate::util::type_cons::DD>::Edge<'id>>
            {
                &self.zbdd_cache
            }
        }
        impl<'id>
            AsMut<::oxidd_rules_zbdd::ZBDDCache<<$dd as $crate::util::type_cons::DD>::Edge<'id>>>
            for $name<'id>
        {
            #[inline(always)]
            fn as_mut(
                &mut self,
            ) -> &mut ::oxidd_rules_zbdd::ZBDDCache<<$dd as $crate::util::type_cons::DD>::Edge<'id>>
            {
                &mut self.zbdd_cache
            }
        }
    };
}

cfg_if::cfg_if! {
    if #[cfg(feature = "manager-pointer")] {
        pub use pointer::{ZBDDFunction, ZBDDManagerRef};
    } else if #[cfg(feature = "manager-index")] {
        pub use index::{ZBDDFunction, ZBDDManagerRef};
    } else {
        pub type ZBDDFunction = ();
        pub type ZBDDManagerRef = ();
    }
}

#[allow(missing_docs)]
#[deprecated = "use ZBDDFunction instead"]
pub type ZBDDSet = ZBDDFunction;

/// Create a new manager for a simple binary decision diagram
#[allow(unused_variables)]
pub fn new_manager(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> ZBDDManagerRef {
    let manager_ref = {
        cfg_if::cfg_if! {
            if #[cfg(feature = "manager-pointer")] {
                pointer::ZBDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
            } else if #[cfg(feature = "manager-index")] {
                index::ZBDDManagerRef::new_manager(inner_node_capacity, 2, apply_cache_capacity, threads)
            } else {
                unreachable!()
            }
        }
    };
    manager_ref.with_manager_exclusive(|manager| ::oxidd_rules_zbdd::ZBDDCache::rebuild(manager));
    manager_ref
}

/// Print statistics to stderr
pub fn print_stats() {
    #[cfg(not(feature = "statistics"))]
    eprintln!("[statistics feature disabled]");

    #[cfg(feature = "statistics")]
    oxidd_rules_zbdd::print_stats();
}

#[cfg(all(feature = "manager-index", not(feature = "manager-pointer")))]
mod index {
    use oxidd_manager_index::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_index::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_zbdd::ZBDDOp;
    use oxidd_rules_zbdd::ZBDDRules;
    use oxidd_rules_zbdd::ZBDDTerminal;

    use crate::util::type_cons::DD;

    crate::util::dd_index_based!(ZBDD {
        node: NodeWithLevelCons<2>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<ZBDDTerminal>,
        rules: ZBDDRulesCons for ZBDDRules,
        manager_data: ZBDDManagerDataCons for ZBDDManagerData,
        terminals: 2,
    });

    manager_data!(ZBDDManagerData for ZBDD, operator: ZBDDOp, cache_max_arity: 3);

    crate::util::manager_ref_index_based!(pub struct ZBDDManagerRef(<ZBDD as DD>::ManagerRef) with ZBDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_zbdd::ZBDDFunction<<ZBDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_zbdd::ZBDDFunctionMT<<ZBDD as DD>::Function>;

    /// Boolean function (or bit vector set) represented as ZBDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::BooleanFunction,
        oxidd_derive::BooleanVecSet,
    )]
    #[use_manager_ref(ZBDDManagerRef, ZBDDManagerRef(inner))]
    pub struct ZBDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: ZBDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for ZBDDFunction {}
}

#[cfg(feature = "manager-pointer")]
mod pointer {
    use oxidd_manager_pointer::node::fixed_arity::NodeWithLevelCons;
    use oxidd_manager_pointer::terminal_manager::StaticTerminalManagerCons;
    use oxidd_rules_zbdd::ZBDDOp;
    use oxidd_rules_zbdd::ZBDDRules;
    use oxidd_rules_zbdd::ZBDDTerminal;

    use crate::util::type_cons::DD;

    crate::util::dd_pointer_based!(ZBDD {
        node: NodeWithLevelCons<2>,
        edge_tag: (),
        terminal_manager: StaticTerminalManagerCons<ZBDDTerminal>,
        rules: ZBDDRulesCons for ZBDDRules,
        manager_data: ZBDDManagerDataCons for ZBDDManagerData,
        tag_bits: 2,
    });

    manager_data!(ZBDDManagerData for ZBDD, operator: ZBDDOp, cache_max_arity: 3);

    crate::util::manager_ref_pointer_based!(pub struct ZBDDManagerRef(<ZBDD as DD>::ManagerRef) with ZBDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_zbdd::ZBDDFunction<<ZBDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_zbdd::ZBDDFunctionMT<<ZBDD as DD>::Function>;

    /// Boolean function (or bit vector set) represented as ZBDD
    #[derive(
        Clone,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        oxidd_derive::Function,
        oxidd_derive::BooleanFunction,
        oxidd_derive::BooleanVecSet,
    )]
    #[use_manager_ref(ZBDDManagerRef, ZBDDManagerRef(inner))]
    pub struct ZBDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: ZBDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for ZBDDFunction {}
}

use oxidd_core::ManagerRef;
pub use oxidd_rules_zbdd::make_node;
pub use oxidd_rules_zbdd::var_boolean_function;
