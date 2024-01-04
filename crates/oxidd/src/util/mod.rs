pub(crate) mod apply_cache;
pub mod num;
pub(crate) mod type_cons;

// We have a few `allow(unused)` attributes here to not spam the user with
// warnings in case no manager implementation is selected.

#[allow(unused)]
macro_rules! manager_data {
    ($name:ident$(<$($gen:ident),*>)? for $dd:ident$(<$($dd_gen:ident),*>)?, operator: $op:ty, cache_max_arity: $arity:expr $(, where $($where:tt)*)?) => {
        pub struct $name<'id, $($($gen),*)?> $(where $($where)*)? {
            apply_cache: $crate::util::apply_cache::ApplyCache<
                <$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>,
                $op,
                $arity,
            >,
        }

        impl<'id, $($($gen),*)?> $name<'id, $($($gen),*)?> $(where $($where)*)? {
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
                }
            }
        }

        impl<'id, $($($gen),*)?> ::oxidd_core::util::DropWith<<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Edge<'id>>
            for $name<'id, $($($gen),*)?> $(where $($where)*)?
        {
            fn drop_with(
                self,
                drop_edge: impl Fn(<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Edge<'id>),
            ) {
                self.apply_cache.drop_with(drop_edge)
            }
        }

        impl<'id, $($($gen),*)?> ::oxidd_core::HasApplyCache<<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>>
            for $name<'id, $($($gen),*)?> $(where $($where)*)?
        {
            type Operator = $op;
            type ApplyCache = $crate::util::apply_cache::ApplyCache<
                <$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>,
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
    };
}
#[allow(unused)]
pub(crate) use manager_data;

#[allow(unused)]
macro_rules! dd_index_based {
    ($dd:ident$(<$($gen:ident),*>)? {
        node: $nc:ty,
        edge_tag: $et:ty,
        terminal_manager: $tmc:ty,
        rules: $rc:ident for $r:ty,
        manager_data: $mdc:ident for $md:ident$(<$($md_gen:ident),*>)?,
        terminals: $terminals:expr,
    } $(where $($where:tt)*)?) => {
        type $dd$(<$($gen),*>)? = $crate::util::type_cons::IndexDD<$nc, $et, $tmc, $rc, $mdc, $terminals>;

        pub struct $rc;
        impl$(<$($gen),*>)? ::oxidd_manager_index::manager::DiagramRulesCons<$nc, $et, $tmc, $mdc, $terminals>
            for $rc $(where $($where)*)?
        {
            type T<'id> = $r;
        }

        pub struct $mdc;
        impl$(<$($gen),*>)? ::oxidd_manager_index::manager::ManagerDataCons<$nc, $et, $tmc, $rc, $terminals>
            for $mdc $(where $($where)*)?
        {
            type T<'id> = $md<'id, $($($md_gen),*)?>;
        }
    };
}
#[allow(unused)]
pub(crate) use dd_index_based;

#[allow(unused)]
macro_rules! dd_pointer_based {
    ($dd:ident$(<$($gen:ident),*>)? {
        node: $nc:ty,
        edge_tag: $et:ty,
        terminal_manager: $tmc:ty,
        rules: $rc:ident for $r:ty,
        manager_data: $mdc:ident for $md:ident$(<$($md_gen:ident),*>)?,
        tag_bits: $tag_bits:expr,
    } $(where $($where:tt)*)?) => {
        type $dd$(<$($gen),*>)? = $crate::util::type_cons::PointerDD<
            $nc,
            $et,
            $tmc,
            $rc,
            $mdc,
            { $crate::PAGE_SIZE },
            $tag_bits,
        >;

        pub struct $rc;
        impl$(<$($gen),*>)?
            ::oxidd_manager_pointer::manager::DiagramRulesCons<
                $nc,
                $et,
                $tmc,
                $mdc,
                { $crate::PAGE_SIZE },
                $tag_bits,
            > for $rc $(where $($where)*)?
        {
            type T<'id> = $r;
        }

        pub struct $mdc;
        impl$(<$($gen),*>)?
            ::oxidd_manager_pointer::manager::ManagerDataCons<
                $nc,
                $et,
                $tmc,
                $rc,
                { $crate::PAGE_SIZE },
                $tag_bits,
            > for $mdc $(where $($where)*)?
        {
            type T<'id> = $md<'id, $($($md_gen),*)?>;
        }
    };
}
#[allow(unused)]
pub(crate) use dd_pointer_based;

#[allow(unused)]
macro_rules! manager_ref_index_based {
    (pub struct $name:ident$(<$($gen:ident),*>)?($inner:ty) with $mgr_data:ty $(where $($where:tt)*)?) => {
        #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name$(<$($gen),*>)?($inner) $(where $($where)*)?;

        impl$(<$($gen),*>)? Clone for $name$(<$($gen),*>)? $(where $($where)*)? {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl$(<$($gen),*>)? $crate::ManagerRef for $name$(<$($gen),*>)? $(where $($where)*)? {
            type Manager<'__id> = <$inner as $crate::ManagerRef>::Manager<'__id>;

            #[inline]
            fn with_manager_shared<__F, __T>(&self, f: __F) -> __T
            where
                __F: for<'__id> FnOnce(&Self::Manager<'__id>) -> __T,
            {
                self.0.with_manager_shared(f)
            }

            #[inline]
            fn with_manager_exclusive<__F, __T>(&self, f: __F) -> __T
            where
                __F: for<'__id> FnOnce(&mut Self::Manager<'__id>) -> __T,
            {
                self.0.with_manager_exclusive(f)
            }
        }

        impl$(<$($gen),*>)? $crate::RawManagerRef for $name$(<$($gen),*>)? $(where $($where)*)? {
            #[inline]
            fn into_raw(self) -> *const std::ffi::c_void {
                self.0.into_raw()
            }

            #[inline]
            unsafe fn from_raw(raw: *const std::ffi::c_void) -> Self {
                // SAFETY: Invariants are upheld by the caller.
                Self(unsafe { ::oxidd_manager_index::manager::ManagerRef::from_raw(raw) })
            }
        }

        impl$(<$($gen),*>)? $name$(<$($gen),*>)? $(where $($where)*)? {
            pub fn new_manager(
                inner_node_capacity: usize,
                terminal_node_capacity: usize,
                apply_cache_capacity: usize,
                threads: u32,
            ) -> Self {
                assert!(
                    inner_node_capacity + terminal_node_capacity <= (1 << u32::BITS),
                    "`inner_node_capacity ({inner_node_capacity}) + terminal_node_capacity ({terminal_node_capacity})` must be <= 2^32"
                );
                Self(::oxidd_manager_index::manager::new_manager(
                    inner_node_capacity as u32,
                    terminal_node_capacity as u32,
                    threads,
                    // SAFETY: The index-based manager implementation guarantees
                    // all node deletions to be wrapped inside an
                    // `ApplyCache::pre_gc()` / `ApplyCache::post_gc()` pair.
                    unsafe { <$mgr_data>::new(apply_cache_capacity) },
                ))
            }
        }
    };
}
#[allow(unused)]
pub(crate) use manager_ref_index_based;

#[allow(unused)]
macro_rules! manager_ref_pointer_based {
    (pub struct $name:ident($inner:ty) with $mgr_data:ty) => {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($inner);

        impl $crate::ManagerRef for $name {
            type Manager<'id> = <$inner as $crate::ManagerRef>::Manager<'id>;

            #[inline]
            fn with_manager_shared<F, T>(&self, f: F) -> T
            where
                F: for<'id> FnOnce(&Self::Manager<'id>) -> T,
            {
                self.0.with_manager_shared(f)
            }

            #[inline]
            fn with_manager_exclusive<F, T>(&self, f: F) -> T
            where
                F: for<'id> FnOnce(&mut Self::Manager<'id>) -> T,
            {
                self.0.with_manager_exclusive(f)
            }
        }

        impl $crate::RawManagerRef for $name {
            #[inline]
            fn into_raw(self) -> *const std::ffi::c_void {
                self.0.into_raw()
            }

            #[inline]
            unsafe fn from_raw(raw: *const std::ffi::c_void) -> Self {
                // SAFETY: Invariants are upheld by the caller.
                Self(unsafe { ::oxidd_manager_pointer::manager::ManagerRef::from_raw(raw) })
            }
        }

        impl $name {
            #[allow(unused_variables)]
            pub fn new_manager(
                inner_node_capacity: usize,
                apply_cache_capacity: usize,
                threads: u32,
            ) -> Self {
                // SAFETY: The pointer-based manager implementation guarantees
                // all node deletions to be wrapped inside an
                // `ApplyCache::pre_gc()` / `ApplyCache::post_gc()` pair.
                Self(::oxidd_manager_pointer::manager::new_manager(
                    unsafe { <$mgr_data>::new(apply_cache_capacity) },
                    threads,
                ))
            }
        }
    };
}
#[allow(unused)]
pub(crate) use manager_ref_pointer_based;

#[allow(unused)]
macro_rules! derive_raw_function_index_based {
    (for: $name:ident$(<$($gen:ident),*>)?, inner: $inner:ty $(, where $($where:tt)*)?) => {
        impl$(<$($gen),*>)? $crate::RawFunction for $name$(<$($gen),*>)? $(where $($where)*)? {
            #[inline]
            fn into_raw(self) -> (*const std::ffi::c_void, u32) {
                self.0.into_inner().into_raw()
            }

            #[inline]
            unsafe fn from_raw(ptr: *const std::ffi::c_void, index: u32) -> Self {
                // SAFETY: Invariants are upheld by the caller.
                Self(<$inner>::from(unsafe {
                    ::oxidd_manager_index::manager::Function::from_raw(ptr, index)
                }))
            }
        }
    };
}
#[allow(unused)]
pub(crate) use derive_raw_function_index_based;

#[allow(unused)]
macro_rules! derive_raw_function_pointer_based {
    (for: $name:ident, inner: $inner:ident) => {
        impl $crate::RawFunction for $name {
            #[inline]
            fn into_raw(self) -> (*const std::ffi::c_void, u32) {
                (self.0.into_inner().into_raw(), 0)
            }

            #[inline]
            unsafe fn from_raw(ptr: *const std::ffi::c_void, _index: u32) -> Self {
                // SAFETY: Invariants are upheld by the caller.
                Self($inner::from(unsafe {
                    ::oxidd_manager_pointer::manager::Function::from_raw(ptr)
                }))
            }
        }
    };
}
#[allow(unused)]
pub(crate) use derive_raw_function_pointer_based;
