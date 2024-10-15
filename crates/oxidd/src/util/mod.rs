//! Various utilities for working with DDs

pub(crate) mod apply_cache;
pub(crate) mod type_cons;

pub use oxidd_core::util::num;
pub use oxidd_core::util::AllocResult;
pub use oxidd_core::util::Borrowed;
pub use oxidd_core::util::IsFloatingPoint;
pub use oxidd_core::util::OptBool;
pub use oxidd_core::util::OutOfMemory;
pub use oxidd_core::util::Rng;
pub use oxidd_core::util::SatCountCache;
pub use oxidd_core::util::SatCountNumber;
pub use rustc_hash::FxHasher;

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
            /// [`oxidd_core::util::GCContainer::pre_gc()`] /
            /// [`oxidd_core::util::GCContainer::post_gc()`]
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

        impl<'id, $($($gen),*)?> ::oxidd_core::util::GCContainer<<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>>
            for $name<'id, $($($gen),*)?> $(where $($where)*)?
        {
            #[inline]
            fn pre_gc(&self, manager: &<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>) {
                self.apply_cache.pre_gc(manager)
            }
            #[inline]
            unsafe fn post_gc(&self, manager: &<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>) {
                // SAFETY: inherited from outer
                unsafe { self.apply_cache.post_gc(manager) }
            }
        }

        impl<'id, $($($gen),*)?> ::oxidd_core::HasApplyCache<<$dd$(<$($dd_gen),*>)? as $crate::util::type_cons::DD>::Manager<'id>, $op>
            for $name<'id, $($($gen),*)?> $(where $($where)*)?
        {
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
        /// Reference to a [`Manager`][crate::Manager]
        #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name$(<$($gen),*>)?($inner) $(where $($where)*)?;

        impl$(<$($gen),*>)? Clone for $name$(<$($gen),*>)? $(where $($where)*)? {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl<'__a, '__id, $($($gen),*)?> From<&'__a <$inner as $crate::ManagerRef>::Manager<'__id>> for $name$(<$($gen),*>)? $(where $($where)*)? {
            #[inline]
            fn from(manager: &'__a <$inner as $crate::ManagerRef>::Manager<'__id>) -> Self {
                Self(<$inner as From<&'__a <$inner as $crate::ManagerRef>::Manager<'__id>>>::from(manager))
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


        impl$(<$($gen),*>)? $crate::HasWorkers for $name$(<$($gen),*>)? $(where $($where)*)? {
            type WorkerPool = <$inner as $crate::HasWorkers>::WorkerPool;

            #[inline]
            fn workers(&self) -> &Self::WorkerPool {
                self.0.workers()
            }
        }

        impl$(<$($gen),*>)? $name$(<$($gen),*>)? $(where $($where)*)? {
            /// Create a new manager instance
            ///
            /// - `inner_node_capacity` is the maximum number of inner nodes that can be
            ///   stored
            /// - `terminal_node_capacity` is the maximum number of terminal nodes
            /// - `apply_cache_capacity` refers to the maximum number of apply cache
            ///   entries
            /// - `threads` is the thread count for the worker pool
            pub fn new_manager(
                inner_node_capacity: usize,
                terminal_node_capacity: usize,
                apply_cache_capacity: usize,
                threads: u32,
            ) -> Self {
                assert!(
                    (inner_node_capacity + terminal_node_capacity) as u64 <= (1 << u32::BITS),
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
        /// Reference to a [`Manager`][::oxidd::Manager]
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($inner);

        impl<'__a, '__id> From<&'__a <$inner as $crate::ManagerRef>::Manager<'__id>> for $name {
            #[inline]
            fn from(manager: &'__a <$inner as $crate::ManagerRef>::Manager<'__id>) -> Self {
                Self(<$inner as From<
                    &'__a <$inner as $crate::ManagerRef>::Manager<'__id>,
                >>::from(manager))
            }
        }

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

        impl $crate::HasWorkers for $name {
            type WorkerPool = <$inner as $crate::HasWorkers>::WorkerPool;

            #[inline]
            fn workers(&self) -> &Self::WorkerPool {
                self.0.workers()
            }
        }

        impl $name {
            /// Create a new manager instance
            ///
            /// - `inner_node_capacity` is the maximum number of inner nodes that can be
            ///   stored
            /// - `apply_cache_capacity` refers to the maximum number of apply cache
            ///   entries
            /// - `threads` is the thread count for the worker pool
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
            fn into_raw(self) -> (*const std::ffi::c_void, usize) {
                let (ptr, index) = self.0.into_inner().into_raw();
                (ptr, index as usize)
            }

            #[inline]
            unsafe fn from_raw(ptr: *const std::ffi::c_void, index: usize) -> Self {
                // SAFETY: Invariants are upheld by the caller.
                Self(<$inner>::from(unsafe {
                    ::oxidd_manager_index::manager::Function::from_raw(ptr, index as u32)
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
            fn into_raw(self) -> (*const std::ffi::c_void, usize) {
                (self.0.into_inner().into_raw(), 0)
            }

            #[inline]
            unsafe fn from_raw(ptr: *const std::ffi::c_void, _index: usize) -> Self {
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
