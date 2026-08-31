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

/// Create a new manager for a list decision diagram
#[allow(unused_variables)]
pub fn new_manager(
    inner_node_capacity: usize,
    apply_cache_capacity: usize,
    threads: u32,
) -> LDDManagerRef {
    let manager_ref = {
        cfg_if::cfg_if! {
            if #[cfg(feature = "manager-pointer")] {
                pointer::LDDManagerRef::new_manager(inner_node_capacity, apply_cache_capacity, threads)
            } else if #[cfg(feature = "manager-index")] {
                index::LDDManagerRef::new_manager(inner_node_capacity, 2, apply_cache_capacity, threads)
            } else {
                unreachable!()
            }
        }
    };

    // All LDD nodes reside on a single level (level 0); ensure it exists so that
    // node creation via `Manager::level(0)` is valid.
    #[cfg(any(feature = "manager-pointer", feature = "manager-index"))]
    {
        use ::oxidd_core::{Manager, ManagerRef};
        manager_ref.with_manager_exclusive(|manager| {
            if manager.num_levels() == 0 {
                manager.add_vars(1);
            }
        });
    }

    manager_ref
}

/// Print statistics to stderr
pub fn print_stats() {
    #[cfg(not(feature = "statistics"))]
    eprintln!("[statistics feature disabled]");

    #[cfg(feature = "statistics")]
    oxidd_rules_ldd::print_stats();
}

/// We only expose a hard coded u32 valued LDDManager.
pub type Value = u32;

/// Result of [`LDDFunction::relation_product_meta`].
///
/// Bundles the meta-LDD encoding the read/write projection together with the
/// positions of the read and write variables in that encoding.
pub struct RelationProductMeta {
    /// The meta-LDD encoding the read/write projection.
    pub meta: LDDFunction,
    /// The positions of the read variables in the meta encoding.
    pub read_positions: Vec<usize>,
    /// The positions of the write variables in the meta encoding.
    pub write_positions: Vec<usize>,
}

/// Generates the inherent methods of the public `LDDFunction` wrapper by
/// delegating to the inner [`oxidd_rules_ldd::LDDFunction`].
///
/// This is a macro because the concrete inner function type (`FunctionInner`)
/// differs between the index- and pointer-based managers and is only nameable
/// inside their respective modules.
macro_rules! ldd_function_methods {
    () => {
        impl LDDFunction {
            /// Returns the empty set `∅`.
            pub fn empty_set(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(FunctionInner::empty_set(manager)?))
            }

            /// Returns the empty set `∅` as an edge.
            pub fn empty_set_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(FunctionInner::empty_set(manager)?))
            }

            /// Returns the set containing only the empty vector.
            pub fn empty_vector(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(FunctionInner::empty_vector(manager)?))
            }

            /// Returns the LDD containing only `vector`, i.e. `{ vector }`.
            pub fn singleton(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                vector: &[Value],
            ) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(FunctionInner::singleton(manager, vector)?))
            }

            /// Computes a meta-LDD encoding the projection onto the indices in
            /// `proj`, suitable as the argument of [`project`][Self::project].
            pub fn projection_meta(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                proj: &[Value],
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::function::Function;
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::projection_meta(manager, proj)?,
                )))
            }

            /// Computes a meta-LDD for [`relational_product`][Self::relational_product]
            /// from the `read` and `write` projection indices, together with the
            /// resulting read and write positions.
            pub fn relation_product_meta(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                read: &[Value],
                write: &[Value],
            ) -> ::oxidd_core::util::AllocResult<$crate::ldd::RelationProductMeta> {
                use ::oxidd_core::function::Function;
                let ::oxidd_rules_ldd::RelationProductMeta {
                    meta,
                    read_positions,
                    write_positions,
                } = FunctionInner::relation_product_meta(manager, read, write)?;
                Ok($crate::ldd::RelationProductMeta {
                    meta: Self(FunctionInner::from_edge(manager, meta)),
                    read_positions,
                    write_positions,
                })
            }

            /// Computes the union `self ∪ other`.
            pub fn union(&self, other: &Self) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(self.0.union(&other.0)?))
            }

            /// Computes the union `f ∪ g` using the already-borrowed `manager`.
            ///
            /// Unlike [`union`][Self::union], this does not acquire the manager
            /// lock itself, so it can be called from within a
            /// [`with_manager_shared`][::oxidd_core::ManagerRef::with_manager_shared]
            /// block (e.g. to build many nodes under a single lock).
            pub fn union_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                f: &Self,
                g: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::Manager;
                use ::oxidd_core::function::Function;
                let f = manager.clone_edge(f.as_edge(manager));
                let g = manager.clone_edge(g.as_edge(manager));
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::union_edge(manager, f, g)?,
                )))
            }

            /// Computes the set difference `self \ other`.
            pub fn minus(&self, other: &Self) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(self.0.minus(&other.0)?))
            }

            /// Computes the set difference `a \ b` using the already-borrowed
            /// `manager`; the lock-free counterpart of [`minus`][Self::minus].
            pub fn minus_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                a: &Self,
                b: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::Manager;
                use ::oxidd_core::function::Function;
                let a = manager.clone_edge(a.as_edge(manager));
                let b = manager.clone_edge(b.as_edge(manager));
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::minus_edge(manager, a, b)?,
                )))
            }

            /// Computes the intersection `self ∩ other`.
            pub fn intersect(&self, other: &Self) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(self.0.intersect(&other.0)?))
            }

            /// Computes the intersection `a ∩ b` using the already-borrowed
            /// `manager`; the lock-free counterpart of [`intersect`][Self::intersect].
            pub fn intersect_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                a: &Self,
                b: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::Manager;
                use ::oxidd_core::function::Function;
                let a = manager.clone_edge(a.as_edge(manager));
                let b = manager.clone_edge(b.as_edge(manager));
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::intersect_edge(manager, a, b)?,
                )))
            }

            /// Computes the set of vectors reachable in one step from `self` via
            /// the sparse relation `rel`, guided by `meta` (produced by
            /// [`relation_product_meta`][Self::relation_product_meta]).
            pub fn relational_product(
                &self,
                rel: &Self,
                meta: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(self.0.relational_product(&rel.0, &meta.0)?))
            }

            /// Computes the relational product using the already-borrowed
            /// `manager`; the lock-free counterpart of
            /// [`relational_product`][Self::relational_product].
            pub fn relational_product_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                set: &Self,
                rel: &Self,
                meta: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::Manager;
                use ::oxidd_core::function::Function;
                let set = manager.clone_edge(set.as_edge(manager));
                let rel = manager.clone_edge(rel.as_edge(manager));
                let meta = manager.clone_edge(meta.as_edge(manager));
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::relational_product_edge(manager, set, rel, meta)?,
                )))
            }

            /// Computes the set of source vectors in `universe` from which a
            /// vector in `self` is reachable in one step via the sparse relation
            /// `rel`, guided by `meta` (produced by
            /// [`relation_product_meta`][Self::relation_product_meta]).
            ///
            /// This is the inverse of
            /// [`relational_product`][Self::relational_product].
            pub fn relational_predecessor(
                &self,
                rel: &Self,
                meta: &Self,
                universe: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(self.0.relational_predecessor(
                    &rel.0,
                    &meta.0,
                    &universe.0,
                )?))
            }

            /// Computes the relational predecessor using the already-borrowed
            /// `manager`; the lock-free counterpart of
            /// [`relational_predecessor`][Self::relational_predecessor].
            pub fn relational_predecessor_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                set: &Self,
                rel: &Self,
                meta: &Self,
                universe: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::Manager;
                use ::oxidd_core::function::Function;
                let set = manager.clone_edge(set.as_edge(manager));
                let rel = manager.clone_edge(rel.as_edge(manager));
                let meta = manager.clone_edge(meta.as_edge(manager));
                let universe = manager.clone_edge(universe.as_edge(manager));
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::relational_predecessor_edge(manager, set, rel, meta, universe)?,
                )))
            }

            /// Projects the vectors in `self` onto the indices encoded by `proj`
            /// (produced by [`projection_meta`][Self::projection_meta]).
            pub fn project(&self, proj: &Self) -> ::oxidd_core::util::AllocResult<Self> {
                Ok(Self(self.0.project(&proj.0)?))
            }

            /// Projects the vectors in `set` onto the indices encoded by `proj`
            /// using the already-borrowed `manager`; the lock-free counterpart
            /// of [`project`][Self::project].
            pub fn project_edge(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                set: &Self,
                proj: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::Manager;
                use ::oxidd_core::function::Function;
                let set = manager.clone_edge(set.as_edge(manager));
                let proj = manager.clone_edge(proj.as_edge(manager));
                Ok(Self(FunctionInner::from_edge(
                    manager,
                    FunctionInner::project_edge(manager, set, proj)?,
                )))
            }

            /// Returns the number of vectors (lists) contained in `self`.
            pub fn len(&self) -> usize {
                self.0.len()
            }

            /// Returns `true` if `self` is the empty set `∅`, i.e. contains no
            /// vectors.
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Returns `true` if `self` is the set containing only the empty
            /// vector, i.e. the `true` terminal.
            pub fn is_empty_vector(&self) -> bool {
                use ::oxidd_core::{Manager, ManagerRef, function::Function};
                self.manager_ref().with_manager_shared(|manager| {
                    manager
                        .get_node(self.as_edge(manager))
                        .is_terminal(&::oxidd_rules_ldd::LDDTerminal::True)
                })
            }

            /// Returns the `(value, down, right)` triple of the root node, or
            /// `None` if `self` is a terminal (the empty set or empty vector).
            pub fn node(&self) -> Option<(Value, Self, Self)> {
                use ::oxidd_core::{InnerNode, Manager, ManagerRef, Node, function::Function};
                self.manager_ref().with_manager_shared(|manager| {
                    match manager.get_node(self.as_edge(manager)) {
                        Node::Terminal(_) => None,
                        Node::Inner(node) => {
                            let value = *node.get_value();
                            let mut children = node.children();
                            let down = children.next().unwrap();
                            let right = children.next().unwrap();
                            Some((
                                value,
                                Self::from_edge(manager, manager.clone_edge(&down)),
                                Self::from_edge(manager, manager.clone_edge(&right)),
                            ))
                        }
                    }
                })
            }

            /// Creates the LDD node `(value, down, right)`.
            pub fn make_node(
                manager: &<Self as ::oxidd_core::function::Function>::Manager<'_>,
                value: Value,
                down: &Self,
                right: &Self,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                use ::oxidd_core::{InnerNode, LevelView, Manager, function::Function};
                let t = manager.clone_edge(down.as_edge(manager));
                let e = manager.clone_edge(right.as_edge(manager));
                let edge = LevelView::get_or_insert(
                    &mut manager.level(0),
                    <<Self as Function>::Manager<'_> as Manager>::InnerNode::new(0, [t, e], value),
                )?;
                Ok(Self::from_edge(manager, edge))
            }

            /// Returns a stable identifier for the root node of `self`, suitable
            /// for distinguishing nodes (e.g. when emitting DOT output).
            pub fn id(&self) -> ::oxidd_core::NodeID {
                use ::oxidd_core::{Edge, ManagerRef, function::Function};
                self.manager_ref()
                    .with_manager_shared(|manager| self.as_edge(manager).node_id())
            }
        }
    };
}

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

    crate::util::manager_data!(LDDManagerData for LDD, operator: LDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_index_based!(pub struct LDDManagerRef(<LDD as DD>::ManagerRef) with LDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_ldd::LDDFunction<<LDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_ldd::mt::LDDFunctionMT<<LDD as DD>::Function>;

    /// Boolean function represented as BDD
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, oxidd_derive::Function)]
    #[use_manager_ref(LDDManagerRef, LDDManagerRef(inner))]
    pub struct LDDFunction(FunctionInner);
    crate::util::derive_raw_function_index_based!(for: LDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for LDDFunction {}

    ldd_function_methods!();
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

    crate::util::manager_data!(LDDManagerData for LDD, operator: LDDOp, cache_entry_capacity: 4);

    crate::util::manager_ref_pointer_based!(pub struct LDDManagerRef(<LDD as DD>::ManagerRef) with LDDManagerData);

    #[cfg(not(feature = "multi-threading"))]
    type FunctionInner = oxidd_rules_ldd::LDDFunction<<LDD as DD>::Function>;
    #[cfg(feature = "multi-threading")]
    type FunctionInner = oxidd_rules_ldd::mt::LDDFunctionMT<<LDD as DD>::Function>;

    /// Boolean function represented as BDD
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, oxidd_derive::Function)]
    #[use_manager_ref(LDDManagerRef, LDDManagerRef(inner))]
    pub struct LDDFunction(FunctionInner);
    crate::util::derive_raw_function_pointer_based!(for: LDDFunction, inner: FunctionInner);

    // Default implementation suffices
    impl oxidd_dump::dot::DotStyle<()> for LDDFunction {}

    ldd_function_methods!();
}
