//! Function traits

use std::fmt::Display;
use std::hash::{BuildHasher, Hash};

use nanorand::Rng;

use crate::util::num::F64;
use crate::util::{
    AllocResult, Borrowed, EdgeDropGuard, NodeSet, OptBool, SatCountCache, SatCountNumber,
    Substitution,
};
use crate::{DiagramRules, Edge, InnerNode, LevelNo, Manager, ManagerRef, Node};

/// Shorthand to get the [`Edge`] type associated with a [`Function`]
pub type EdgeOfFunc<'id, F> = <<F as Function>::Manager<'id> as Manager>::Edge;
/// Shorthand to get the edge tag type associated with a [`Function`]
pub type ETagOfFunc<'id, F> = <<F as Function>::Manager<'id> as Manager>::EdgeTag;
/// Shorthand to get the [`InnerNode`] type associated with a [`Function`]
pub type INodeOfFunc<'id, F> = <<F as Function>::Manager<'id> as Manager>::InnerNode;

/// Function in a decision diagram
///
/// A function is some kind of external reference to a node as opposed to
/// [`Edge`]s, which are diagram internal. A function also includes a reference
/// to the diagram's manager. So one may view a function as an [`Edge`] plus a
/// [`ManagerRef`].
///
/// Functions are what the library's user mostly works with. There may be
/// subtraits such as `BooleanFunction` in the `oxidd-rules-bdd` crate which
/// provide more functionality, in this case applying connectives of boolean
/// logic to other functions.
///
/// For some methods of this trait, there are notes on locking behavior. In a
/// concurrent setting, a manager has some kind of read/write lock, and
/// [`Self::with_manager_shared()`] / [`Self::with_manager_exclusive()`] acquire
/// this lock accordingly. In a sequential implementation, a
/// [`RefCell`][std::cell::RefCell] or the like may be used instead of lock.
///
/// # Safety
///
/// An implementation must ensure that the "[`Edge`] part" of the function
/// points to a node that is stored in the manager referenced  by the
/// "[`ManagerRef`] part" of the function. All functions of this trait must
/// maintain this link accordingly. In particular, [`Self::as_edge()`] and
/// [`Self::into_edge()`] must panic as specified there.
pub unsafe trait Function: Clone + Ord + Hash {
    /// Type of the associated manager
    ///
    /// This type is generic over a lifetime `'id` to permit the "lifetime
    /// trick" used, e.g., in [`GhostCell`][GhostCell]: The idea is to make the
    /// [`Manager`], [`Edge`] and [`InnerNode`] types [invariant][variance] over
    /// `'id`. Any call to one of the
    /// [`with_manager_shared()`][Function::with_manager_shared] /
    /// [`with_manager_exclusive()`][Function::with_manager_exclusive] functions
    /// of the [`Function`] or [`ManagerRef`] trait, which "generate" a fresh
    /// lifetime `'id`. Now the type system ensures that every edge or node with
    /// `'id` comes belongs to the manager from the `with_manager_*()` call.
    /// This means that we can reduce the amount of runtime checks needed to
    /// uphold the invariant that the children of a node stored in [`Manager`] M
    /// are stored in M as well.
    ///
    /// Note that [`Function`] and [`ManagerRef`] are (typically) outside the
    /// scope of this lifetime trick to make the library more flexible.
    ///
    /// [GhostCell]: https://plv.mpi-sws.org/rustbelt/ghostcell/
    /// [variance]: https://doc.rust-lang.org/reference/subtyping.html
    type Manager<'id>: Manager;

    /// [Manager references][ManagerRef] for [`Self::Manager`]
    type ManagerRef: for<'id> ManagerRef<Manager<'id> = Self::Manager<'id>>;

    /// Create a new function from a manager reference and an edge
    fn from_edge<'id>(manager: &Self::Manager<'id>, edge: EdgeOfFunc<'id, Self>) -> Self;

    /// Create a new function from a manager reference and an edge reference
    #[inline(always)]
    fn from_edge_ref<'id>(manager: &Self::Manager<'id>, edge: &EdgeOfFunc<'id, Self>) -> Self {
        Self::from_edge(manager, manager.clone_edge(edge))
    }

    /// Converts this function into the underlying edge (as reference), checking
    /// that it belongs to the given `manager`
    ///
    /// Panics if the function does not belong to `manager`.
    fn as_edge<'id>(&self, manager: &Self::Manager<'id>) -> &EdgeOfFunc<'id, Self>;

    /// Converts this function into the underlying edge, checking that it
    /// belongs to the given `manager`
    ///
    /// Panics if the function does not belong to `manager`.
    fn into_edge<'id>(self, manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;

    /// Clone the [`ManagerRef`] part
    fn manager_ref(&self) -> Self::ManagerRef;

    /// Obtain a shared manager reference as well as the underlying edge
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxidd_core::function::Function;
    /// fn my_eq<F: Function>(f: &F, g: &F) -> bool {
    ///     f.with_manager_shared(|manager, f_edge| {
    ///         // Do something meaningful with `manager` and `edge` (the following
    ///         // is better done using `f == g` without `with_manager_shared()`)
    ///         let g_edge = g.as_edge(manager);
    ///         f_edge == g_edge
    ///     })
    /// }
    /// ```
    fn with_manager_shared<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> T;

    /// Obtain an exclusive manager reference as well as the underlying edge
    ///
    /// Locking behavior: acquires the manager's lock for exclusive access.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxidd_core::{*, function::Function, util::AllocResult};
    /// /// Adds a binary node on a new level with children `f` and `g`
    /// fn foo<F: Function>(f: &F, g: &F) -> AllocResult<F> {
    ///     f.with_manager_exclusive(|manager, f_edge| {
    ///         let fe = manager.clone_edge(f_edge);
    ///         let ge = manager.clone_edge(g.as_edge(manager));
    ///         let he = manager.add_level(|level| InnerNode::new(level, [fe, ge]))?;
    ///         Ok(F::from_edge(manager, he))
    ///     })
    /// }
    /// ```
    fn with_manager_exclusive<F, T>(&self, f: F) -> T
    where
        F: for<'id> FnOnce(&mut Self::Manager<'id>, &EdgeOfFunc<'id, Self>) -> T;

    /// Count the number of nodes in this function, including terminal nodes
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn node_count(&self) -> usize {
        fn inner<M: Manager>(manager: &M, e: &M::Edge, set: &mut M::NodeSet) {
            if set.insert(e) {
                if let Node::Inner(node) = manager.get_node(e) {
                    for e in node.children() {
                        inner(manager, &*e, set)
                    }
                }
            }
        }

        self.with_manager_shared(|manager, edge| {
            let mut set = Default::default();
            inner(manager, edge, &mut set);
            set.len()
        })
    }
}

/// Substitution extension for [`Function`]
pub trait FunctionSubst: Function {
    /// Substitute variables in `self` according to `substitution`
    ///
    /// The substitution is performed in a parallel fashion, e.g.:
    /// `(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and the function in `substitution` don't belong to the
    /// same manager.
    fn substitute<'a>(
        &'a self,
        substitution: impl Substitution<Var = &'a Self, Replacement = &'a Self>,
    ) -> AllocResult<Self> {
        if substitution.pairs().len() == 0 {
            return Ok(self.clone());
        }
        self.with_manager_shared(|manager, edge| {
            Ok(Self::from_edge(
                manager,
                Self::substitute_edge(
                    manager,
                    edge,
                    substitution.map(|(v, r)| {
                        (v.as_edge(manager).borrowed(), r.as_edge(manager).borrowed())
                    }),
                )?,
            ))
        })
    }

    /// `Edge` version of [`Self::substitute()`]
    #[must_use]
    fn substitute_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        substitution: impl Substitution<
            Var = Borrowed<'a, EdgeOfFunc<'id, Self>>,
            Replacement = Borrowed<'a, EdgeOfFunc<'id, Self>>,
        >,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
}

/// Boolean functions 𝔹ⁿ → 𝔹
///
/// As a user of this trait, you are probably most interested in methods like
/// [`Self::not()`], [`Self::and()`], and [`Self::or()`]. As an implementor, it
/// suffices to implement the functions operating on edges.
pub trait BooleanFunction: Function {
    /// Get the always false function `⊥`
    fn f<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::f_edge(manager))
    }
    /// Get the always true function `⊤`
    fn t<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::t_edge(manager))
    }

    /// Get a fresh variable, i.e., a function that is true if and only if the
    /// variable is true. This adds a new level to a decision diagram.
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self>;

    /// Get the cofactors `(f_true, f_false)` of `self`
    ///
    /// Let f(x₀, …, xₙ) be represented by `self`, where x₀ is (currently) the
    /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ)
    /// and f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
    ///
    /// Note that the domain of f is 𝔹ⁿ⁺¹ while the domain of f<sub>true</sub>
    /// and f<sub>false</sub> is 𝔹ⁿ. This is irrelevant in case of BDDs and
    /// BCDDs, but not for ZBDDs: For instance, g(x₀) = x₀ and g'(x₀, x₁) = x₀
    /// have the same representation as BDDs or BCDDs, but different
    /// representations as ZBDDs.
    ///
    /// Structurally, the cofactors are simply the children in case of BDDs and
    /// ZBDDs. (For BCDDs, the edge tags are adjusted accordingly.) On these
    /// representations, runtime is thus in O(1).
    ///
    /// Returns `None` iff `self` references a terminal node. If you only need
    /// `f_true` or `f_false`, [`Self::cofactor_true`] or
    /// [`Self::cofactor_false`] are slightly more efficient.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactors(&self) -> Option<(Self, Self)> {
        self.with_manager_shared(|manager, f| {
            let (ft, ff) = Self::cofactors_edge(manager, f)?;
            Some((
                Self::from_edge_ref(manager, &ft),
                Self::from_edge_ref(manager, &ff),
            ))
        })
    }

    /// Get the cofactor `f_true` of `self`
    ///
    /// This method is slightly more efficient than [`Self::cofactors`] in case
    /// `f_false` is not needed.
    ///
    /// For a more detailed description, see [`Self::cofactors`].
    ///
    /// Returns `None` iff `self` references a terminal node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactor_true(&self) -> Option<Self> {
        self.with_manager_shared(|manager, f| {
            let (ft, _) = Self::cofactors_edge(manager, f)?;
            Some(Self::from_edge_ref(manager, &ft))
        })
    }

    /// Get the cofactor `f_false` of `self`
    ///
    /// This method is slightly more efficient than [`Self::cofactors`] in case
    /// `f_true` is not needed.
    ///
    /// For a more detailed description, see [`Self::cofactors`].
    ///
    /// Returns `None` iff `self` references a terminal node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactor_false(&self) -> Option<Self> {
        self.with_manager_shared(|manager, f| {
            let (_, ff) = Self::cofactors_edge(manager, f)?;
            Some(Self::from_edge_ref(manager, &ff))
        })
    }

    /// Compute the negation `¬self`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn not(&self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, edge| {
            Ok(Self::from_edge(manager, Self::not_edge(manager, edge)?))
        })
    }
    /// Compute the negation `¬self`, owned version
    ///
    /// Compared to [`Self::not()`], this method does not need to clone the
    /// function, so when the implementation is using (e.g.) complemented edges,
    /// this might be a little bit faster than [`Self::not()`].
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn not_owned(self) -> AllocResult<Self> {
        self.not()
    }
    /// Compute the conjunction `self ∧ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn and(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::and_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the disjunction `self ∨ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn or(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::or_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the negated conjunction `self ⊼ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn nand(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::nand_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the negated disjunction `self ⊽ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn nor(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::nor_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the exclusive disjunction `self ⊕ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn xor(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::xor_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the equivalence `self ↔ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn equiv(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::equiv_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the implication `self → rhs` (or `self ≤ rhs`)
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn imp(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::imp_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the strict implication `self < rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn imp_strict(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::imp_strict_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Get the always false function `⊥` as edge
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;
    /// Get the always true function `⊤` as edge
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;

    /// Get the cofactors `(f_true, f_false)` of `f`, edge version
    ///
    /// Returns `None` iff `f` references a terminal node. For more details on
    /// the semantics of `f_true` and `f_false`, see [`Self::cofactors`].
    #[inline]
    #[allow(clippy::type_complexity)]
    fn cofactors_edge<'a, 'id>(
        manager: &'a Self::Manager<'id>,
        f: &'a EdgeOfFunc<'id, Self>,
    ) -> Option<(
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
    )> {
        if let Node::Inner(node) = manager.get_node(f) {
            Some(Self::cofactors_node(f.tag(), node))
        } else {
            None
        }
    }

    /// Get the cofactors `(f_true, f_false)` of `node`, assuming an incoming
    /// edge with `EdgeTag`
    ///
    /// Returns `None` iff `f` references a terminal node. For more details on
    /// the semantics of `f_true` and `f_false`, see [`Self::cofactors`].
    ///
    /// Implementation note: The default implementation assumes that
    /// [cofactor 0][DiagramRules::cofactor] corresponds to `f_true` and
    /// [cofactor 1][DiagramRules::cofactor] corresponds to `f_false`.
    #[inline]
    fn cofactors_node<'a, 'id>(
        tag: ETagOfFunc<'id, Self>,
        node: &'a INodeOfFunc<'id, Self>,
    ) -> (
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
    ) {
        let cofactor = <<Self::Manager<'id> as Manager>::Rules as DiagramRules<_, _, _>>::cofactor;
        (cofactor(tag, node, 0), cofactor(tag, node, 1))
    }

    /// Compute the negation `¬edge`, edge version
    #[must_use]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the negation `¬edge`, owned edge version
    ///
    /// Compared to [`Self::not_edge()`], this method does not need to clone the
    /// edge, so when the implementation is using (e.g.) complemented edges,
    /// this might be a little bit faster than [`Self::not_edge()`].
    #[must_use]
    fn not_edge_owned<'id>(
        manager: &Self::Manager<'id>,
        edge: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Self::not_edge(manager, &edge)
    }

    /// Compute the conjunction `lhs ∧ rhs`, edge version
    #[must_use]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the disjunction `lhs ∨ rhs`, edge version
    #[must_use]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the negated conjunction `lhs ⊼ rhs`, edge version
    #[must_use]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the negated disjunction `lhs ⊽ rhs`, edge version
    #[must_use]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the exclusive disjunction `lhs ⊕ rhs`, edge version
    #[must_use]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the equivalence `lhs ↔ rhs`, edge version
    #[must_use]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the implication `lhs → rhs`, edge version
    #[must_use]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the strict implication `lhs < rhs`, edge version
    #[must_use]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Returns `true` iff `self` is satisfiable, i.e. is not `⊥`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn satisfiable(&self) -> bool {
        self.with_manager_shared(|manager, edge| {
            let f = EdgeDropGuard::new(manager, Self::f_edge(manager));
            edge != &*f
        })
    }

    /// Returns `true` iff `self` is valid, i.e. is `⊤`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn valid(&self) -> bool {
        self.with_manager_shared(|manager, edge| {
            let t = EdgeDropGuard::new(manager, Self::t_edge(manager));
            edge == &*t
        })
    }

    /// Compute `if self { then_case } else { else_case }`
    ///
    /// This is equivalent to `(self ∧ then_case) ∨ (¬self ∧ else_case)` but
    /// possibly more efficient than computing all the
    /// conjunctions/disjunctions.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self`, `then_case`, and `else_case` don't belong to the same
    /// manager.
    fn ite(&self, then_case: &Self, else_case: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, if_edge| {
            let then_edge = then_case.as_edge(manager);
            let else_edge = else_case.as_edge(manager);
            let res = Self::ite_edge(manager, if_edge, then_edge, else_edge)?;
            Ok(Self::from_edge(manager, res))
        })
    }

    /// Compute `if if_edge { then_edge } else { else_edge }` (edge version)
    ///
    /// This is equivalent to `(self ∧ then_case) ∨ (¬self ∧ else_case)` but
    /// possibly more efficient than computing all the
    /// conjunctions/disjunctions.
    #[must_use]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let f = EdgeDropGuard::new(manager, Self::and_edge(manager, if_edge, then_edge)?);
        let g = EdgeDropGuard::new(manager, Self::imp_strict_edge(manager, if_edge, else_edge)?);
        Self::or_edge(manager, &*f, &*g)
    }

    /// Count the number of satisfying assignments, assuming `vars` input
    /// variables
    ///
    /// The `cache` can be used to speed up multiple model counting operations
    /// for functions in the same decision diagram. If the model counts of just
    /// one function are of interest, one may simply pass an empty
    /// [`SatCountCache`] (using `&mut SatCountCache::default()`). The cache
    /// will automatically be invalidated in case there have been reordering
    /// operations or `vars` changed since the last query (see
    /// [`SatCountCache::clear_if_invalid()`]). Still, it is the caller's
    /// responsibility to not use the cache for different managers.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn sat_count<N: SatCountNumber, S: std::hash::BuildHasher>(
        &self,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        self.with_manager_shared(|manager, edge| Self::sat_count_edge(manager, edge, vars, cache))
    }

    /// `Edge` version of [`Self::sat_count()`]
    fn sat_count_edge<'id, N: SatCountNumber, S: std::hash::BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N;

    /// Pick a cube of this function
    ///
    /// A cube `c` of a function `f` is a satisfying assignment, i.e., `c → f`
    /// holds, and can be represented as a conjunction of literals. It does
    /// not necessarily define all variables in the function's domain (it is
    /// not necessarily a canonical minterm). For most (if not all) kinds of
    /// decision diagrams, cubes have at most one node per level.
    ///
    /// `order` is a list of variables. If it is non-empty, it must contain as
    /// many variables as there are levels.
    ///
    /// Returns `None` if the function is false. Otherwise, this method returns
    /// a vector where the i-th entry indicates if the i-th variable of `order`
    /// (or the variable currently at the i-th level in case `order` is empty)
    /// is true, false, or "don't care".
    ///
    /// Whenever a value for a variable needs to be chosen (i.e., it cannot be
    /// left as a don't care), `choice` is called to determine the valuation for
    /// this variable. The argument of type [`LevelNo`] is the level
    /// corresponding to that variable. It is guaranteed that `choice` will
    /// only be called at most once for each level. The [`Edge`] argument is
    /// guaranteed to point to an inner node at the respective level. (We
    /// pass an [`Edge`] and not an [`InnerNode`] reference since [`Edge`]s
    /// provide more information, e.g., the [`NodeID`][Edge::node_id()].)
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn pick_cube<'a, I: ExactSizeIterator<Item = &'a Self>>(
        &'a self,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl for<'id> FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> Option<Vec<OptBool>> {
        self.with_manager_shared(|manager, edge| {
            Self::pick_cube_edge(
                manager,
                edge,
                order.into_iter().map(|f| f.as_edge(manager)),
                choice,
            )
        })
    }

    /// Pick a symbolic cube of this function, i.e., as decision diagram
    ///
    /// A cube `c` of a function `f` is a satisfying assignment, i.e., `c → f`
    /// holds, and can be represented as a conjunction of literals. It does
    /// not necessarily define all variables in the function's domain (it is
    /// not necessarily a canonical minterm). For most (if not all) kinds of
    /// decision diagrams, cubes have at most one node per level.
    ///
    /// Whenever a value for a variable needs to be chosen (i.e., it cannot be
    /// left as a don't care), `choice` is called to determine the valuation for
    /// this variable. The argument of type [`LevelNo`] is the level
    /// corresponding to that variable. It is guaranteed that `choice` will
    /// only be called at most once for each level. The [`Edge`] argument is
    /// guaranteed to point to an inner node at the respective level. (We
    /// pass an [`Edge`] and not an [`InnerNode`] reference since [`Edge`]s
    /// provide more information, e.g., the [`NodeID`][Edge::node_id()].)
    ///
    /// If `self` is `⊥`, then the return value will be `⊥`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn pick_cube_dd(
        &self,
        choice: impl for<'id> FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> AllocResult<Self> {
        self.with_manager_shared(|manager, edge| {
            let res = Self::pick_cube_dd_edge(manager, edge, choice)?;
            Ok(Self::from_edge(manager, res))
        })
    }

    /// Pick a symbolic cube of this function, i.e., as decision diagram, using
    /// the literals in `literal_set` if there is a choice
    ///
    /// A cube `c` of a function `f` is a satisfying assignment, i.e., `c → f`
    /// holds, and can be represented as a conjunction of literals. It does
    /// not necessarily define all variables in the function's domain (it is
    /// not necessarily a canonical minterm). For most (if not all) kinds of
    /// decision diagrams, cubes have at most one node per level.
    ///
    /// `literal_set` is represented as a conjunction of literals. Whenever
    /// there is a choice for a variable, it will be set to true if the variable
    /// has a positive occurrence in `literal_set`, and set to false if it
    /// occurs negated in `literal_set`. If the variable does not occur in
    /// `literal_set`, then it will be left as don't care if possible, otherwise
    /// an arbitrary (not necessarily random) choice will be performed.
    ///
    /// If `self` is `⊥`, then the return value will be `⊥`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn pick_cube_dd_set(&self, literal_set: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, edge| {
            let res = Self::pick_cube_dd_set_edge(manager, edge, literal_set.as_edge(manager))?;
            Ok(Self::from_edge(manager, res))
        })
    }

    /// `Edge` version of [`Self::pick_cube()`]
    fn pick_cube_edge<'id, 'a, I>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>;

    /// `Edge` version of [`Self::pick_cube_dd()`]
    fn pick_cube_dd_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// `Edge` version of [`Self::pick_cube_dd_set()`]
    fn pick_cube_dd_set_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        literal_set: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Pick a random cube of this function, where each cube has the same
    /// probability of being chosen
    ///
    /// `order` is a list of variables. If it is non-empty, it must contain as
    /// many variables as there are levels.
    ///
    /// Returns `None` if the function is false. Otherwise, this method returns
    /// a vector where the i-th entry indicates if the i-th variable of `order`
    /// (or the variable currently at the i-th level in case `order` is empty)
    /// is true, false, or "don't care". To obtain a total valuation from this
    /// partial valuation, it suffices to pick true or false with probability ½.
    /// (Note that this function returns a partial valuation with n "don't
    /// cares" with a probability that is 2<sup>n</sup> as high as the
    /// probability of any total valuation.)
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn pick_cube_uniform<'a, I: ExactSizeIterator<Item = &'a Self>, S: BuildHasher>(
        &'a self,
        order: impl IntoIterator<IntoIter = I>,
        cache: &mut SatCountCache<F64, S>,
        rng: &mut crate::util::Rng,
    ) -> Option<Vec<OptBool>> {
        self.with_manager_shared(|manager, edge| {
            Self::pick_cube_uniform_edge(
                manager,
                edge,
                order.into_iter().map(|f| f.as_edge(manager)),
                cache,
                rng,
            )
        })
    }

    /// `Edge` version of [`Self::pick_cube_uniform()`]
    fn pick_cube_uniform_edge<'id, 'a, I, S>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        order: impl IntoIterator<IntoIter = I>,
        cache: &mut SatCountCache<F64, S>,
        rng: &mut crate::util::Rng,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
        S: BuildHasher,
    {
        let vars = manager.num_levels();
        Self::pick_cube_edge(manager, edge, order, |manager, edge, _| {
            let tag = edge.tag();
            // `edge` is guaranteed to point to an inner node
            let node = manager.get_node(edge).unwrap_inner();
            let (t, e) = Self::cofactors_node(tag, node);
            let t_count = Self::sat_count_edge(manager, &*t, vars, cache).0;
            let e_count = Self::sat_count_edge(manager, &*e, vars, cache).0;
            rng.generate::<f64>() < t_count / (t_count + e_count)
        })
    }

    /// Evaluate this Boolean function
    ///
    /// `args` consists of pairs `(variable, value)` and determines the
    /// valuation for all variables. Missing values are assumed to be false.
    /// However, note that the arguments may also determine the domain,
    /// e.g., in case of ZBDDs. If values are specified multiple times, the
    /// last one counts.
    ///
    /// Note that all variables in `args` must be handles for the respective
    /// decision diagram levels, i.e., the Boolean function representing the
    /// variable in case of B(C)DDs, and a singleton set for ZBDDs.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if any function in `args` refers to a terminal node.
    fn eval<'a>(&'a self, args: impl IntoIterator<Item = (&'a Self, bool)>) -> bool {
        self.with_manager_shared(|manager, edge| {
            Self::eval_edge(
                manager,
                edge,
                args.into_iter()
                    .map(|(f, b)| (f.as_edge(manager).borrowed(), b)),
            )
        })
    }

    /// `Edge` version of [`Self::eval()`]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
    ) -> bool;
}

// The `cfg_attr` below is used such that cbindgen does not output the
// Rust-specific documentation.

/// Binary operators on Boolean functions
#[cfg_attr(
    all(),
    doc = "

The operators are used by the combined apply and quantification operations of
the [`BooleanFunctionQuant`] trait. The operators themselves correspond to the
ones defined in [`BooleanFunction`]."
)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum BooleanOperator {
    /// Conjunction `lhs ∧ rhs`
    And,
    /// Disjunction `lhs ∨ rhs`
    Or,
    /// Exclusive disjunction `lhs ⊕ rhs`
    Xor,
    /// Equivalence `lhs ↔ rhs`
    Equiv,
    /// Negated conjunction `lhs ⊼ rhs`
    Nand,
    /// Negated disjunction `lhs ⊽ rhs`
    Nor,
    /// Implication `lhs → rhs`
    Imp,
    /// Strict implication `lhs < rhs`
    ImpStrict,
}

/// cbindgen:ignore
unsafe impl crate::Countable for BooleanOperator {
    const MAX_VALUE: usize = BooleanOperator::ImpStrict as usize;

    fn as_usize(self) -> usize {
        self as usize
    }

    fn from_usize(value: usize) -> Self {
        use BooleanOperator::*;
        match () {
            _ if value == And as usize => And,
            _ if value == Or as usize => Or,
            _ if value == Xor as usize => Xor,
            _ if value == Equiv as usize => Equiv,
            _ if value == Nand as usize => Nand,
            _ if value == Nor as usize => Nor,
            _ if value == Imp as usize => Imp,
            _ if value == ImpStrict as usize => ImpStrict,
            _ => panic!("{value} does not correspond to a Boolean operator"),
        }
    }
}

/// cbindgen:ignore
impl Display for BooleanOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use BooleanOperator::*;
        match self {
            And => write!(f, "∧"),
            Or => write!(f, "∨"),
            Xor => write!(f, "⊕"),
            Equiv => write!(f, "↔"),
            Nand => write!(f, "⊼"),
            Nor => write!(f, "⊽"),
            Imp => write!(f, "→"),
            ImpStrict => write!(f, "<"),
        }
    }
}

/// Quantification extension for [`BooleanFunction`]
pub trait BooleanFunctionQuant: BooleanFunction {
    /// Restrict a set of `vars` to constant values
    ///
    /// `vars` conceptually is a partial assignment, represented as the
    /// conjunction of positive or negative literals, depending on whether the
    /// variable should be mapped to true or false.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `vars` don't belong to the same manager.
    fn restrict(&self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::restrict_edge(manager, root, vars.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Compute the universal quantification over `vars`
    ///
    /// `vars` is a set of variables, which in turn is just the conjunction of
    /// the variables. This operation removes all occurrences of the variables
    /// by universal quantification. Universal quantification `∀x. f(…, x, …)`
    /// of a Boolean function `f(…, x, …)` over a single variable `x` is
    /// `f(…, 0, …) ∧ f(…, 1, …)`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `vars` don't belong to the same manager.
    fn forall(&self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::forall_edge(manager, root, vars.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Compute the existential quantification over `vars`
    ///
    /// `vars` is a set of variables, which in turn is just the conjunction of
    /// the variables. This operation removes all occurrences of the variables
    /// by existential quantification. Existential quantification
    /// `∃x. f(…, x, …)` of a Boolean function `f(…, x, …)` over a single
    /// variable `x` is `f(…, 0, …) ∨ f(…, 1, …)`.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `vars` don't belong to the same manager.
    fn exist(&self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::exist_edge(manager, root, vars.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Compute the unique quantification over `vars`
    ///
    /// `vars` is a set of variables, which in turn is just the conjunction of
    /// the variables. This operation removes all occurrences of the variables
    /// by unique quantification. Unique quantification `∃!x. f(…, x, …)` of a
    /// Boolean function `f(…, x, …)` over a single variable `x` is
    /// `f(…, 0, …) ⊕ f(…, 1, …)`.
    ///
    /// Unique quantification is also known as the
    /// [Boolean difference](https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors)
    /// or
    /// [Boolean derivative](https://en.wikipedia.org/wiki/Boolean_differential_calculus).
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `vars` don't belong to the same manager.
    fn unique(&self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::unique_edge(manager, root, vars.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Combined application of `op` and quantification `∀x. self <op> rhs`,
    /// where `<op>` is any of the operations from [`BooleanOperator`]
    ///
    /// See also [`Self::forall()`] and the trait [`BooleanFunction`] for more
    /// details.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` and `vars` don't belong to the same manager.
    fn apply_forall(&self, op: BooleanOperator, rhs: &Self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::apply_forall_edge(
                manager,
                op,
                root,
                rhs.as_edge(manager),
                vars.as_edge(manager),
            )?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Combined application of `op` and quantification `∃x. self <op> rhs`,
    /// where `<op>` is any of the operations from [`BooleanOperator`]
    ///
    /// See also [`Self::exist()`] and the trait [`BooleanFunction`] for more
    /// details.
    ///
    /// Panics if `self` and `rhs` and `vars` don't belong to the same manager.
    fn apply_exist(&self, op: BooleanOperator, rhs: &Self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::apply_exist_edge(
                manager,
                op,
                root,
                rhs.as_edge(manager),
                vars.as_edge(manager),
            )?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Combined application of `op` and quantification `∃!x. self <op> rhs`,
    /// where `<op>` is any of the operations from [`BooleanOperator`]
    ///
    /// See also [`Self::unique()`] and the trait [`BooleanFunction`] for more
    /// details.
    ///
    /// Panics if `self` and `rhs` and `vars` don't belong to the same manager.
    fn apply_unique(&self, op: BooleanOperator, rhs: &Self, vars: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, root| {
            let e = Self::apply_unique_edge(
                manager,
                op,
                root,
                rhs.as_edge(manager),
                vars.as_edge(manager),
            )?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Restrict a set of `vars` to constant values, edge version
    ///
    /// See [`Self::restrict()`] for more details.
    #[must_use]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the universal quantification of `root` over `vars`, edge
    /// version
    ///
    /// See [`Self::forall()`] for more details.
    #[must_use]
    fn forall_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the existential quantification of `root` over `vars`, edge
    /// version
    ///
    /// See [`Self::exist()`] for more details.
    #[must_use]
    fn exist_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the unique quantification of `root` over `vars`, edge version
    ///
    /// See [`Self::unique()`] for more details.
    #[must_use]
    fn unique_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Combined application of `op` and forall quantification, edge version
    ///
    /// See [`Self::apply_forall()`] for more details.
    #[must_use]
    fn apply_forall_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        // Naive default implementation
        use BooleanOperator::*;
        let inner = EdgeDropGuard::new(
            manager,
            match op {
                And => Self::and_edge(manager, lhs, rhs),
                Or => Self::or_edge(manager, lhs, rhs),
                Xor => Self::xor_edge(manager, lhs, rhs),
                Equiv => Self::equiv_edge(manager, lhs, rhs),
                Nand => Self::nand_edge(manager, lhs, rhs),
                Nor => Self::nor_edge(manager, lhs, rhs),
                Imp => Self::imp_edge(manager, lhs, rhs),
                ImpStrict => Self::imp_strict_edge(manager, lhs, rhs),
            }?,
        );

        Self::forall_edge(manager, &inner, vars)
    }

    /// Combined application of `op` and existential quantification, edge
    /// version
    ///
    /// See [`Self::apply_exist()`] for more details.
    #[must_use]
    fn apply_exist_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        // Naive default implementation
        use BooleanOperator::*;
        let inner = EdgeDropGuard::new(
            manager,
            match op {
                And => Self::and_edge(manager, lhs, rhs),
                Or => Self::or_edge(manager, lhs, rhs),
                Xor => Self::xor_edge(manager, lhs, rhs),
                Equiv => Self::equiv_edge(manager, lhs, rhs),
                Nand => Self::nand_edge(manager, lhs, rhs),
                Nor => Self::nor_edge(manager, lhs, rhs),
                Imp => Self::imp_edge(manager, lhs, rhs),
                ImpStrict => Self::imp_strict_edge(manager, lhs, rhs),
            }?,
        );

        Self::exist_edge(manager, &inner, vars)
    }

    /// Combined application of `op` and unique quantification, edge version
    ///
    /// See [`Self::apply_unique()`] for more details.
    #[must_use]
    fn apply_unique_edge<'id>(
        manager: &Self::Manager<'id>,
        op: BooleanOperator,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        // Naive default implementation
        use BooleanOperator::*;
        let inner = EdgeDropGuard::new(
            manager,
            match op {
                And => Self::and_edge(manager, lhs, rhs),
                Or => Self::or_edge(manager, lhs, rhs),
                Xor => Self::xor_edge(manager, lhs, rhs),
                Equiv => Self::equiv_edge(manager, lhs, rhs),
                Nand => Self::nand_edge(manager, lhs, rhs),
                Nor => Self::nor_edge(manager, lhs, rhs),
                Imp => Self::imp_edge(manager, lhs, rhs),
                ImpStrict => Self::imp_strict_edge(manager, lhs, rhs),
            }?,
        );

        Self::unique_edge(manager, &inner, vars)
    }
}

/// Set of Boolean vectors
///
/// A Boolean function f: 𝔹ⁿ → 𝔹 may also be regarded as a set S ∈ 𝒫(𝔹ⁿ), where
/// S = {v ∈ 𝔹ⁿ | f(v) = 1}. f is also called the characteristic function of S.
/// We can even view a Boolean vector as a subset of some "Universe" U, so we
/// also have S ∈ 𝒫(𝒫(U)). For example, let U = {a, b, c}. The function a is
/// the set of all sets containing a, {a, ab, abc, ac} (for the sake of
/// readability, we write ab for the set {a, b}). Conversely, the set {a} is the
/// function a ∧ ¬b ∧ ¬c.
///
/// Counting the number of elements in a `BoolVecSet` is equivalent to counting
/// the number of satisfying assignments of its characteristic function. Hence,
/// you may use [`BooleanFunction::sat_count()`] for this task.
///
/// The functions of this trait can be implemented efficiently for ZBDDs.
///
/// As a user of this trait, you are probably most interested in methods like
/// [`Self::union()`], [`Self::intsec()`], and [`Self::diff()`]. As an
/// implementor, it suffices to implement the functions operating on edges.
pub trait BooleanVecSet: Function {
    /// Add a new variable to the manager and get the corresponding singleton
    /// set
    ///
    /// This adds a new level to the decision diagram.
    fn new_singleton<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self>;

    /// Get the empty set ∅
    ///
    /// This corresponds to the Boolean function ⊥.
    fn empty<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::empty_edge(manager))
    }

    /// Get the set {∅}
    ///
    /// This corresponds to the Boolean function ¬x₁ ∧ … ∧ ¬xₙ
    fn base<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::base_edge(manager))
    }

    /// Get the set of subsets of `self` not containing `var`, formally
    /// `{s ∈ self | var ∉ s}`
    ///
    /// `var` must be a singleton set, otherwise the result is unspecified.
    /// Ideally, the implementation panics.
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `var` do not belong to the same manager.
    fn subset0(&self, var: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let e = Self::subset0_edge(manager, set, var.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Get the set of subsets of `self` containing `var`, formally
    /// `{s ∈ self | var ∈ s}`
    ///
    /// `var` must be a singleton set, otherwise the result is unspecified.
    /// Ideally, the implementation panics.
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `var` do not belong to the same manager.
    fn subset1(&self, var: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let e = Self::subset1_edge(manager, set, var.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Get the set of subsets derived from `self` by adding `var` to the
    /// subsets that do not contain `var`, and removing `var` from the subsets
    /// that contain `var`, formally
    /// `{s ∪ {var} | s ∈ self ∧ var ∉ s} ∪ {s ∖ {var} | s ∈ self ∧ var ∈ s}`
    ///
    /// `var` must be a singleton set, otherwise the result is unspecified.
    /// Ideally, the implementation panics.
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `var` do not belong to the same manager.
    fn change(&self, var: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, set| {
            let e = Self::change_edge(manager, set, var.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Compute the union `self ∪ rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn union(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::union_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Compute the intersection `self ∩ rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn intsec(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::intsec_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Compute the set difference `self ∖ rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn diff(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::diff_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Edge version of [`Self::empty()`]
    fn empty_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;

    /// Edge version of [`Self::base()`]
    fn base_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;

    /// Edge version of [`Self::subset0()`]
    fn subset0_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &EdgeOfFunc<'id, Self>,
        var: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Edge version of [`Self::subset1()`]
    fn subset1_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &EdgeOfFunc<'id, Self>,
        var: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Edge version of [`Self::change()`]
    fn change_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &EdgeOfFunc<'id, Self>,
        var: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the union `lhs ∪ rhs`, edge version
    fn union_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the intersection `lhs ∩ rhs`, edge version
    fn intsec_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the set difference `lhs ∖ rhs`, edge version
    fn diff_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
}

/// Basic trait for numbers
///
/// [`zero()`][Self::zero], [`one()`][Self::one], and [`nan()`][Self::nan] are
/// implemented as functions because depending on the number underlying type,
/// it can be hard/impossible to obtain a `const` for these values.
/// This trait also includes basic arithmetic methods. This is to avoid cloning
/// of big integers. We could also require `&Self: Add<&Self, Output = Self>`
/// etc., but this would lead to ugly trait bounds.
///
/// Used by [`PseudoBooleanFunction::Number`]
pub trait NumberBase: Clone + Eq + Hash + PartialOrd {
    /// Get the value 0
    fn zero() -> Self;
    /// Get the value 1
    fn one() -> Self;

    /// Get the value "not a number," e.g. the result of a division 0/0.
    ///
    /// `Self::nan() == Self::nan()` should evaluate to `true`.
    fn nan() -> Self;

    /// Returns `true` iff `self == Self::zero()`
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
    /// Returns `true` iff `self == Self::one()`
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
    /// Returns `true` iff `self == Self::nan()`
    fn is_nan(&self) -> bool {
        self == &Self::nan()
    }

    /// Compute `self + rhs`
    fn add(&self, rhs: &Self) -> Self;
    /// Compute `self - rhs`
    fn sub(&self, rhs: &Self) -> Self;
    /// Compute `self * rhs`
    fn mul(&self, rhs: &Self) -> Self;
    /// Compute `self / rhs`
    fn div(&self, rhs: &Self) -> Self;
}

/// Pseudo-Boolean function 𝔹ⁿ → ℝ
pub trait PseudoBooleanFunction: Function {
    /// The number type used for the functions' target set.
    type Number: NumberBase;

    /// Get the constant `value`
    fn constant<'id>(manager: &Self::Manager<'id>, value: Self::Number) -> AllocResult<Self> {
        Ok(Self::from_edge(
            manager,
            Self::constant_edge(manager, value)?,
        ))
    }

    /// Get a fresh variable, i.e. a function that is 1 if the variable is true
    /// and 0 otherwise. This adds a new level to a decision diagram.
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self>;

    /// Point-wise addition `self + rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn add(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::add_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Point-wise subtraction `self - rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn sub(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::sub_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Point-wise multiplication `self * rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn mul(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::mul_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Point-wise division `self / rhs`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn div(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::div_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Point-wise minimum `min(self, rhs)`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn min(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::min_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Point-wise maximum `max(self, rhs)`
    ///
    /// Locking behavior: acquires a shared manager lock
    ///
    /// Panics if `self` and `rhs` do not belong to the same manager.
    fn max(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::max_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Get the constant `value`, edge version
    fn constant_edge<'id>(
        manager: &Self::Manager<'id>,
        value: Self::Number,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Point-wise addition `self + rhs`, edge version
    fn add_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Point-wise subtraction `self - rhs`, edge version
    fn sub_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Point-wise multiplication `self * rhs`, edge version
    fn mul_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Point-wise division `self / rhs`, edge version
    fn div_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Point-wise minimum `min(self, rhs)`, edge version
    fn min_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Point-wise maximum `max(self, rhs)`, edge version
    fn max_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
}

/// Function of three valued logic
pub trait TVLFunction: Function {
    /// Get the always false function `⊥`
    fn f<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::f_edge(manager))
    }
    /// Get the always true function `⊤`
    fn t<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::t_edge(manager))
    }
    /// Get the "unknown" function `U`
    fn u<'id>(manager: &Self::Manager<'id>) -> Self {
        Self::from_edge(manager, Self::t_edge(manager))
    }

    /// Get the cofactors `(f_true, f_unknown, f_false)` of `self`
    ///
    /// Let f(x₀, …, xₙ) be represented by `self`, where x₀ is (currently) the
    /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ)
    /// and f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
    ///
    /// Note that the domain of f is 𝔹ⁿ⁺¹ while the domain of f<sub>true</sub>
    /// and f<sub>false</sub> is 𝔹ⁿ.
    ///
    /// Returns `None` iff `self` references a terminal node. If you only need
    /// `f_true`, `f_unknown`, or `f_false`, [`Self::cofactor_true`],
    /// [`Self::cofactor_unknown`], or [`Self::cofactor_false`] are slightly
    /// more efficient.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactors(&self) -> Option<(Self, Self, Self)> {
        self.with_manager_shared(|manager, f| {
            let (ft, fu, ff) = Self::cofactors_edge(manager, f)?;
            Some((
                Self::from_edge_ref(manager, &ft),
                Self::from_edge_ref(manager, &fu),
                Self::from_edge_ref(manager, &ff),
            ))
        })
    }

    /// Get the cofactor `f_true` of `self`
    ///
    /// This method is slightly more efficient than [`Self::cofactors`] in case
    /// `f_unknown` and `f_false` are not needed.
    ///
    /// For a more detailed description, see [`Self::cofactors`].
    ///
    /// Returns `None` iff `self` references a terminal node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactor_true(&self) -> Option<Self> {
        self.with_manager_shared(|manager, f| {
            let (ft, _, _) = Self::cofactors_edge(manager, f)?;
            Some(Self::from_edge_ref(manager, &ft))
        })
    }
    /// Get the cofactor `f_unknown` of `self`
    ///
    /// This method is slightly more efficient than [`Self::cofactors`] in case
    /// `f_true` and `f_false` are not needed.
    ///
    /// For a more detailed description, see [`Self::cofactors`].
    ///
    /// Returns `None` iff `self` references a terminal node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactor_unknown(&self) -> Option<Self> {
        self.with_manager_shared(|manager, f| {
            let (_, fu, _) = Self::cofactors_edge(manager, f)?;
            Some(Self::from_edge_ref(manager, &fu))
        })
    }
    /// Get the cofactor `f_false` of `self`
    ///
    /// This method is slightly more efficient than [`Self::cofactors`] in case
    /// `f_true` and `f_unknown` are not needed.
    ///
    /// For a more detailed description, see [`Self::cofactors`].
    ///
    /// Returns `None` iff `self` references a terminal node.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn cofactor_false(&self) -> Option<Self> {
        self.with_manager_shared(|manager, f| {
            let (_, _, ff) = Self::cofactors_edge(manager, f)?;
            Some(Self::from_edge_ref(manager, &ff))
        })
    }

    /// Get a fresh variable, i.e. a function that is true if the variable is
    /// true, false if the variable is false, and undefined otherwise. This adds
    /// a new level to a decision diagram.
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self>;

    /// Compute the negation `¬self`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn not(&self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, edge| {
            Ok(Self::from_edge(manager, Self::not_edge(manager, edge)?))
        })
    }
    /// Compute the negation `¬self`, owned version
    ///
    /// Compared to [`Self::not()`], this method does not need to clone the
    /// function, so when the implementation is using (e.g.) complemented edges,
    /// this might be a little bit faster than [`Self::not()`].
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    fn not_owned(self) -> AllocResult<Self> {
        self.not()
    }
    /// Compute the conjunction `self ∧ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn and(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::and_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the disjunction `self ∨ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn or(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::or_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the negated conjunction `self ⊼ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn nand(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::nand_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the negated disjunction `self ⊽ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn nor(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::nor_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the exclusive disjunction `self ⊕ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn xor(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::xor_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the equivalence `self ↔ rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn equiv(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::equiv_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the implication `self → rhs` (or `self ≤ rhs`)
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn imp(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::imp_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }
    /// Compute the strict implication `self < rhs`
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self` and `rhs` don't belong to the same manager.
    fn imp_strict(&self, rhs: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, lhs| {
            let e = Self::imp_strict_edge(manager, lhs, rhs.as_edge(manager))?;
            Ok(Self::from_edge(manager, e))
        })
    }

    /// Get the always false function `⊥` as edge
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;
    /// Get the always true function `⊤` as edge
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;
    /// Get the "unknown" function `U` as edge
    fn u_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self>;

    /// Get the cofactors `(f_true, f_unknown, f_false)` of `f`, edge version
    ///
    /// Returns `None` iff `f` references a terminal node. For more details on
    /// the semantics of `f_true` and `f_false`, see [`Self::cofactors`].
    #[inline]
    #[allow(clippy::type_complexity)]
    fn cofactors_edge<'a, 'id>(
        manager: &'a Self::Manager<'id>,
        f: &'a EdgeOfFunc<'id, Self>,
    ) -> Option<(
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
    )> {
        if let Node::Inner(node) = manager.get_node(f) {
            Some(Self::cofactors_node(f.tag(), node))
        } else {
            None
        }
    }

    /// Get the cofactors `(f_true, f_unknown, f_false)` of `node`, assuming an
    /// incoming edge with `EdgeTag`
    ///
    /// Returns `None` iff `f` references a terminal node. For more details on
    /// the semantics of `f_true` and `f_false`, see [`Self::cofactors`].
    ///
    /// Implementation note: The default implementation assumes that
    /// [cofactor 0][DiagramRules::cofactor] corresponds to `f_true`,
    /// [cofactor 1][DiagramRules::cofactor] corresponds to `f_unknown`, and
    /// [cofactor 2][DiagramRules::cofactor] corresponds to `f_false`.
    #[inline]
    #[allow(clippy::type_complexity)]
    fn cofactors_node<'a, 'id>(
        tag: ETagOfFunc<'id, Self>,
        node: &'a INodeOfFunc<'id, Self>,
    ) -> (
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
        Borrowed<'a, EdgeOfFunc<'id, Self>>,
    ) {
        let cofactor = <<Self::Manager<'id> as Manager>::Rules as DiagramRules<_, _, _>>::cofactor;
        (
            cofactor(tag, node, 0),
            cofactor(tag, node, 1),
            cofactor(tag, node, 2),
        )
    }

    /// Compute the negation `¬edge`, edge version
    #[must_use]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute the negation `¬edge`, owned edge version
    ///
    /// Compared to [`Self::not_edge()`], this method does not need to clone the
    /// edge, so when the implementation is using (e.g.) complemented edges,
    /// this might be a little bit faster than [`Self::not_edge()`].
    #[must_use]
    fn not_edge_owned<'id>(
        manager: &Self::Manager<'id>,
        edge: EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Self::not_edge(manager, &edge)
    }

    /// Compute the conjunction `lhs ∧ rhs`, edge version
    #[must_use]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the disjunction `lhs ∨ rhs`, edge version
    #[must_use]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the negated conjunction `lhs ⊼ rhs`, edge version
    #[must_use]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the negated disjunction `lhs ⊽ rhs`, edge version
    #[must_use]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the exclusive disjunction `lhs ⊕ rhs`, edge version
    #[must_use]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the equivalence `lhs ↔ rhs`, edge version
    #[must_use]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the implication `lhs → rhs`, edge version
    #[must_use]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;
    /// Compute the strict implication `lhs < rhs`, edge version
    #[must_use]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>>;

    /// Compute `if self { then_case } else { else_case }`
    ///
    /// This is equivalent to `(self ∧ then_case) ∨ (¬self ∧ else_case)` but
    /// possibly more efficient than computing all the
    /// conjunctions/disjunctions.
    ///
    /// Locking behavior: acquires the manager's lock for shared access.
    ///
    /// Panics if `self`, `then_case`, and `else_case` don't belong to the same
    /// manager.
    fn ite(&self, then_case: &Self, else_case: &Self) -> AllocResult<Self> {
        self.with_manager_shared(|manager, if_edge| {
            let then_edge = then_case.as_edge(manager);
            let else_edge = else_case.as_edge(manager);
            let res = Self::ite_edge(manager, if_edge, then_edge, else_edge)?;
            Ok(Self::from_edge(manager, res))
        })
    }

    /// Compute `if if_edge { then_edge } else { else_edge }` (edge version)
    ///
    /// This is equivalent to `(self ∧ then_case) ∨ (¬self ∧ else_case)` but
    /// possibly more efficient than computing all the
    /// conjunctions/disjunctions.
    #[must_use]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let f = EdgeDropGuard::new(manager, Self::and_edge(manager, if_edge, then_edge)?);
        let g = EdgeDropGuard::new(manager, Self::imp_strict_edge(manager, if_edge, else_edge)?);
        Self::or_edge(manager, &*f, &*g)
    }
}
