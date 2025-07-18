/// @file  concepts.hpp
/// @brief C++20 concepts

#pragma once

#include <concepts>

#include <oxidd/bridge.hpp>
#include <oxidd/util.hpp>

/// C++20 concepts to ease meta-programming
///
/// Since OxiDD 0.11, the concepts declared in this namespace are simply defined
/// via inheritance from classes in the oxidd::bridge namespace. Previously, the
/// concepts were defined just by requiring certain methods and associated
/// types. While the previous approach permits alternative implementations of
/// the concepts that do not just expose OxiDD's C FFI, it requires a higher
/// maintenance effort. Should there be a desire to restore the old behavior,
/// please open an issue.
namespace oxidd::concepts {

/// Function represented as decision diagram
///
/// A function is the combination of a reference to an
/// `oxidd::concepts::manager` and a (possibly tagged) edge pointing to a node.
/// Obtaining the manager reference is possible via the `containing_manager()`
/// method. Every function implementation `F` has an associated manager type
/// `F::manager` modelling `oxidd::concepts::manager`, and
/// `containing_manager()` returns an instance thereof.
///
/// ### Reference Counting
///
/// In some sense, implementations behave similar to `std::shared_ptr`: copying
/// instances just increments an atomic reference count, moving out of an
/// instance invalidates it. Invalid function instances are also produced by the
/// default constructor and decision diagram operations that run out of memory.
/// Checking if an instance is invalid is possible via the `is_invalid()`
/// method. Unless explicitly stated otherwise, invalid functions may be passed
/// to a method. If one operand of a decision diagram operation is invalid, then
/// the operation produces an invalid function. This permits chaining of these
/// operations without checking for out of memory issues in between all the
/// time.
///
/// ### Ordering, Hashing
///
/// Functions are hashable and totally ordered according to an arbitrary order.
/// Two functions `f`, `g` are equal `f == g` iff they are stored in the same
/// manager and are structurally equal, i.e., they have the same underlying
/// edge. In the following, we denote structural equality as `f ≅ g`.
///
/// By “ordered according to an arbitrary order” we mean: Assuming two functions
/// `f`, `g` with `f < g`, then if either `f` or `g` is deleted and recreated
/// later on, `f < g` does not necessarily hold anymore. Similarly, the hash
/// value of `f` may change if deleting `f` and recreating it later on.
/// Moreover, assume `f < g` and two structurally equivalent functions `f2`,
/// `g2` in different managers (i.e., `f ≅ f2` but `f != f2`, and `g ≅ g2` but
/// `g != g2`), then `f2 < g2` does not necessarily hold.
///
/// In general, structural equality on decision diagrams implies semantic
/// equality. The comparisons and the hash implementation typically operate on
/// underlying pointer or index values, making the implementations very
/// efficient. However, such implementations also lead to the aforementioned
/// restrictions.
template <class F>
concept function =
    std::regular<F> && std::totally_ordered<F> &&
    std::same_as<typename F::manager::function, F> &&
    std::derived_from<
        F, bridge::function<F, typename F::manager, typename F::c_api_t>> &&
    // manager requirements
    std::regular<typename F::manager> &&
    std::derived_from<
        typename F::manager,
        bridge::manager<typename F::manager, F, typename F::manager::c_api_t>>;

/// Manager storing nodes and ensuring their uniqueness
///
/// A manager is the data structure responsible for storing nodes and ensuring
/// their uniqueness. It also defines the variable order.
///
/// Every manager implementation `M` has an associated type `M::function`
/// modelling the `oxidd::concepts::function` concept.
///
/// ### Reference Counting
///
/// Implementations of this concept are similar to `std::shared_ptr`: copying
/// instances just increments an atomic reference count, moving out of an
/// instance invalidates it. Also, the default constructor yields an invalid
/// manager instance. Checking if an instance is invalid is possible via the
/// is_invalid() method.
///
/// ### Concurrency
///
/// Implementations supporting concurrency have an internal read/write lock.
/// Many operations acquire this lock for reading (shared) or writing
/// (exclusive).
template <class M>
concept manager = function<typename M::function> &&
                  std::same_as<typename M::function::manager, M>;

/// Substitution extension for `oxidd::concepts::function`
///
/// See `oxidd::concepts::substitution` for more details on the motivation
/// behind substitution classes.
template <class F>
concept function_subst =
    function<F> && std::same_as<typename F::substitution::function, F> &&
    std::derived_from<F, bridge::function_subst<F>> &&
    // substitution requirements
    std::same_as<typename F::substitution, bridge::substitution<F>>;

/// Substitution mapping variables to replacement functions
///
/// The intent behind substitution classes is to optimize the case where the
/// same substitution is applied multiple times. We would like to re-use apply
/// cache entries across these operations, and therefore, we need a compact
/// identifier for the substitution.
template <class S>
concept substitution = function_subst<typename S::function> &&
                       std::same_as<typename S::function::substitution, S>;

/// Boolean function 𝔹ⁿ → 𝔹 represented as decision diagram
///
/// This extends `oxidd::concepts::function`.
template <class F>
concept boolean_function =
    function<F> && std::derived_from<F, bridge::boolean_function<F>> &&
    // manager requirements
    std::derived_from<typename F::manager,
                      bridge::boolean_function_manager<typename F::manager, F>>;

/// Manager for Boolean functions
///
/// @see  `oxidd::concepts::manager`, `oxidd::concepts::boolean_function`
template <class M>
concept boolean_function_manager =
    manager<M> && boolean_function<typename M::function>;

/// Quantification extension for `oxidd::concepts::boolean_function`
///
/// All operations here acquire the manager's lock for shared access.
template <class F>
concept boolean_function_quant =
    boolean_function<F> &&
    std::derived_from<F, bridge::boolean_function_quant<F>>;

} // namespace oxidd::concepts
