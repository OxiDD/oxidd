/// @file  concepts.hpp
/// @brief C++20 concepts

#pragma once
#if __cplusplus >= 202002L

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>

#include <oxidd/util.hpp>

/// C++20 concepts to ease meta-programming
namespace oxidd::concepts {

/// Function represented as decision diagram
///
/// A function is the combination of a reference to an oxidd::concepts::manager
/// and a (possibly tagged) edge pointing to a node.
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
///
/// ### Decision Diagram Operations
///
/// - `f.node_count()` counts the descendant nodes including terminal nodes. `f`
///   must be valid. Acquires a shared manager lock.
template <class F>
concept function =
    std::regular<F> && std::totally_ordered<F> && requires(const F f) {
      { f.is_invalid() } -> std::same_as<bool>;

      { std::hash<F>{}(f) } -> std::convertible_to<std::size_t>;

      { f.node_count() } -> std::same_as<std::size_t>;
    };

/// Manager storing nodes and ensuring their uniqueness
///
/// A manager is the data structure responsible for storing nodes and ensuring
/// their uniqueness. It also defines the variable order.
///
/// Every manager implementation `M` has an associated type `M::function`
/// modelling the oxidd::concepts::function concept.
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
///
/// ### Decision Diagram Operations
///
/// - `m.num_inner_nodes()` returns the count of inner nodes. `m` must be valid.
///   Acquires a shared manager lock.
template <class M>
concept manager =
    std::regular<M> && function<typename M::function> && requires(const M m) {
      { m.is_invalid() } -> std::same_as<bool>;

      { m.num_inner_nodes() } -> std::same_as<size_t>;
    };

/// Boolean function 𝔹ⁿ → 𝔹 represented as decision diagram
///
/// - `f.cofactors()` computes the cofactors `(f_true, f_false)` of `f`
/// - `f.cofactor_true()` computes the cofactor `f_true` of `f`
/// - `f.cofactor_false()` computes the cofactor `f_false` of `f`
/// - `~f` computes the negation of `f`
/// - `f & g` computes the conjunction `f ∧ g`
/// - `f &= g` is shorthand for `f = f & g`
/// - `f | g` computes the disjunction `f ∨ g`
/// - `f |= g` is shorthand for `f = f | g`
/// - `f ^ g` computes the disjunction `f ⊕ g`
/// - `f ^= g` is shorthand for `f = f ^ g`
/// - `f.nand(g)` computes the negated conjunction `f ⊼ g`
/// - `f.nor(g)` computes the negated disjunction `f ⊽ g`
/// - `f.equiv(g)` computes the equivalence `f ↔ g`
/// - `f.imp(g)` computes the implication `f → g` (or `f ≤ g`)
/// - `f.imp_strict(g)` computes the strict implication `f < g`
/// - `f.ite(g, h)` computes the conditional `f ? g : h`
/// - `f.satisfiable()` checks if there is a satisfying assignment for `f`. `f`
///   must be valid.
/// - `f.valid()` checks if there are only satisfying assignments for `f`. `f`
///   must be valid.
/// - `f.sat_count_double(vars)` computes the count of satisfying assignments
///   assuming the domain of `f` has `vars` many variables. `f` must be valid.
/// - `f.pick_cube()` returns a satisfying oxidd::util::assignment if `f` is
///   satisfiable, or an empty oxidd::util::assignment otherwise. `f` must be
///   valid.
/// - `f.eval(args)` evaluates the Boolean function `f` with arguments `args`
///   and returns the respective result. `args` determines the valuation for all
///   variables. Missing values are assumed to be false. The order is
///   irrelevant. All variables must point to inner nodes. `f` and all variables
///   must be valid.
///
/// All of the operations above acquire a shared manager lock.
///
/// @see oxidd::concepts::function
template <class F>
concept boolean_function =
    function<F> && requires(const F bf, F mut_bf, level_no_t levels,
                            util::slice<std::pair<F, bool>> args) {
      // cofactors
      { bf.cofactors() } -> std::same_as<std::pair<F, F>>;
      { bf.cofactor_true() } -> std::same_as<F>;
      { bf.cofactor_false() } -> std::same_as<F>;

      // negation
      { ~bf } -> std::same_as<F>;
      // conjunction
      { bf &bf } -> std::same_as<F>;
      { mut_bf &= bf } -> std::same_as<F &>;
      // disjunction
      { bf | bf } -> std::same_as<F>;
      { mut_bf |= bf } -> std::same_as<F &>;
      // exclusive disjunction
      { bf ^ bf } -> std::same_as<F>;
      { mut_bf ^= bf } -> std::same_as<F &>;
      // negated conjunction
      { bf.nand(bf) } -> std::same_as<F>;
      // negated disjunction
      { bf.nor(bf) } -> std::same_as<F>;
      // equivalence
      { bf.equiv(bf) } -> std::same_as<F>;
      // implication
      { bf.imp(bf) } -> std::same_as<F>;
      // strict implication
      { bf.imp_strict(bf) } -> std::same_as<F>;
      // if-then-else
      { bf.ite(bf, bf) } -> std::same_as<F>;

      { bf.satisfiable() } -> std::same_as<bool>;
      { bf.valid() } -> std::same_as<bool>;

      { bf.sat_count_double(levels) } -> std::same_as<double>;

      { bf.pick_cube() } -> std::same_as<util::assignment>;

      { bf.eval(args) } -> std::same_as<bool>;
    };

/// Quantification extension for oxidd::concepts::boolean_function
///
/// `f.forall(vars)` computes the universal quantification, `f.exist(vars)` the
/// existential, and `f.unique(vars)` the unique quantification over `vars`.
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. These operations remove all occurrences of the variables by the
/// respective quantification. Quantification of a Boolean function `f(…, x, …)`
/// over a single variable `x` is `f(…, 0, …) o f(…, 1, …)`, where `o` is `∧`
/// for universal, `∨` for existential, and `⊕` for unique quantification.
///
/// All operations here acquire a shared manager lock.
template <class F>
concept boolean_function_quant = boolean_function<F> && requires(const F bf) {
  { bf.forall(bf) } -> std::same_as<F>;
  { bf.exist(bf) } -> std::same_as<F>;
  { bf.unique(bf) } -> std::same_as<F>;
};

/// Manager for Boolean functions
///
/// - `m.new_var()` creates a fresh variable, i.e., a function that is true iff
///   the variable is true. This adds a new level to the decision diagram. `m`
///   must be valid. Acquires an exclusive manager lock.
/// - `m.t()` returns the constant true Boolean function ⊤. `m` must be valid.
///   Acquires a shared manager lock.
/// - `m.f()` returns the constant false Boolean function ⊥. `m` must be valid.
///   Acquires a shared manager lock.
///
/// @see oxidd::concepts::manager, oxidd::concepts::boolean_function
template <class M>
concept boolean_function_manager =
    manager<M> && boolean_function<typename M::function> &&
    requires(const M m, M mut_m) {
      { mut_m.new_var() } -> std::same_as<typename M::function>;
      { m.t() } -> std::same_as<typename M::function>;
      { m.f() } -> std::same_as<typename M::function>;
    };

} // namespace oxidd::concepts

#endif // __cplusplus >= 202002L
