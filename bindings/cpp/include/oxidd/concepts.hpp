/// @file  concepts.hpp
/// @brief C++20 concepts

#pragma once

#include <version> // for __cpp_lib_concepts
#ifdef __cpp_lib_concepts

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>

#ifdef __cpp_lib_ranges
#include <ranges>
#endif // __cpp_lib_ranges

#include <oxidd/util.hpp>

/// C++20 concepts to ease meta-programming
namespace oxidd::concepts {

/// @cond

namespace detail {

// Dummy input_iterator and input_range to specify the substitution concept

template <typename T> struct input_iterator {
  using difference_type = std::ptrdiff_t;
  using value_type = T;

  const T &operator*() const;

  input_iterator &operator++();
  void operator++(int);

  bool operator==(const input_iterator &) const;
};

static_assert(std::input_iterator<input_iterator<int>>);
static_assert(std::equality_comparable<input_iterator<int>>);

#ifdef __cpp_lib_ranges

template <typename T> struct input_range {
  input_iterator<T> begin();
  input_iterator<T> end();
};

static_assert(std::ranges::input_range<input_range<int>>);

#endif // __cpp_lib_ranges

} // namespace detail

/// @endcond

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
///
/// ### Decision Diagram Operations
///
/// - `f.node_count()` counts the descendant nodes including terminal nodes. `f`
///   must be valid. Acquires a shared manager lock.
template <class F>
concept function = (
  std::regular<F> &&
  std::totally_ordered<F> &&
  std::same_as<typename F::manager::function, F> &&
  requires(const F &f) {
    { f.is_invalid() } -> std::same_as<bool>;
    { f.containing_manager() } -> std::same_as<typename F::manager>;

    { std::hash<F>{}(f) } -> std::convertible_to<std::size_t>;

    { f.node_count() } -> std::same_as<std::size_t>;
  } &&
  // manager requirements
  std::regular<typename F::manager> &&
  requires(const typename F::manager &m) {
    { m.is_invalid() } -> std::same_as<bool>;

    { m.num_inner_nodes() } -> std::same_as<size_t>;
  }
);

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
///
/// ### Decision Diagram Operations
///
/// - `m.num_inner_nodes()` returns the count of inner nodes. `m` must be valid.
///   Acquires the manager's lock for shared access.
template <class M>
concept manager = function<typename M::function> &&
                  std::same_as<typename M::function::manager, M>;

/// Substitution extension for `oxidd::concepts::function`
///
/// For every implementation `F` of this concept and every `F f`:
/// - `f.substitute(subst)` computes `f` with variables according to `subst`,
///   where `subst` is of type `F::substitution`.
/// - `F::substitution` models `oxidd::concepts::substitution`.
///
/// See `oxidd::concepts::substitution` for more details on the motivation
/// behind substitution classes.
template <class F>
concept function_subst =
    function<F> && std::same_as<typename F::substitution::function, F> &&
    requires(const F &f, const typename F::substitution &s) {
      { f.substitute(s) } -> std::same_as<F>;
    } &&
    // substitution requirements
    std::movable<typename F::substitution> &&
    std::default_initializable<typename F::substitution> &&
    std::constructible_from<
        typename F::substitution,
        detail::input_iterator<std::tuple<const F &, const F &>>,
        detail::input_iterator<std::tuple<const F &, const F &>>> &&
#ifdef __cpp_lib_ranges
    std::constructible_from<
        typename F::substitution,
        detail::input_range<std::tuple<const F &, const F &>>> &&
#endif // __cpp_lib_ranges
    requires(const typename F::substitution &s) {
      { s.is_invalid() } -> std::same_as<bool>;
    };

/// Substitution mapping variables to replacement functions
///
/// The intent behind substitution classes is to optimize the case where the
/// same substitution is applied multiple times. We would like to re-use apply
/// cache entries across these operations, and therefore, we need a compact
/// identifier for the substitution.
///
/// Every substitution implementation `S` has an associated type `S::function`
/// modelling `oxidd::concepts::function_subst`. Substitutions can be
/// constructed from iterators and ranges over pairs `(const typename
/// S::function &var, const typename S::function &replacement)`.
///
/// This behaves like a `std::unique_ptr` with respect to default/move
/// construction and assignment.
template <class S>
concept substitution = function_subst<typename S::function> &&
                       std::same_as<typename S::function::substitution, S>;

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
/// - `f ^ g` computes the exclusive disjunction `f ⊕ g`
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
/// - `f.pick_cube()` returns a satisfying `oxidd::util::assignment` if `f` is
///   satisfiable, or an empty `oxidd::util::assignment` otherwise. `f` must be
///   valid.
/// - `f.pick_cube_dd()` returns a satisfying assignment as decision diagram if
///   there is one or ⊥.
/// - `f.pick_cube_dd_set(literal_set)` returns a satisfying assignment as
///   decision diagram if there is one or ⊥. Whenever there is a choice for a
///   variable, and the variable is defined in `literal_set`, the respective
///   polarity is used.
/// - `f.eval(args)` evaluates the Boolean function `f` with arguments `args`
///   and returns the respective result. `args` determines the valuation for all
///   variables. Missing values are assumed to be false. The order is
///   irrelevant. All variables must point to inner nodes. `f` and all variables
///   must be valid.
///
/// All of the operations above acquire the manager's lock for shared access.
///
/// This extends `oxidd::concepts::function`.
template <class F>
concept boolean_function =
    function<F> &&
    // `f.sat_count_double` should take `levels` by-value. We require that it
    // isn't taken as a mutable reference, at least.
    requires(const F &f, F &mut_f, const level_no_t &levels,
             util::slice<std::pair<F, bool>> args) {
      // cofactors
      { f.cofactors() } -> std::same_as<std::pair<F, F>>;
      { f.cofactor_true() } -> std::same_as<F>;
      { f.cofactor_false() } -> std::same_as<F>;

      // negation
      { ~f } -> std::same_as<F>;
      // conjunction
      { f &f } -> std::same_as<F>;
      { mut_f &= f } -> std::same_as<F &>;
      // disjunction
      { f | f } -> std::same_as<F>;
      { mut_f |= f } -> std::same_as<F &>;
      // exclusive disjunction
      { f ^ f } -> std::same_as<F>;
      { mut_f ^= f } -> std::same_as<F &>;
      // negated conjunction
      { f.nand(f) } -> std::same_as<F>;
      // negated disjunction
      { f.nor(f) } -> std::same_as<F>;
      // equivalence
      { f.equiv(f) } -> std::same_as<F>;
      // implication
      { f.imp(f) } -> std::same_as<F>;
      // strict implication
      { f.imp_strict(f) } -> std::same_as<F>;
      // if-then-else
      { f.ite(f, f) } -> std::same_as<F>;

      { f.satisfiable() } -> std::same_as<bool>;
      { f.valid() } -> std::same_as<bool>;

      { f.sat_count_double(levels) } -> std::same_as<double>;

      { f.pick_cube() } -> std::same_as<util::assignment>;
      { f.pick_cube_dd() } -> std::same_as<F>;
      { f.pick_cube_dd_set(f) } -> std::same_as<F>;

      { f.eval(args) } -> std::same_as<bool>;
    } &&
    // manager requirements
    requires(const typename F::manager &m, typename F::manager &mut_m) {
      { mut_m.new_var() } -> std::same_as<F>;
      { m.t() } -> std::same_as<F>;
      { m.f() } -> std::same_as<F>;
    };

/// Manager for Boolean functions
///
/// For every implementation `M` of this concept and every `M m`:
/// - `M::function` models `oxidd::concepts::boolean_function`.
/// - `m.new_var()` creates a fresh variable, i.e., a function that is true iff
///   the variable is true. This adds a new level to the decision diagram. `m`
///   must be valid. Acquires the manager's lock for exclusive access.
/// - `m.t()` returns the constant true Boolean function ⊤. `m` must be valid.
///   Acquires the manager's lock for shared access.
/// - `m.f()` returns the constant false Boolean function ⊥. `m` must be valid.
///   Acquires the manager's lock for shared access.
///
/// @see , oxidd::concepts::boolean_function
template <class M>
concept boolean_function_manager =
    manager<M> && boolean_function<typename M::function>;

/// Quantification extension for `oxidd::concepts::boolean_function`
///
/// `f.forall(vars)` computes the universal quantification, `f.exist(vars)` the
/// existential, and `f.unique(vars)` the unique quantification over `vars`.
/// `vars` is a set of variables, which in turn is just the conjunction of the
/// variables. These operations remove all occurrences of the variables by the
/// respective quantification. Quantification of a Boolean function `f(…, x, …)`
/// over a single variable `x` is `f(…, 0, …) o f(…, 1, …)`, where `o` is `∧`
/// for universal, `∨` for existential, and `⊕` for unique quantification.
///
/// There are specialized algorithms for applying an operator and abstracting
/// over some variables via quantification. They may be implemented by the
/// following member functions:
///
/// - `f.apply_forall(op, g, vars)` computes `∀ vars. f <op> g` (i.e., is
///   equivalent to `f.op(g).forall(vars)`)
/// - `f.apply_exist(op, g, vars)` computes `∃ vars. f <op> g`
/// - `f.apply_unique(op, g, vars)` computes `∃! vars. f <op> g`
///
/// All operations here acquire the manager's lock for shared access.
template <class F>
concept boolean_function_quant =
    boolean_function<F> &&
    // `op` should be passed by-value. We require that the methods don't take it
    // as a mutable reference, at least.
    requires(const F &f, const util::boolean_operator &op) {
      { f.forall(f) } -> std::same_as<F>;
      { f.exist(f) } -> std::same_as<F>;
      { f.unique(f) } -> std::same_as<F>;

      { f.apply_forall(op, f, f) } -> std::same_as<F>;
      { f.apply_exist(op, f, f) } -> std::same_as<F>;
      { f.apply_unique(op, f, f) } -> std::same_as<F>;
    };

} // namespace oxidd::concepts

#endif // __cpp_lib_concepts
