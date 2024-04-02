/// @file  bdd.hpp
/// @brief Binary decision diagrams (without complement edges)

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <oxidd/capi.h>
#include <oxidd/util.hpp>

namespace oxidd {

class bdd_function;

/// Manager for binary decision diagrams (without complement edges)
///
/// Models the oxidd::boolean_function_manager concept.
///
/// Instances can safely be sent to other threads.
class bdd_manager {
  /// Wrapped CAPI BDD manager
  capi::oxidd_bdd_manager_t _manager = {._p = nullptr};

  /// Create a new BDD manager from a manager instance of the CAPI
  bdd_manager(capi::oxidd_bdd_manager_t manager) noexcept : _manager(manager) {}

public:
  /// Associated function type
  using function = bdd_function;

  /// Default constructor, yields an invalid manager
  bdd_manager() noexcept = default;
  /// Create a new BDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  bdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
              uint32_t threads) noexcept
      : _manager(capi::oxidd_bdd_manager_new(inner_node_capacity,
                                             apply_cache_capacity, threads)) {}
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Runtime complexity: O(1)
  bdd_manager(const bdd_manager &other) noexcept : _manager(other._manager) {
    capi::oxidd_bdd_manager_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  bdd_manager(bdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~bdd_manager() noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bdd_manager_unref(_manager);
  }

  /// Copy assignment operator
  bdd_manager &operator=(const bdd_manager &rhs) noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bdd_manager_unref(_manager);
    _manager = rhs._manager;
    capi::oxidd_bdd_manager_ref(_manager);
    return *this;
  }
  /// Move assignment operator
  bdd_manager &operator=(bdd_manager &&rhs) noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bdd_manager_unref(_manager);
    _manager = rhs._manager;
    rhs._manager._p = nullptr;
    return *this;
  }

  /// Compare two managers for referential equality
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same manager
  friend bool operator==(const bdd_manager &lhs,
                         const bdd_manager &rhs) noexcept {
    return lhs._manager._p == rhs._manager._p;
  }
  /// Same as `!(lhs == rhs)` (see @ref operator==)
  friend bool operator!=(const bdd_manager &lhs,
                         const bdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor bdd_manager() is
  /// invalid as well as a @ref bdd_manager instance that has been moved (via
  /// bdd_manager(bdd_manager &&other)).
  ///
  /// @returns  `true` iff this manager reference is invalid
  [[nodiscard]] bool is_invalid() const noexcept {
    return _manager._p == nullptr;
  }

  /// @name BDD Construction
  /// @{

  /// Get a fresh variable, i.e., a function that is true if and only if the
  /// variable is true. This adds a new level to the decision diagram.
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires an exclusive manager lock.
  ///
  /// @returns  The BDD function representing the variable
  [[nodiscard]] bdd_function new_var() noexcept;

  /// Get the constant true BCDD function ⊤
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊤ as BDD function
  [[nodiscard]] bdd_function t() const noexcept;
  /// Get the constant false BCDD function ⊥
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊥ as BDD function
  [[nodiscard]] bdd_function f() const noexcept;

  /// @}
  /// @name Statistics
  /// @{

  /// Get the number of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The number of inner nodes
  [[nodiscard]] size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return capi::oxidd_bdd_num_inner_nodes(_manager);
  }

  /// @}
};

/// Boolean function represented as a binary decision diagram (BDD, without
/// complement edges)
///
/// This is essentially a reference to a BDD node.
///
/// Models the oxidd::boolean_function_quant concept.
///
/// Instances can safely be sent to other threads.
class bdd_function {
  /// Wrapped BDD function
  capi::oxidd_bdd_t _func = {._p = nullptr};

  friend class bdd_manager;
  friend struct std::hash<bdd_function>;

  /// Create a new @ref bdd_function from a BDD function instance of the CAPI
  bdd_function(capi::oxidd_bdd_t func) noexcept : _func(func) {}

public:
  /// Default constructor, yields an invalid BCDD function
  bdd_function() noexcept = default;
  /// Copy constructor: increments the internal reference counters
  ///
  /// Runtime complexity: O(1)
  bdd_function(const bdd_function &other) noexcept : _func(other._func) {
    capi::oxidd_bdd_ref(_func);
  }
  /// Move constructor: invalidates `other`
  bdd_function(bdd_function &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~bdd_function() noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bdd_unref(_func);
  }

  /// Copy assignment operator
  bdd_function &operator=(const bdd_function &rhs) noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bdd_unref(_func);
    _func = rhs._func;
    if (_func._p != nullptr)
      capi::oxidd_bdd_ref(_func);
    return *this;
  }
  /// Move assignment operator
  bdd_function &operator=(bdd_function &&rhs) noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bdd_unref(_func);
    _func = rhs._func;
    rhs._func._p = nullptr;
    return *this;
  }

  /// Compare two functions for referential equality
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same node
  friend bool operator==(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  /// Same as `!(lhs == rhs)` (see @ref operator==)
  friend bool operator!=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(lhs == rhs);
  }
  /// Check if `lhs` is less than `rhs` according to an arbitrary total order
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` is less than `rhs` according to the total order
  friend bool operator<(const bdd_function &lhs,
                        const bdd_function &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  /// @ref operator< with arguments swapped
  friend bool operator>(const bdd_function &lhs,
                        const bdd_function &rhs) noexcept {
    return rhs < lhs;
  }
  /// Same as `!(rhs < lhs)` (see @ref operator<)
  friend bool operator<=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(rhs < lhs);
  }
  /// Same as `!(lhs < rhs)` (see @ref operator<)
  friend bool operator>=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(lhs < rhs);
  }

  /// Check if this BDD function is invalid
  ///
  /// A BDD function created by the default constructor bdd_function() is
  /// invalid as well as a @ref bdd_function instance that has been moved
  /// (via bdd_function(bdd_function &&other)). Moreover, if an operation
  /// tries to allocate new nodes but runs out of memory, then it returns an
  /// invalid function.
  ///
  /// @returns  `true` iff this BCDD function is invalid
  [[nodiscard]] bool is_invalid() const noexcept { return _func._p == nullptr; }

  /// @name BDD Construction Operations
  /// @{

  /// Get the cofactors `(f_true, f_false)` of `f`
  ///
  /// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the
  /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
  /// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
  ///
  /// Structurally, the cofactors are the children. If you only need one of the
  /// cofactors, then use cofactor_true() or cofactor_false(). These functions
  /// are slightly more efficient then.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The pair `{f_true, f_false}` if `f` is valid and references an
  ///           inner node, otherwise a pair of invalid functions.
  [[nodiscard]] std::pair<bdd_function, bdd_function>
  cofactors() const noexcept {
    const capi::oxidd_bdd_pair_t p = capi::oxidd_bdd_cofactors(_func);
    return {p.first, p.second};
  }
  /// Get the cofactor `f_true` of `f`
  ///
  /// This function is slightly more efficient than cofactors() in case
  /// `f_false` is not needed.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_true` if `f` is valid and references an inner node, otherwise
  ///           an invalid function.
  [[nodiscard]] bdd_function cofactor_true() const noexcept {
    return capi::oxidd_bdd_cofactor_true(_func);
  }
  /// Get the cofactor `f_false` of `f`
  ///
  /// This function is slightly more efficient than cofactors() in case `f_true`
  /// is not needed.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_false` if `f` is valid and references an inner node,
  ///           otherwise an invalid function.
  [[nodiscard]] bdd_function cofactor_false() const noexcept {
    return capi::oxidd_bdd_cofactor_true(_func);
  }

  /// Compute the BDD for the negation `¬this`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function operator~() const noexcept {
    return capi::oxidd_bdd_not(_func);
  }
  /// Compute the BDD for the conjunction `lhs ∧ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend bdd_function
  operator&(const bdd_function &lhs, const bdd_function &rhs) noexcept {
    return capi::oxidd_bdd_and(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator&
  bdd_function &operator&=(const bdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  /// Compute the BDD for the disjunction `lhs ∨ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  friend bdd_function operator|(const bdd_function &lhs,
                                const bdd_function &rhs) noexcept {
    return capi::oxidd_bdd_or(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator|
  bdd_function &operator|=(const bdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  /// Compute the BDD for the exclusive disjunction `lhs ⊕ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  friend bdd_function operator^(const bdd_function &lhs,
                                const bdd_function &rhs) noexcept {
    return capi::oxidd_bdd_xor(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator^
  bdd_function &operator^=(const bdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  /// Compute the BDD for the negated conjunction `this ⊼ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function nand(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_nand(_func, rhs._func);
  }
  /// Compute the BDD for the negated disjunction `this ⊽ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function nor(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_nor(_func, rhs._func);
  }
  /// Compute the BDD for the equivalence `this ↔ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function equiv(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_equiv(_func, rhs._func);
  }
  /// Compute the BDD for the implication `this → rhs` (or `this ≤ rhs`)
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function imp(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_imp(_func, rhs._func);
  }
  /// Compute the BDD for the strict implication `this < rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function
  imp_strict(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_imp_strict(_func, rhs._func);
  }
  /// Compute the BDD for the conditional `this ? t : e`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |t| · |e|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function ite(const bdd_function &t,
                                 const bdd_function &e) const noexcept {
    return capi::oxidd_bdd_ite(_func, t._func, e._func);
  }

  /// Compute the BDD for the universal quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// universal quantification. Universal quantification of a Boolean function
  /// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function forall(const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_forall(_func, vars._func);
  }
  /// Compute the BDD for the existential quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// existential quantification. Existential quantification of a Boolean
  /// function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ∨ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function exist(const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_exist(_func, vars._func);
  }
  /// Compute the BDD for the unique quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// unique quantification. Unique quantification of a Boolean function
  /// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ⊕ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function unique(const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_unique(_func, vars._func);
  }

  /// @}
  /// @name Query Operations on BDDs
  /// @{

  /// Count descendant nodes
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  Node count including the two terminal nodes
  [[nodiscard]] uint64_t node_count() const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_node_count(_func);
  }

  /// Check for satisfiability
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  `true` iff there is a satisfying assignment
  [[nodiscard]] bool satisfiable() const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_satisfiable(_func);
  }

  /// Check for validity
  ///
  /// `this` must not be invalid (in the technical, not the mathematical sense).
  /// Check via is_invalid().
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  `true` iff there is are only satisfying assignment
  [[nodiscard]] bool valid() const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_valid(_func);
  }

  /// Count the number of satisfying assignments
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  Count of satisfying assignments
  [[nodiscard]] double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_sat_count_double(_func, vars);
  }

  /// Pick a satisfying assignment
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  A satisfying assignment if there exists one. If this function is
  ///           unsatisfiable, the assignment is empty.
  [[nodiscard]] util::assignment pick_cube() const noexcept {
    assert(_func._p);
    return util::assignment(capi::oxidd_bdd_pick_cube(_func));
  }

  /// Evaluate this Boolean function with arguments `args`
  ///
  /// `args` determines the valuation for all variables. Missing values are
  /// assumed to be false. The order is irrelevant. All elements must point to
  /// inner nodes.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @param  args  Slice of pairs `(variable, value)`, where all variables are
  ///               not invalid
  ///
  /// @returns  Result of the evaluation with `args`
  [[nodiscard]] bool
  eval(util::slice<std::pair<bdd_function, bool>> args) const noexcept {
    assert(_func._p);

    // From a C++ perspective, it is nicer to have elements of type
    // `std::pair<bdd_function, bool>` than `capi::oxidd_bdd_bool_pair_t`. In
    // the following we ensure that their layouts are compatible such that the
    // pointer cast below is safe.
    using c_pair = capi::oxidd_bdd_bool_pair_t;
    using cpp_pair = std::pair<bdd_function, bool>;
    static_assert(std::is_standard_layout_v<cpp_pair>);
    static_assert(sizeof(cpp_pair) == sizeof(c_pair));
    static_assert(offsetof(cpp_pair, first) == offsetof(c_pair, func));
    static_assert(offsetof(cpp_pair, second) == offsetof(c_pair, val));
    static_assert(alignof(cpp_pair) == alignof(c_pair));

    return capi::oxidd_bdd_eval(
        _func,
        reinterpret_cast<const c_pair *>(args.data()), // NOLINT(*-cast)
        args.size());
  }

  /// @}
};

inline bdd_function bdd_manager::new_var() noexcept {
  assert(_manager._p);
  return capi::oxidd_bdd_new_var(_manager);
}
inline bdd_function bdd_manager::t() const noexcept {
  assert(_manager._p);
  return capi::oxidd_bdd_true(_manager);
}
inline bdd_function bdd_manager::f() const noexcept {
  assert(_manager._p);
  return capi::oxidd_bdd_false(_manager);
}

} // namespace oxidd

/// @cond

/// Partial specialization for oxidd::bdd_function
template <> struct std::hash<oxidd::bdd_function> {
  std::size_t operator()(const oxidd::bdd_function &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

/// @endcond
