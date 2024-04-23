/// @file  bcdd.hpp
/// @brief Complement edge binary decision diagrams

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

class bcdd_function;

/// Manager for binary decision diagrams with complement edges
///
/// Models the oxidd::boolean_function_manager concept.
///
/// Instances can safely be sent to other threads.
class bcdd_manager {
  /// Wrapped CAPI BCDD manager
  capi::oxidd_bcdd_manager_t _manager = {._p = nullptr};

  friend class bcdd_function;

  /// Create a new BCDD manager from a manager instance of the CAPI
  bcdd_manager(capi::oxidd_bcdd_manager_t manager) noexcept
      : _manager(manager) {}

public:
  /// Associated function type
  using function = bcdd_function;

  /// Default constructor, yields an invalid manager
  bcdd_manager() noexcept = default;
  /// Create a new BCDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  bcdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept
      : _manager(capi::oxidd_bcdd_manager_new(inner_node_capacity,
                                              apply_cache_capacity, threads)) {}
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Runtime complexity: O(1)
  bcdd_manager(const bcdd_manager &other) noexcept : _manager(other._manager) {
    capi::oxidd_bcdd_manager_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  bcdd_manager(bcdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~bcdd_manager() noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bcdd_manager_unref(_manager);
  }

  /// Copy assignment operator
  bcdd_manager &operator=(const bcdd_manager &rhs) noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bcdd_manager_unref(_manager);
    _manager = rhs._manager;
    capi::oxidd_bcdd_manager_ref(_manager);
    return *this;
  }
  /// Move assignment operator
  bcdd_manager &operator=(bcdd_manager &&rhs) noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bcdd_manager_unref(_manager);
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
  friend bool operator==(const bcdd_manager &lhs,
                         const bcdd_manager &rhs) noexcept {
    return lhs._manager._p == rhs._manager._p;
  }
  /// Same as `!(lhs == rhs)` (see @ref operator==)
  friend bool operator!=(const bcdd_manager &lhs,
                         const bcdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor bcdd_manager() is
  /// invalid as well as a @ref bcdd_manager instance that has been moved (via
  /// bcdd_manager(bcdd_manager &&other)).
  ///
  /// @returns  `true` iff this manager reference is invalid
  [[nodiscard]] bool is_invalid() const noexcept {
    return _manager._p == nullptr;
  }

  /// @name BCDD Construction
  /// @{

  /// Get a fresh variable, i.e., a function that is true if and only if the
  /// variable is true. This adds a new level to the decision diagram.
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @returns  The BCDD function representing the variable
  [[nodiscard]] bcdd_function new_var() noexcept;

  /// Get the constant true BCDD function ⊤
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊤ as BCDD function
  [[nodiscard]] bcdd_function t() const noexcept;
  /// Get the constant false BCDD function ⊥
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊥ as BCDD function
  [[nodiscard]] bcdd_function f() const noexcept;

  /// @}
  /// @name Statistics
  /// @{

  /// Get the number of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The number of inner nodes
  [[nodiscard]] size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return capi::oxidd_bcdd_num_inner_nodes(_manager);
  }

  /// @}
};

/// Boolean function represented as a binary decision diagram with complement
/// edges (BCDD)
///
/// This is essentially a tagged reference to a BCDD node.
///
/// Models the oxidd::boolean_function_quant concept.
///
/// Instances can safely be sent to other threads.
class bcdd_function {
  /// Wrapped BCDD function
  capi::oxidd_bcdd_t _func = {._p = nullptr};

  friend class bcdd_manager;
  friend struct std::hash<bcdd_function>;

  /// Create a new @ref bcdd_function from a BCDD function instance of the CAPI
  bcdd_function(capi::oxidd_bcdd_t func) noexcept : _func(func) {}

public:
  /// Associated manager type
  using manager = bcdd_manager;

  /// Default constructor, yields an invalid BCDD function
  bcdd_function() noexcept = default;
  /// Copy constructor: increments the internal reference counters
  ///
  /// Runtime complexity: O(1)
  bcdd_function(const bcdd_function &other) noexcept : _func(other._func) {
    capi::oxidd_bcdd_ref(_func);
  }
  /// Move constructor: invalidates `other`
  bcdd_function(bcdd_function &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~bcdd_function() noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bcdd_unref(_func);
  }

  /// Copy assignment operator
  bcdd_function &operator=(const bcdd_function &rhs) noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bcdd_unref(_func);
    _func = rhs._func;
    if (_func._p != nullptr)
      capi::oxidd_bcdd_ref(_func);
    return *this;
  }
  /// Move assignment operator
  bcdd_function &operator=(bcdd_function &&rhs) noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bcdd_unref(_func);
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
  /// @returns  `true` iff `lhs` and `rhs` reference the same node and have the
  ///           same edge tag
  friend bool operator==(const bcdd_function &lhs,
                         const bcdd_function &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  /// Same as `!(lhs == rhs)` (see @ref operator==)
  friend bool operator!=(const bcdd_function &lhs,
                         const bcdd_function &rhs) noexcept {
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
  friend bool operator<(const bcdd_function &lhs,
                        const bcdd_function &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  /// @ref operator< with arguments swapped
  friend bool operator>(const bcdd_function &lhs,
                        const bcdd_function &rhs) noexcept {
    return rhs < lhs;
  }
  /// Same as `!(rhs < lhs)` (see @ref operator<)
  friend bool operator<=(const bcdd_function &lhs,
                         const bcdd_function &rhs) noexcept {
    return !(rhs < lhs);
  }
  /// Same as `!(lhs < rhs)` (see @ref operator<)
  friend bool operator>=(const bcdd_function &lhs,
                         const bcdd_function &rhs) noexcept {
    return !(lhs < rhs);
  }

  /// Check if this BCDD function is invalid
  ///
  /// A BCDD function created by the default constructor bcdd_function() is
  /// invalid as well as a @ref bcdd_function instance that has been moved
  /// (via bcdd_function(bcdd_function &&other)). Moreover, if an operation
  /// tries to allocate new nodes but runs out of memory, then it returns an
  /// invalid function.
  ///
  /// @returns  `true` iff this BCDD function is invalid
  [[nodiscard]] bool is_invalid() const noexcept { return _func._p == nullptr; }

  /// Get the containing manager
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// @returns  The bcdd_manager
  [[nodiscard]] bcdd_manager containing_manager() const noexcept {
    assert(!is_invalid());
    return capi::oxidd_bcdd_containing_manager(_func);
  }

  /// @name BCDD Construction Operations
  /// @{

  /// Get the cofactors `(f_true, f_false)` of `f`
  ///
  /// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the
  /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
  /// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
  ///
  /// Structurally, the cofactors are children with edge tags are adjusted
  /// accordingly. If you only need one of the cofactors, then use
  /// cofactor_true() or cofactor_false(). These functions are slightly more
  /// efficient then.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The pair `{f_true, f_false}` if `f` is valid and references an
  ///           inner node, otherwise a pair of invalid functions.
  [[nodiscard]] std::pair<bcdd_function, bcdd_function>
  cofactors() const noexcept {
    const capi::oxidd_bcdd_pair_t p = capi::oxidd_bcdd_cofactors(_func);
    return {p.first, p.second};
  }
  /// Get the cofactor `f_true` of `f`
  ///
  /// This function is slightly more efficient than cofactors() in case
  /// `f_false` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_true` if `f` is valid and references an inner node, otherwise
  ///           an invalid function.
  [[nodiscard]] bcdd_function cofactor_true() const noexcept {
    return capi::oxidd_bcdd_cofactor_true(_func);
  }
  /// Get the cofactor `f_false` of `f`
  ///
  /// This function is slightly more efficient than cofactors() in case `f_true`
  /// is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_false` if `f` is valid and references an inner node,
  ///           otherwise an invalid function.
  [[nodiscard]] bcdd_function cofactor_false() const noexcept {
    return capi::oxidd_bcdd_cofactor_true(_func);
  }

  /// Compute the BCDD for the negation `¬this`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The BCDD function
  [[nodiscard]] bcdd_function operator~() const noexcept {
    return capi::oxidd_bcdd_not(_func);
  }
  /// Compute the BCDD for the conjunction `lhs ∧ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend bcdd_function
  operator&(const bcdd_function &lhs, const bcdd_function &rhs) noexcept {
    return capi::oxidd_bcdd_and(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator&
  bcdd_function &operator&=(const bcdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  /// Compute the BCDD for the disjunction `lhs ∨ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend bcdd_function
  operator|(const bcdd_function &lhs, const bcdd_function &rhs) noexcept {
    return capi::oxidd_bcdd_or(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator|
  bcdd_function &operator|=(const bcdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  /// Compute the BCDD for the exclusive disjunction `lhs ⊕ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend bcdd_function
  operator^(const bcdd_function &lhs, const bcdd_function &rhs) noexcept {
    return capi::oxidd_bcdd_xor(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator^
  bcdd_function &operator^=(const bcdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  /// Compute the BCDD for the negated conjunction `this ⊼ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function nand(const bcdd_function &rhs) const noexcept {
    return capi::oxidd_bcdd_nand(_func, rhs._func);
  }
  /// Compute the BCDD for the negated disjunction `this ⊽ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function nor(const bcdd_function &rhs) const noexcept {
    return capi::oxidd_bcdd_nor(_func, rhs._func);
  }
  /// Compute the BCDD for the equivalence `this ↔ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function equiv(const bcdd_function &rhs) const noexcept {
    return capi::oxidd_bcdd_equiv(_func, rhs._func);
  }
  /// Compute the BCDD for the implication `this → rhs` (or `this ≤ rhs`)
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function imp(const bcdd_function &rhs) const noexcept {
    return capi::oxidd_bcdd_imp(_func, rhs._func);
  }
  /// Compute the BCDD for the strict implication `this < rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function
  imp_strict(const bcdd_function &rhs) const noexcept {
    return capi::oxidd_bcdd_imp_strict(_func, rhs._func);
  }
  /// Compute the BCDD for the conditional `this ? t : e`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |t| · |e|)
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function ite(const bcdd_function &t,
                                  const bcdd_function &e) const noexcept {
    return capi::oxidd_bcdd_ite(_func, t._func, e._func);
  }

  /// Compute the BCDD for the universal quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// universal quantification. Universal quantification of a Boolean function
  /// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ∧ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function forall(const bcdd_function &vars) const noexcept {
    return capi::oxidd_bcdd_forall(_func, vars._func);
  }
  /// Compute the BCDD for the existential quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// existential quantification. Existential quantification of a Boolean
  /// function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ∨ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function exist(const bcdd_function &vars) const noexcept {
    return capi::oxidd_bcdd_exist(_func, vars._func);
  }
  /// Compute the BCDD for the unique quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// unique quantification. Unique quantification of a Boolean function
  /// `f(…, x, …)` over a single variable `x` is `f(…, 0, …) ⊕ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BCDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bcdd_function unique(const bcdd_function &vars) const noexcept {
    return capi::oxidd_bcdd_unique(_func, vars._func);
  }

  /// @}
  /// @name Query Operations on BCDDs
  /// @{

  /// Count descendant nodes
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  Node count including the terminal node
  [[nodiscard]] std::size_t node_count() const noexcept {
    assert(_func._p);
    return capi::oxidd_bcdd_node_count(_func);
  }

  /// Check for satisfiability
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is a satisfying assignment
  [[nodiscard]] bool satisfiable() const noexcept {
    assert(_func._p);
    return capi::oxidd_bcdd_satisfiable(_func);
  }

  /// Check for validity
  ///
  /// `this` must not be invalid (in the technical, not the mathematical sense).
  /// Check via is_invalid().
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is are only satisfying assignment
  [[nodiscard]] bool valid() const noexcept {
    assert(_func._p);
    return capi::oxidd_bcdd_valid(_func);
  }

  /// Count the number of satisfying assignments
  ///
  /// This method assumes that the function's domain of has `vars` many
  /// variables.
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  vars  Number of variables in the function's domain
  ///
  /// @returns  Count of satisfying assignments
  [[nodiscard]] double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return capi::oxidd_bcdd_sat_count_double(_func, vars);
  }

  /// Pick a satisfying assignment
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. If this function is
  ///           unsatisfiable, the assignment is empty.
  [[nodiscard]] util::assignment pick_cube() const noexcept {
    assert(_func._p);
    return util::assignment(capi::oxidd_bcdd_pick_cube(_func));
  }

  /// Evaluate this Boolean function with arguments `args`
  ///
  /// `args` determines the valuation for all variables. Missing values are
  /// assumed to be false. The order is irrelevant. All elements must point to
  /// inner nodes.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  args  Slice of pairs `(variable, value)`, where all variables are
  ///               not invalid
  ///
  /// @returns  Result of the evaluation with `args`
  [[nodiscard]] bool
  eval(util::slice<std::pair<bcdd_function, bool>> args) const noexcept {
    assert(_func._p);

    // From a C++ perspective, it is nicer to have elements of type
    // `std::pair<bcdd_function, bool>` than `capi::oxidd_bcdd_bool_pair_t`. In
    // the following we ensure that their layouts are compatible such that the
    // pointer cast below is safe.
    using c_pair = capi::oxidd_bcdd_bool_pair_t;
    using cpp_pair = std::pair<bcdd_function, bool>;
    static_assert(std::is_standard_layout_v<cpp_pair>);
    static_assert(sizeof(cpp_pair) == sizeof(c_pair));
    static_assert(offsetof(cpp_pair, first) == offsetof(c_pair, func));
    static_assert(offsetof(cpp_pair, second) == offsetof(c_pair, val));
    static_assert(alignof(cpp_pair) == alignof(c_pair));

    return capi::oxidd_bcdd_eval(
        _func,
        reinterpret_cast<const c_pair *>(args.data()), // NOLINT(*-cast)
        args.size());
  }

  /// @}
};

inline bcdd_function bcdd_manager::new_var() noexcept {
  assert(_manager._p);
  return capi::oxidd_bcdd_new_var(_manager);
}
inline bcdd_function bcdd_manager::t() const noexcept {
  assert(_manager._p);
  return capi::oxidd_bcdd_true(_manager);
}
inline bcdd_function bcdd_manager::f() const noexcept {
  assert(_manager._p);
  return capi::oxidd_bcdd_false(_manager);
}

} // namespace oxidd

/// @cond

/// Partial specialization for oxidd::bcdd_function
template <> struct std::hash<oxidd::bcdd_function> {
  std::size_t operator()(const oxidd::bcdd_function &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

/// @endcond
