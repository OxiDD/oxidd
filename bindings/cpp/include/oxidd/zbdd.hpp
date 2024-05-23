/// @file  zbdd.hpp
/// @brief Zero-suppressed binary decision diagrams

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

class zbdd_function;

/// Manager for zero-suppressed binary decision diagrams
///
/// Instances can safely be sent to other threads.
///
/// Models `oxidd::concepts::boolean_function_manager`
class zbdd_manager {
  /// Wrapped CAPI ZBDD manager
  capi::oxidd_zbdd_manager_t _manager = {._p = nullptr};

  friend class zbdd_function;

  /// Create a new ZBDD manager from a manager instance of the CAPI
  zbdd_manager(capi::oxidd_zbdd_manager_t manager) noexcept
      : _manager(manager) {}

public:
  /// Associated function type
  using function = zbdd_function;

  /// Default constructor, yields an invalid manager
  zbdd_manager() noexcept = default;
  /// Create a new BCDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  zbdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept
      : _manager(capi::oxidd_zbdd_manager_new(inner_node_capacity,
                                              apply_cache_capacity, threads)) {}
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Runtime complexity: O(1)
  zbdd_manager(const zbdd_manager &other) noexcept : _manager(other._manager) {
    capi::oxidd_zbdd_manager_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  zbdd_manager(zbdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~zbdd_manager() noexcept { capi::oxidd_zbdd_manager_unref(_manager); }

  /// Copy assignment operator
  zbdd_manager &operator=(const zbdd_manager &rhs) noexcept {
    if (this != &rhs) {
      capi::oxidd_zbdd_manager_unref(_manager);
      _manager = rhs._manager;
      capi::oxidd_zbdd_manager_ref(_manager);
    }
    return *this;
  }
  /// Move assignment operator: invalidates `rhs`
  zbdd_manager &operator=(zbdd_manager &&rhs) noexcept {
    assert(this != &rhs || !rhs._manager._p);
    capi::oxidd_zbdd_manager_unref(_manager);
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
  friend bool operator==(const zbdd_manager &lhs,
                         const zbdd_manager &rhs) noexcept {
    return lhs._manager._p == rhs._manager._p;
  }
  /// Same as `!(lhs == rhs)` (see `operator==()`)
  friend bool operator!=(const zbdd_manager &lhs,
                         const zbdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor `zbdd_manager()` is
  /// invalid as well as a `zbdd_manager` instance that has been moved (via
  /// `zbdd_manager(zbdd_manager &&other)`).
  ///
  /// @returns  `true` iff this manager reference is invalid
  [[nodiscard]] bool is_invalid() const noexcept {
    return _manager._p == nullptr;
  }

  /// @name ZBDD Construction
  /// @{

  /// Get a fresh variable in the form of a singleton set. This adds a new level
  /// to a decision diagram.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @returns  The ZBDD Boolean function representing the variable.
  [[nodiscard]] zbdd_function new_singleton() noexcept;
  /// Get a fresh variable, i.e., a function that is true if and only if the
  /// variable is true. This adds a new level to the decision diagram.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @returns  The ZBDD set representing the variable
  [[nodiscard]] zbdd_function new_var() noexcept;

  /// Get the ZBDD set ∅
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  ∅ as ZBDD set
  [[nodiscard]] zbdd_function empty() const noexcept;
  /// Get the ZBDD set {∅}
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  {∅} as ZBDD set
  [[nodiscard]] zbdd_function base() const noexcept;
  /// Get the constant true ZBDD Boolean function ⊤
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊤ as ZBDD Boolean function
  [[nodiscard]] zbdd_function t() const noexcept;
  /// Get the constant false ZBDD Boolean function ⊥
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊥ as ZBDD Boolean function
  [[nodiscard]] zbdd_function f() const noexcept;

  /// @}
  /// @name Statistics
  /// @{

  /// Get the number of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The number of inner nodes
  [[nodiscard]] size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return capi::oxidd_zbdd_num_inner_nodes(_manager);
  }

  /// @}
};

/// Boolean function 𝔹ⁿ → 𝔹 (or set of Boolean vectors 𝔹ⁿ) represented as
/// zero-suppressed binary decision diagram
///
/// Models `oxidd::concepts::boolean_function`
class zbdd_function {
  /// Wrapped ZBDD function
  capi::oxidd_zbdd_t _func = {._p = nullptr};

  friend class zbdd_manager;
  friend struct std::hash<zbdd_function>;

  /// Create a new `zbdd_function` from a ZBDD function instance of the CAPI
  zbdd_function(capi::oxidd_zbdd_t func) noexcept : _func(func) {}

public:
  /// Associated manager type
  using manager = zbdd_manager;

  /// Default constructor, yields an invalid ZBDD function
  zbdd_function() noexcept = default;
  /// Copy constructor: increments the internal reference counters
  ///
  /// Runtime complexity: O(1)
  zbdd_function(const zbdd_function &other) noexcept : _func(other._func) {
    capi::oxidd_zbdd_ref(_func);
  }
  /// Move constructor: invalidates `other`
  zbdd_function(zbdd_function &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~zbdd_function() noexcept { capi::oxidd_zbdd_unref(_func); }

  /// Copy assignment operator
  zbdd_function &operator=(const zbdd_function &rhs) noexcept {
    if (this != &rhs) {
      capi::oxidd_zbdd_unref(_func);
      _func = rhs._func;
      capi::oxidd_zbdd_ref(_func);
    }
    return *this;
  }
  /// Move assignment operator: invalidates `rhs`
  zbdd_function &operator=(zbdd_function &&rhs) noexcept {
    assert(this != &rhs || !rhs._func._p);
    capi::oxidd_zbdd_unref(_func);
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
  friend bool operator==(const zbdd_function &lhs,
                         const zbdd_function &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  /// Same as `!(lhs == rhs)` (see `operator==()`)
  friend bool operator!=(const zbdd_function &lhs,
                         const zbdd_function &rhs) noexcept {
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
  friend bool operator<(const zbdd_function &lhs,
                        const zbdd_function &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  /// `operator<()` with arguments swapped
  friend bool operator>(const zbdd_function &lhs,
                        const zbdd_function &rhs) noexcept {
    return rhs < lhs;
  }
  /// Same as `!(rhs < lhs)` (see `operator<()`)
  friend bool operator<=(const zbdd_function &lhs,
                         const zbdd_function &rhs) noexcept {
    return !(rhs < lhs);
  }
  /// Same as `!(lhs < rhs)` (see `operator<()`)
  friend bool operator>=(const zbdd_function &lhs,
                         const zbdd_function &rhs) noexcept {
    return !(lhs < rhs);
  }

  /// Check if this ZBDD function is invalid
  ///
  /// A ZBDD function created by the default constructor `zbdd_function()` is
  /// invalid as well as a `zbdd_function` instance that has been moved (via
  /// `zbdd_function(zbdd_function &&other)`). Moreover, if an operation tries
  /// to allocate new nodes but runs out of memory, then it returns an invalid
  /// function.
  ///
  /// @returns  `true` iff this ZBDD function is invalid
  [[nodiscard]] bool is_invalid() const noexcept { return _func._p == nullptr; }

  /// Get the containing manager
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// @returns  The bdd_manager
  [[nodiscard]] zbdd_manager containing_manager() const noexcept {
    assert(!is_invalid());
    return capi::oxidd_zbdd_containing_manager(_func);
  }

  /// @name ZBDD Construction Operations
  /// @{

  /// Get the cofactors `(f_true, f_false)` of `f`
  ///
  /// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the
  /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
  /// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
  ///
  /// Structurally, the cofactors are the children. If you only need one of the
  /// cofactors, then use `cofactor_true()` or `cofactor_false()`. These
  /// functions are slightly more efficient then.
  ///
  /// Note that the domain of f is 𝔹<sup>n+1</sup> while the domain of
  /// f<sub>true</sub> and f<sub>false</sub> is 𝔹<sup>n</sup>. (Remember that,
  /// e.g., g(x₀) = x₀ and g'(x₀, x₁) = x₀ have different representations as
  /// ZBDDs.)
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The pair `{f_true, f_false}` if `f` is valid and references an
  ///           inner node, otherwise a pair of invalid functions.
  [[nodiscard]] std::pair<zbdd_function, zbdd_function>
  cofactors() const noexcept {
    const capi::oxidd_zbdd_pair_t p = capi::oxidd_zbdd_cofactors(_func);
    return {p.first, p.second};
  }
  /// Get the cofactor `f_true` of `f`
  ///
  /// This function is slightly more efficient than `cofactors()` in case
  /// `f_false` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_true` if `f` is valid and references an inner node, otherwise
  ///           an invalid function.
  [[nodiscard]] zbdd_function cofactor_true() const noexcept {
    return capi::oxidd_zbdd_cofactor_true(_func);
  }
  /// Get the cofactor `f_false` of `f`
  ///
  /// This function is slightly more efficient than `cofactors()` in case
  /// `f_true` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_false` if `f` is valid and references an inner node,
  ///           otherwise an invalid function.
  [[nodiscard]] zbdd_function cofactor_false() const noexcept {
    return capi::oxidd_zbdd_cofactor_true(_func);
  }

  /// Get the ZBDD Boolean function v for the singleton set {v}
  ///
  /// `this` must be a singleton set.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The ZBDD Boolean function v
  [[nodiscard]] zbdd_function var_boolean_function() const noexcept {
    return capi::oxidd_zbdd_var_boolean_function(_func);
  }

  /// Compute the ZBDD for the negation `¬this`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function operator~() const noexcept {
    return capi::oxidd_zbdd_not(_func);
  }
  /// Compute the ZBDD for the conjunction `lhs ∧ rhs`
  ///
  /// The conjunction on Boolean functions may equivalently be viewed as an
  /// intersection of sets `lhs ∩ rhs`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend zbdd_function
  operator&(const zbdd_function &lhs, const zbdd_function &rhs) noexcept {
    return capi::oxidd_zbdd_intsec(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator&
  zbdd_function &operator&=(const zbdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  /// Compute the ZBDD for the disjunction `lhs ∨ rhs`
  ///
  /// The disjunction on Boolean functions may equivalently be viewed as a
  /// union of sets `lhs ∪ rhs`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend zbdd_function
  operator|(const zbdd_function &lhs, const zbdd_function &rhs) noexcept {
    return capi::oxidd_zbdd_union(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator|
  zbdd_function &operator|=(const zbdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  /// Compute the ZBDD for the exclusive disjunction `lhs ⊕ rhs`
  ///
  /// The disjunction on Boolean functions may equivalently be viewed as a
  /// symmetric difference on sets.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend zbdd_function
  operator^(const zbdd_function &lhs, const zbdd_function &rhs) noexcept {
    return capi::oxidd_zbdd_xor(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator^
  zbdd_function &operator^=(const zbdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  /// Compute the ZBDD for the set difference `lhs ∖ rhs`
  ///
  /// This is equivalent to the strict implication `rhs < lhs` on Boolean
  /// functions.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD set/function (may be invalid if the operation runs out
  ///           of memory)
  [[nodiscard]] friend zbdd_function
  operator-(const zbdd_function &lhs, const zbdd_function &rhs) noexcept {
    return capi::oxidd_zbdd_diff(lhs._func, rhs._func);
  }
  /// Assignment version of @ref operator-
  [[nodiscard]] zbdd_function &operator-=(const zbdd_function &rhs) noexcept {
    return (*this = *this - rhs);
  }
  /// Compute the ZBDD for the negated conjunction `this ⊼ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function nand(const zbdd_function &rhs) const noexcept {
    return capi::oxidd_zbdd_nand(_func, rhs._func);
  }
  /// Compute the ZBDD for the negated disjunction `this ⊽ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function nor(const zbdd_function &rhs) const noexcept {
    return capi::oxidd_zbdd_nor(_func, rhs._func);
  }
  /// Compute the ZBDD for the equivalence `this ↔ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function equiv(const zbdd_function &rhs) const noexcept {
    return capi::oxidd_zbdd_equiv(_func, rhs._func);
  }
  /// Compute the ZBDD for the implication `this → rhs` (or `this ≤ rhs`)
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function imp(const zbdd_function &rhs) const noexcept {
    return capi::oxidd_zbdd_imp(_func, rhs._func);
  }
  /// Compute the ZBDD for the strict implication `this < rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function
  imp_strict(const zbdd_function &rhs) const noexcept {
    return capi::oxidd_zbdd_imp_strict(_func, rhs._func);
  }
  /// Compute the ZBDD for the conditional `this ? t : e`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |t| · |e|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] zbdd_function ite(const zbdd_function &t,
                                  const zbdd_function &e) const noexcept {
    return capi::oxidd_zbdd_ite(_func, t._func, e._func);
  }

  /// Create a new ZBDD node at the level of `this` with the given `hi` and `lo`
  /// edges
  ///
  /// `var` must be a singleton set.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The ZBDD set representing referencing the new node (may be
  ///           invalid if the operation runs out of memory)
  // NOLINTBEGIN(*-moved)
  [[nodiscard]] zbdd_function make_node(zbdd_function &&hi,
                                        zbdd_function &&lo) const noexcept {
    const capi::oxidd_zbdd_t h = hi._func, l = lo._func;
    hi._func._p = nullptr;
    lo._func._p = nullptr;
    return capi::oxidd_zbdd_make_node(_func, h, l);
  }
  // NOLINTEND(*-moved)

  /// @}
  /// @name Query Operations on ZBDDs
  /// @{

  /// Count descendant nodes
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  Node count including the two terminal nodes
  [[nodiscard]] std::size_t node_count() const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_node_count(_func);
  }

  /// Check for satisfiability
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is a satisfying assignment
  [[nodiscard]] bool satisfiable() const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_satisfiable(_func);
  }

  /// Check for validity
  ///
  /// `this` must not be invalid (in the technical, not the mathematical sense).
  /// Check via `is_invalid()`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is are only satisfying assignment
  [[nodiscard]] bool valid() const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_valid(_func);
  }

  /// Count the number of satisfying assignments
  ///
  /// This method assumes that the function's domain of has `vars` many
  /// variables.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  vars  Number of variables in the function's domain
  ///
  /// @returns  Count of satisfying assignments
  [[nodiscard]] double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_sat_count_double(_func, vars);
  }

  /// Pick a satisfying assignment
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. If this function is
  ///           unsatisfiable, the assignment is empty.
  [[nodiscard]] util::assignment pick_cube() const noexcept {
    assert(_func._p);
    return util::assignment(capi::oxidd_zbdd_pick_cube(_func));
  }

  /// Evaluate this Boolean function with arguments `args`
  ///
  /// `args` determines the valuation for all variables. Missing values are
  /// assumed to be false. The order is irrelevant. All elements must point to
  /// inner nodes. Note that the domain of `f` is treated somewhat implicitly,
  /// it contains at least all `args` and all variables in the support of the
  /// ZBDD. Unlike BDDs, extending the domain changes the semantics of ZBDDs.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  args  Slice of pairs `(variable, value)`, where all variables are
  ///               not invalid
  ///
  /// @returns  Result of the evaluation with `args`
  [[nodiscard]] bool
  eval(util::slice<std::pair<zbdd_function, bool>> args) const noexcept {
    assert(_func._p);

    // From a C++ perspective, it is nicer to have elements of type
    // `std::pair<bdd_function, bool>` than `capi::oxidd_bdd_bool_pair_t`. In
    // the following we ensure that their layouts are compatible such that the
    // pointer cast below is safe.
    using c_pair = capi::oxidd_zbdd_bool_pair_t;
    using cpp_pair = std::pair<zbdd_function, bool>;
    static_assert(std::is_standard_layout_v<cpp_pair>);
    static_assert(sizeof(cpp_pair) == sizeof(c_pair));
    static_assert(offsetof(cpp_pair, first) == offsetof(c_pair, func));
    static_assert(offsetof(cpp_pair, second) == offsetof(c_pair, val));
    static_assert(alignof(cpp_pair) == alignof(c_pair));

    return capi::oxidd_zbdd_eval(
        _func,
        reinterpret_cast<const c_pair *>(args.data()), // NOLINT(*-cast)
        args.size());
  }

  /// @}
};

inline zbdd_function zbdd_manager::new_singleton() noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_new_singleton(_manager);
}
inline zbdd_function zbdd_manager::new_var() noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_new_var(_manager);
}
inline zbdd_function zbdd_manager::empty() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_empty(_manager);
}
inline zbdd_function zbdd_manager::base() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_base(_manager);
}
inline zbdd_function zbdd_manager::t() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_true(_manager);
}
inline zbdd_function zbdd_manager::f() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_false(_manager);
}

} // namespace oxidd

/// @cond

/// Partial specialization for `oxidd::zbdd_function`
template <> struct std::hash<oxidd::zbdd_function> {
  std::size_t operator()(const oxidd::zbdd_function &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

/// @endcond
