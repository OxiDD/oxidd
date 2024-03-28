/// @file  zbdd.hpp
/// @brief Zero-suppressed binary decision diagrams

#pragma once

#include <oxidd/utils.hpp>

namespace oxidd {

class zbdd_set;

/// Manager for zero-suppressed binary decision diagrams
///
/// Models the oxidd::boolean_function_manager concept.
///
/// Instances can safely be sent to other threads.
class zbdd_manager {
  /// Wrapped CAPI ZBDD manager
  capi::oxidd_zbdd_manager_t _manager;

  /// Create a new ZBDD manager from a manager instance of the CAPI
  zbdd_manager(capi::oxidd_zbdd_manager_t manager) noexcept
      : _manager(manager) {}

public:
  /// Associated function/set type
  using function = zbdd_set;

  /// Default constructor, yields an invalid manager
  zbdd_manager() noexcept { _manager._p = nullptr; }
  /// Create a new BCDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  zbdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept {
    _manager = capi::oxidd_zbdd_manager_new(inner_node_capacity,
                                            apply_cache_capacity, threads);
  }
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Runtime complexity: O(1)
  zbdd_manager(const zbdd_manager &other) noexcept : _manager(other._manager) {
    oxidd_zbdd_manager_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  zbdd_manager(zbdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~zbdd_manager() noexcept {
    if (_manager._p != nullptr)
      oxidd_zbdd_manager_unref(_manager);
  }

  /// Copy assignment operator
  zbdd_manager &operator=(const zbdd_manager &rhs) noexcept {
    if (_manager._p != nullptr)
      oxidd_zbdd_manager_unref(_manager);
    _manager = rhs._manager;
    oxidd_zbdd_manager_ref(_manager);
    return *this;
  }
  /// Move assignment operator
  zbdd_manager &operator=(zbdd_manager &&rhs) noexcept {
    if (_manager._p != nullptr)
      oxidd_zbdd_manager_unref(_manager);
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
  /// Same as `!(lhs == rhs)` (see \ref operator==)
  friend bool operator!=(const zbdd_manager &lhs,
                         const zbdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor zbdd_manager() is
  /// invalid as well as a \ref zbdd_manager instance that has been moved (via
  /// zbdd_manager(zbdd_manager &&other)).
  ///
  /// @returns  `true` iff this manager reference is invalid
  bool is_invalid() const noexcept { return !_manager._p; }

  /// Get a fresh variable in the form of a singleton set. This adds a new level
  /// to a decision diagram.
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires an exclusive manager lock.
  ///
  /// @returns  The ZBDD Boolean function representing the variable.
  zbdd_set new_singleton() noexcept;
  /// Get a fresh variable, i.e., a function that is true if and only if the
  /// variable is true. This adds a new level to the decision diagram.
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires an exclusive manager lock.
  ///
  /// @returns  The ZBDD set representing the variable
  zbdd_set new_var() noexcept;

  /// Get the ZBDD set ∅
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  ∅ as ZBDD set
  zbdd_set empty() const noexcept;
  /// Get the ZBDD set \{∅\}
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  \{∅\} as ZBDD set
  zbdd_set base() const noexcept;
  /// Get the constant true ZBDD Boolean function ⊤
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊤ as ZBDD Boolean function
  zbdd_set top() const noexcept;
  /// Get the constant false ZBDD Boolean function ⊥
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊥ as ZBDD Boolean function
  zbdd_set bot() const noexcept;

  /// Get the number of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The number of inner nodes
  size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return oxidd_zbdd_num_inner_nodes(_manager);
  }
};

/// Set of Boolean vectors 𝔹ⁿ represented as zero-suppressed binary decision
/// diagram
class zbdd_set {
  /// Wrapped ZBDD set
  capi::oxidd_zbdd_t _func;

  friend class zbdd_manager;
  friend struct std::hash<zbdd_set>;

  /// Create a new \ref zbdd_set from a ZBDD set instance of the CAPI
  zbdd_set(capi::oxidd_zbdd_t func) noexcept : _func(func) {}

public:
  /// Default constructor, yields an invalid ZBDD set
  zbdd_set() noexcept { _func._p = nullptr; }
  /// Copy constructor: increments the internal reference counters
  ///
  /// Runtime complexity: O(1)
  zbdd_set(const zbdd_set &other) noexcept : _func(other._func) {
    oxidd_zbdd_ref(_func);
  }
  /// Move constructor: invalidates `other`
  zbdd_set(zbdd_set &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~zbdd_set() noexcept {
    if (_func._p != nullptr)
      oxidd_zbdd_unref(_func);
  }

  /// Copy assignment operator
  zbdd_set &operator=(const zbdd_set &rhs) noexcept {
    if (_func._p != nullptr)
      oxidd_zbdd_unref(_func);
    _func = rhs._func;
    if (_func._p != nullptr)
      oxidd_zbdd_ref(_func);
    return *this;
  }
  /// Move assignment operator
  zbdd_set &operator=(zbdd_set &&rhs) noexcept {
    if (_func._p != nullptr)
      oxidd_zbdd_unref(_func);
    _func = rhs._func;
    rhs._func._p = nullptr;
    return *this;
  }

  /// Compare two sets for referential equality
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same node
  friend bool operator==(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  /// Same as `!(lhs == rhs)` (see \ref operator==)
  friend bool operator!=(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
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
  friend bool operator<(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  /// \ref operator< with arguments swapped
  friend bool operator>(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return rhs < lhs;
  }
  /// Same as `!(rhs < lhs)` (see \ref operator<)
  friend bool operator<=(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return !(rhs < lhs);
  }
  /// Same as `!(lhs < rhs)` (see \ref operator<)
  friend bool operator>=(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return !(lhs < rhs);
  }

  /// Check if this ZBDD set is invalid
  ///
  /// A ZBDD set created by the default constructor zbdd_set() is invalid as
  /// well as a \ref zbdd_set instance that has been moved (via
  /// zbdd_set(zbdd_set &&other)). Moreover, if an operation tries to allocate
  /// new nodes but runs out of memory, then it returns an invalid function.
  ///
  /// @returns  `true` iff this ZBDD set is invalid
  bool is_invalid() const noexcept { return _func._p == nullptr; }

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
  /// Note that the domain of f is 𝔹ⁿ⁺¹ while the domain of f<sub>true</sub> and
  /// f<sub>false</sub> is 𝔹ⁿ. (Remember that, e.g., g(x₀) = x₀ and
  /// g'(x₀, x₁) = x₀ have different representations as ZBDDs.)
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The pair `{f_true, f_false}` if `f` is valid and references an
  ///           inner node, otherwise a pair of invalid functions.
  std::pair<zbdd_set, zbdd_set> cofactors() const noexcept {
    capi::oxidd_zbdd_pair_t p = capi::oxidd_zbdd_cofactors(_func);
    return {p.a, p.b};
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
  zbdd_set cofactor_true() const noexcept {
    return capi::oxidd_zbdd_cofactor_true(_func);
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
  zbdd_set cofactor_false() const noexcept {
    return capi::oxidd_zbdd_cofactor_true(_func);
  }

  /// Compute the ZBDD for the negation `¬this`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this|)
  ///
  /// @returns  The ZBDD set (may be invalid if the operation runs out of
  ///           memory)
  zbdd_set operator~() const noexcept { return capi::oxidd_zbdd_not(_func); }
  /// Compute the ZBDD for the conjunction `lhs ∧ rhs`
  ///
  /// The conjunction on Boolean functions may equivalently be viewed as an
  /// intersection of sets `lhs ∩ rhs`.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD set/function (may be invalid if the operation runs out
  ///           of memory)
  friend zbdd_set operator&(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return capi::oxidd_zbdd_intsec(lhs._func, rhs._func);
  }
  /// Assignment version of \ref operator&
  zbdd_set &operator&=(const zbdd_set &rhs) noexcept {
    return (*this = *this & rhs);
  }
  /// Compute the ZBDD for the disjunction `lhs ∨ rhs`
  ///
  /// The disjunction on Boolean functions may equivalently be viewed as a
  /// union of sets `lhs ∪ rhs`.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD set/function (may be invalid if the operation runs out
  ///           of memory)
  friend zbdd_set operator|(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return capi::oxidd_zbdd_union(lhs._func, rhs._func);
  }
  /// Assignment version of \ref operator|
  zbdd_set &operator|=(const zbdd_set &rhs) noexcept {
    return (*this = *this | rhs);
  }
  /// Compute the ZBDD for the exclusive disjunction `lhs ⊕ rhs`
  ///
  /// The disjunction on Boolean functions may equivalently be viewed as a
  /// symmetric difference on sets.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD set/function (may be invalid if the operation runs out
  ///           of memory)
  friend zbdd_set operator^(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return capi::oxidd_zbdd_xor(lhs._func, rhs._func);
  }
  /// Assignment version of \ref operator^
  zbdd_set &operator^=(const zbdd_set &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  /// Compute the ZBDD for the set difference `lhs ∖ rhs`
  ///
  /// This is equivalent to the strict implication `rhs < lhs` on Boolean
  /// functions.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The ZBDD set/function (may be invalid if the operation runs out
  ///           of memory)
  friend zbdd_set operator-(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return capi::oxidd_zbdd_diff(lhs._func, rhs._func);
  }
  /// Assignment version of \ref operator-
  zbdd_set &operator-=(const zbdd_set &rhs) noexcept {
    return (*this = *this - rhs);
  }
  /// Compute the ZBDD for the negated conjunction `this ⊼ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  zbdd_set nand(const zbdd_set &rhs) const noexcept {
    return capi::oxidd_zbdd_nand(_func, rhs._func);
  }
  /// Compute the ZBDD for the negated disjunction `this ⊽ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  zbdd_set nor(const zbdd_set &rhs) const noexcept {
    return capi::oxidd_zbdd_nor(_func, rhs._func);
  }
  /// Compute the ZBDD for the equivalence `this ↔ rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  zbdd_set equiv(const zbdd_set &rhs) const noexcept {
    return capi::oxidd_zbdd_equiv(_func, rhs._func);
  }
  /// Compute the ZBDD for the implication `this → rhs` (or `this ≤ rhs`)
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  zbdd_set imp(const zbdd_set &rhs) const noexcept {
    return capi::oxidd_zbdd_imp(_func, rhs._func);
  }
  /// Compute the ZBDD for the strict implication `this < rhs`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The ZBDD function (may be invalid if the operation runs out of
  ///           memory)
  zbdd_set imp_strict(const zbdd_set &rhs) const noexcept {
    return capi::oxidd_zbdd_imp_strict(_func, rhs._func);
  }
  /// Compute the ZBDD for the conditional `this ? t : e`
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(|this| · |t| · |e|)
  ///
  /// @returns  The ZBDD Boolean function (may be invalid if the operation runs
  ///           out of memory)
  zbdd_set ite(const zbdd_set &t, const zbdd_set &e) const noexcept {
    return capi::oxidd_zbdd_ite(_func, t._func, e._func);
  }

  /// Create a new ZBDD node at the level of `this` with the given `hi` and `lo`
  /// edges
  ///
  /// `var` must be a singleton set.
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The ZBDD set representing referencing the new node (may be
  ///           invalid if the operation runs out of memory)
  zbdd_set make_node(zbdd_set &&hi, zbdd_set &&lo) const noexcept {
    capi::oxidd_zbdd_t h = hi._func, l = lo._func;
    hi._func._p = nullptr;
    lo._func._p = nullptr;
    return capi::oxidd_zbdd_make_node(_func, h, l);
  }

  /// Count descendant nodes
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  Node count including the two terminal nodes
  uint64_t node_count() const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_node_count(_func);
  }

  /// Count the number of satisfying assignments
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  Count of satisfying assignments
  double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_sat_count_double(_func, vars);
  }

  /// Pick a satisfying assignment
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  A satisfying assignment. If `f` is unsatisfiable, the assignment
  ///           is empty.
  assignment pick_cube() const noexcept {
    assert(_func._p);
    return capi::oxidd_zbdd_pick_cube(_func);
  }
};

inline zbdd_set zbdd_manager::new_singleton() noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_new_singleton(_manager);
}
inline zbdd_set zbdd_manager::new_var() noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_new_var(_manager);
}
inline zbdd_set zbdd_manager::empty() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_empty(_manager);
}
inline zbdd_set zbdd_manager::base() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_base(_manager);
}
inline zbdd_set zbdd_manager::top() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_true(_manager);
}
inline zbdd_set zbdd_manager::bot() const noexcept {
  assert(_manager._p);
  return capi::oxidd_zbdd_false(_manager);
}

} // namespace oxidd

/// @cond

/// Partial specialization for oxidd::zbdd_set
template <> struct std::hash<oxidd::zbdd_set> {
  std::size_t operator()(const oxidd::zbdd_set &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

/// @endcond
