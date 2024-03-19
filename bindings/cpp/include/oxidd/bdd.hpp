/// @file  bdd.hpp
/// @brief Binary decision diagrams (without complement edges)

#pragma once

#include <oxidd/utils.hpp>

namespace oxidd {

class bdd_function;

/// Manager for binary decision diagrams (without complement edges)
///
/// Models the oxidd::manager concept.
class bdd_manager {
  /// Wrapped CAPI BDD manager
  capi::oxidd_bdd_manager_t _manager;

  /// Create a new BDD manager from a manager instance of the CAPI
  bdd_manager(capi::oxidd_bdd_manager_t manager) noexcept : _manager(manager) {}

public:
  /// Default constructor, yields an invalid manager
  bdd_manager() noexcept { _manager._p = nullptr; }
  /// Create a new BDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  bdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
              uint32_t threads) noexcept {
    _manager = capi::oxidd_bdd_manager_new(inner_node_capacity,
                                           apply_cache_capacity, threads);
  }
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Runtime complexity: O(1)
  bdd_manager(const bdd_manager &other) noexcept : _manager(other._manager) {
    oxidd_bdd_manager_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  bdd_manager(bdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~bdd_manager() noexcept {
    if (_manager._p != nullptr)
      oxidd_bdd_manager_unref(_manager);
  }

  /// Copy assignment operator
  bdd_manager &operator=(const bdd_manager &rhs) noexcept {
    if (_manager._p != nullptr)
      oxidd_bdd_manager_unref(_manager);
    _manager = rhs._manager;
    oxidd_bdd_manager_ref(_manager);
    return *this;
  }
  /// Move assignment operator
  bdd_manager &operator=(bdd_manager &&rhs) noexcept {
    if (_manager._p != nullptr)
      oxidd_bdd_manager_unref(_manager);
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
  /// Same as `!(lhs == rhs)` (see \ref operator==)
  friend bool operator!=(const bdd_manager &lhs,
                         const bdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor bdd_manager() is
  /// invalid as well as a \ref bdd_manager instance that has been moved (via
  /// bdd_manager(bdd_manager &&other)).
  ///
  /// @returns  `true` iff this manager reference is invalid
  bool is_invalid() const noexcept { return !_manager._p; }

  /// Get a fresh variable, i.e., a function that is true if and only if the
  /// variable is true. This adds a new level to a decision diagram.
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires an exclusive manager lock.
  ///
  /// @returns  The BDD function representing the variable
  bdd_function new_var() noexcept;

  /// Get the constant true BCDD function ⊤
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊤ as BDD function
  bdd_function top() const noexcept;
  /// Get the constant false BCDD function ⊥
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊥ as BDD function
  bdd_function bot() const noexcept;

  /// Get the number of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires a shared manager lock.
  ///
  /// @returns  The number of inner nodes
  size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return oxidd_bdd_num_inner_nodes(_manager);
  }
};

class bdd_function {
  /// Wrapped BDD function
  capi::oxidd_bdd_t _func;

  friend class bdd_manager;
  friend struct std::hash<bdd_function>;

  bdd_function(capi::oxidd_bdd_t func) noexcept : _func(func) {}

public:
  bdd_function() noexcept { _func._p = nullptr; }
  bdd_function(const bdd_function &other) noexcept : _func(other._func) {
    oxidd_bdd_ref(_func);
  }
  bdd_function(bdd_function &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~bdd_function() noexcept {
    if (_func._p != nullptr)
      oxidd_bdd_unref(_func);
  }

  bdd_function &operator=(const bdd_function &rhs) noexcept {
    if (_func._p != nullptr)
      oxidd_bdd_unref(_func);
    _func = rhs._func;
    if (_func._p != nullptr)
      oxidd_bdd_ref(_func);
    return *this;
  }
  bdd_function &operator=(bdd_function &&rhs) noexcept {
    if (_func._p != nullptr)
      oxidd_bdd_unref(_func);
    _func = rhs._func;
    rhs._func._p = nullptr;
    return *this;
  }

  bool is_invalid() const noexcept { return _func._p == nullptr; }

  friend bool operator==(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  friend bool operator!=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(lhs == rhs);
  }

  friend bool operator<(const bdd_function &lhs,
                        const bdd_function &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  friend bool operator>(const bdd_function &lhs,
                        const bdd_function &rhs) noexcept {
    return rhs < lhs;
  }
  friend bool operator<=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(rhs < lhs);
  }
  friend bool operator>=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(lhs < rhs);
  }

  bdd_function operator~() const noexcept { return oxidd_bdd_not(_func); }
  bdd_function operator&(const bdd_function &rhs) const noexcept {
    return bdd_function(oxidd_bdd_and(_func, rhs._func));
  }
  bdd_function &operator&=(const bdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  bdd_function operator|(const bdd_function &rhs) const noexcept {
    return bdd_function(oxidd_bdd_or(_func, rhs._func));
  }
  bdd_function &operator|=(const bdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  bdd_function operator^(const bdd_function &rhs) const noexcept {
    return bdd_function(oxidd_bdd_xor(_func, rhs._func));
  }
  bdd_function &operator^=(const bdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  bdd_function ite(const bdd_function &t,
                   const bdd_function &e) const noexcept {
    return bdd_function(oxidd_bdd_ite(_func, t._func, e._func));
  }

  bdd_function forall(const bdd_function &vars) const noexcept {
    return oxidd_bdd_forall(_func, vars._func);
  }
  bdd_function exist(const bdd_function &vars) const noexcept {
    return oxidd_bdd_exist(_func, vars._func);
  }
  bdd_function unique(const bdd_function &vars) const noexcept {
    return oxidd_bdd_unique(_func, vars._func);
  }

  uint64_t node_count() const noexcept {
    assert(_func._p);
    return oxidd_bdd_node_count(_func);
  }

  double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return oxidd_bdd_sat_count_double(_func, vars);
  }

  assignment pick_cube() const noexcept {
    assert(_func._p);
    return assignment(oxidd_bdd_pick_cube(_func));
  }
};

inline bdd_function bdd_manager::new_var() noexcept {
  assert(_manager._p);
  return bdd_function(oxidd_bdd_new_var(_manager));
}
inline bdd_function bdd_manager::top() const noexcept {
  assert(_manager._p);
  return oxidd_bdd_true(_manager);
}
inline bdd_function bdd_manager::bot() const noexcept {
  assert(_manager._p);
  return oxidd_bdd_false(_manager);
}

} // namespace oxidd

namespace std {

/// Partial specialization for oxidd::bdd_function
template <> struct std::hash<oxidd::bdd_function> {
  std::size_t operator()(const oxidd::bdd_function &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

} // namespace std
