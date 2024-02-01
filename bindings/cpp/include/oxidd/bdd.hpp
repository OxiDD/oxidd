/// Binary decision diagrams (without complement edges)

#pragma once

#include <oxidd/utils.hpp>

namespace oxidd {

class bdd_function;

class bdd_manager {
  capi::oxidd_bdd_manager_t _manager;

  bdd_manager(capi::oxidd_bdd_manager_t manager) noexcept : _manager(manager) {}

public:
  bdd_manager() noexcept { _manager.__p = nullptr; }
  bdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
              uint32_t threads) noexcept {
    _manager = capi::oxidd_bdd_manager_new(inner_node_capacity,
                                           apply_cache_capacity, threads);
  }
  bdd_manager(const bdd_manager &other) noexcept : _manager(other._manager) {
    oxidd_bdd_manager_ref(_manager);
  }
  bdd_manager(bdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager.__p = nullptr;
  }

  ~bdd_manager() noexcept {
    if (_manager.__p != nullptr)
      oxidd_bdd_manager_unref(_manager);
  }

  bdd_manager &operator=(const bdd_manager &rhs) noexcept {
    if (_manager.__p != nullptr)
      oxidd_bdd_manager_unref(_manager);
    _manager = rhs._manager;
    oxidd_bdd_manager_ref(_manager);
    return *this;
  }

  bool is_invalid() const noexcept { return !_manager.__p; }

  bdd_function new_var() noexcept;

  bdd_function top() const noexcept;
  bdd_function bot() const noexcept;

  size_t num_inner_nodes() const noexcept {
    assert(_manager.__p != nullptr);
    return oxidd_bdd_num_inner_nodes(_manager);
  }
};

class bdd_function {
  capi::oxidd_bdd_t _func;

  friend class bdd_manager;
  bdd_function(capi::oxidd_bdd_t func) noexcept : _func(func) {}

public:
  bdd_function() noexcept { _func.__p = nullptr; }
  bdd_function(const bdd_function &other) noexcept : _func(other._func) {
    oxidd_bdd_ref(_func);
  }
  bdd_function(bdd_function &&other) noexcept : _func(other._func) {
    other._func.__p = nullptr;
  }

  ~bdd_function() noexcept {
    if (_func.__p != nullptr)
      oxidd_bdd_unref(_func);
  }

  bdd_function &operator=(const bdd_function &rhs) noexcept {
    if (_func.__p != nullptr)
      oxidd_bdd_unref(_func);
    _func = rhs._func;
    if (_func.__p != nullptr)
      oxidd_bdd_ref(_func);
    return *this;
  }
  bdd_function &operator=(bdd_function &&rhs) noexcept {
    if (_func.__p != nullptr)
      oxidd_bdd_unref(_func);
    _func = rhs._func;
    rhs._func.__p = nullptr;
    return *this;
  }

  bool is_invalid() const noexcept { return _func.__p == nullptr; }

  bool operator==(const bdd_function &rhs) const noexcept {
    return (_func.__p == rhs._func.__p);
  }
  bool operator!=(const bdd_function &rhs) const noexcept {
    return !(*this == rhs);
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
    assert(_func.__p);
    return oxidd_bdd_node_count(_func);
  }

  uint64_t sat_count_uint64(oxidd_level_no_t vars) const noexcept {
    assert(_func.__p);
    return oxidd_bdd_sat_count_uint64(_func, vars);
  }
  double sat_count_double(oxidd_level_no_t vars) const noexcept {
    assert(_func.__p);
    return oxidd_bdd_sat_count_double(_func, vars);
  }

  assignment pick_cube() const noexcept {
    assert(_func.__p);
    return assignment(oxidd_bdd_pick_cube(_func));
  }
};

inline bdd_function bdd_manager::new_var() noexcept {
  assert(_manager.__p);
  return bdd_function(oxidd_bdd_new_var(_manager));
}
inline bdd_function bdd_manager::top() const noexcept {
  assert(_manager.__p);
  return oxidd_bdd_true(_manager);
}
inline bdd_function bdd_manager::bot() const noexcept {
  assert(_manager.__p);
  return oxidd_bdd_false(_manager);
}

} // namespace oxidd
