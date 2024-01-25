/// Complement edge binary decision diagrams

#pragma once

#include <oxidd-utils.hpp>

namespace oxidd {

class cbdd_function;

class cbdd_manager {
  capi::oxidd_cbdd_manager_t _manager;

  cbdd_manager(capi::oxidd_cbdd_manager_t manager) noexcept
      : _manager(manager) {}

public:
  cbdd_manager() noexcept { _manager.__p = nullptr; }
  cbdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept {
    _manager = capi::oxidd_cbdd_manager_new(inner_node_capacity,
                                            apply_cache_capacity, threads);
  }
  cbdd_manager(const cbdd_manager &other) noexcept : _manager(other._manager) {
    oxidd_cbdd_manager_ref(_manager);
  }
  cbdd_manager(cbdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager.__p = nullptr;
  }

  ~cbdd_manager() noexcept {
    if (_manager.__p != nullptr)
      oxidd_cbdd_manager_unref(_manager);
  }

  cbdd_manager &operator=(const cbdd_manager &rhs) noexcept {
    if (_manager.__p != nullptr)
      oxidd_cbdd_manager_unref(_manager);
    _manager = rhs._manager;
    oxidd_cbdd_manager_ref(_manager);
    return *this;
  }

  bool is_invalid() noexcept { return !_manager.__p; }

  cbdd_function new_var() noexcept;

  cbdd_function top() const noexcept;
  cbdd_function bot() const noexcept;

  size_t num_inner_nodes() const noexcept {
    assert(_manager.__p != nullptr);
    return oxidd_cbdd_num_inner_nodes(_manager);
  }
};

class cbdd_function {
  capi::oxidd_cbdd_t _func;

  friend class cbdd_manager;
  cbdd_function(capi::oxidd_cbdd_t func) noexcept : _func(func) {}

public:
  cbdd_function() noexcept { _func.__p = nullptr; }
  cbdd_function(const cbdd_function &other) noexcept : _func(other._func) {
    oxidd_cbdd_ref(_func);
  }
  cbdd_function(cbdd_function &&other) noexcept : _func(other._func) {
    other._func.__p = nullptr;
  }

  ~cbdd_function() noexcept {
    if (_func.__p != nullptr)
      oxidd_cbdd_unref(_func);
  }

  cbdd_function &operator=(const cbdd_function &rhs) noexcept {
    if (_func.__p != nullptr)
      oxidd_cbdd_unref(_func);
    _func = rhs._func;
    if (_func.__p != nullptr)
      oxidd_cbdd_ref(_func);
    return *this;
  }
  cbdd_function &operator=(cbdd_function &&rhs) noexcept {
    if (_func.__p != nullptr)
      oxidd_cbdd_unref(_func);
    _func = rhs._func;
    rhs._func.__p = nullptr;
    return *this;
  }

  bool is_invalid() noexcept { return _func.__p == nullptr; }

  bool operator==(const cbdd_function &rhs) const noexcept {
    return (_func.__p == rhs._func.__p);
  }
  bool operator!=(const cbdd_function &rhs) const noexcept {
    return !(*this == rhs);
  }

  cbdd_function operator~() const noexcept { return oxidd_cbdd_not(_func); }
  cbdd_function operator&(const cbdd_function &rhs) const noexcept {
    return cbdd_function(oxidd_cbdd_and(_func, rhs._func));
  }
  cbdd_function &operator&=(const cbdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  cbdd_function operator|(const cbdd_function &rhs) const noexcept {
    return cbdd_function(oxidd_cbdd_or(_func, rhs._func));
  }
  cbdd_function &operator|=(const cbdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  cbdd_function operator^(const cbdd_function &rhs) const noexcept {
    return cbdd_function(oxidd_cbdd_xor(_func, rhs._func));
  }
  cbdd_function &operator^=(const cbdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  cbdd_function ite(const cbdd_function &t,
                    const cbdd_function &e) const noexcept {
    return cbdd_function(oxidd_cbdd_ite(_func, t._func, e._func));
  }

  cbdd_function forall(const cbdd_function &vars) const noexcept {
    return oxidd_cbdd_forall(_func, vars._func);
  }
  cbdd_function exist(const cbdd_function &vars) const noexcept {
    return oxidd_cbdd_exist(_func, vars._func);
  }
  cbdd_function unique(const cbdd_function &vars) const noexcept {
    return oxidd_cbdd_unique(_func, vars._func);
  }

  uint64_t node_count() const noexcept {
    assert(_func.__p);
    return oxidd_cbdd_node_count(_func);
  }

  uint64_t sat_count_uint64(oxidd_level_no_t vars) const noexcept {
    assert(_func.__p);
    return oxidd_cbdd_sat_count_uint64(_func, vars);
  }
  double sat_count_double(oxidd_level_no_t vars) const noexcept {
    assert(_func.__p);
    return oxidd_cbdd_sat_count_double(_func, vars);
  }

  assignment pick_cube() const noexcept {
    assert(_func.__p);
    return assignment(oxidd_cbdd_pick_cube(_func));
  }
};

inline cbdd_function cbdd_manager::new_var() noexcept {
  assert(_manager.__p);
  return cbdd_function(oxidd_cbdd_new_var(_manager));
}
inline cbdd_function cbdd_manager::top() const noexcept {
  assert(_manager.__p);
  return oxidd_cbdd_true(_manager);
}
inline cbdd_function cbdd_manager::bot() const noexcept {
  assert(_manager.__p);
  return oxidd_cbdd_false(_manager);
}

} // namespace oxidd
