/// Complement edge binary decision diagrams

#pragma once

#include <oxidd/utils.hpp>

namespace oxidd {

class bcdd_function;

class bcdd_manager {
  capi::oxidd_bcdd_manager_t _manager;

  bcdd_manager(capi::oxidd_bcdd_manager_t manager) noexcept
      : _manager(manager) {}

public:
  bcdd_manager() noexcept { _manager.__p = nullptr; }
  bcdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept {
    _manager = capi::oxidd_bcdd_manager_new(inner_node_capacity,
                                            apply_cache_capacity, threads);
  }
  bcdd_manager(const bcdd_manager &other) noexcept : _manager(other._manager) {
    oxidd_bcdd_manager_ref(_manager);
  }
  bcdd_manager(bcdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager.__p = nullptr;
  }

  ~bcdd_manager() noexcept {
    if (_manager.__p != nullptr)
      oxidd_bcdd_manager_unref(_manager);
  }

  bcdd_manager &operator=(const bcdd_manager &rhs) noexcept {
    if (_manager.__p != nullptr)
      oxidd_bcdd_manager_unref(_manager);
    _manager = rhs._manager;
    oxidd_bcdd_manager_ref(_manager);
    return *this;
  }

  bool is_invalid() noexcept { return !_manager.__p; }

  bcdd_function new_var() noexcept;

  bcdd_function top() const noexcept;
  bcdd_function bot() const noexcept;

  size_t num_inner_nodes() const noexcept {
    assert(_manager.__p != nullptr);
    return oxidd_bcdd_num_inner_nodes(_manager);
  }
};

class bcdd_function {
  capi::oxidd_bcdd_t _func;

  friend class bcdd_manager;
  bcdd_function(capi::oxidd_bcdd_t func) noexcept : _func(func) {}

public:
  bcdd_function() noexcept { _func.__p = nullptr; }
  bcdd_function(const bcdd_function &other) noexcept : _func(other._func) {
    oxidd_bcdd_ref(_func);
  }
  bcdd_function(bcdd_function &&other) noexcept : _func(other._func) {
    other._func.__p = nullptr;
  }

  ~bcdd_function() noexcept {
    if (_func.__p != nullptr)
      oxidd_bcdd_unref(_func);
  }

  bcdd_function &operator=(const bcdd_function &rhs) noexcept {
    if (_func.__p != nullptr)
      oxidd_bcdd_unref(_func);
    _func = rhs._func;
    if (_func.__p != nullptr)
      oxidd_bcdd_ref(_func);
    return *this;
  }
  bcdd_function &operator=(bcdd_function &&rhs) noexcept {
    if (_func.__p != nullptr)
      oxidd_bcdd_unref(_func);
    _func = rhs._func;
    rhs._func.__p = nullptr;
    return *this;
  }

  bool is_invalid() noexcept { return _func.__p == nullptr; }

  bool operator==(const bcdd_function &rhs) const noexcept {
    return (_func.__p == rhs._func.__p);
  }
  bool operator!=(const bcdd_function &rhs) const noexcept {
    return !(*this == rhs);
  }

  bcdd_function operator~() const noexcept { return oxidd_bcdd_not(_func); }
  bcdd_function operator&(const bcdd_function &rhs) const noexcept {
    return bcdd_function(oxidd_bcdd_and(_func, rhs._func));
  }
  bcdd_function &operator&=(const bcdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  bcdd_function operator|(const bcdd_function &rhs) const noexcept {
    return bcdd_function(oxidd_bcdd_or(_func, rhs._func));
  }
  bcdd_function &operator|=(const bcdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  bcdd_function operator^(const bcdd_function &rhs) const noexcept {
    return bcdd_function(oxidd_bcdd_xor(_func, rhs._func));
  }
  bcdd_function &operator^=(const bcdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  bcdd_function ite(const bcdd_function &t,
                    const bcdd_function &e) const noexcept {
    return bcdd_function(oxidd_bcdd_ite(_func, t._func, e._func));
  }

  bcdd_function forall(const bcdd_function &vars) const noexcept {
    return oxidd_bcdd_forall(_func, vars._func);
  }
  bcdd_function exist(const bcdd_function &vars) const noexcept {
    return oxidd_bcdd_exist(_func, vars._func);
  }
  bcdd_function unique(const bcdd_function &vars) const noexcept {
    return oxidd_bcdd_unique(_func, vars._func);
  }

  uint64_t node_count() const noexcept {
    assert(_func.__p);
    return oxidd_bcdd_node_count(_func);
  }

  uint64_t sat_count_uint64(oxidd_level_no_t vars) const noexcept {
    assert(_func.__p);
    return oxidd_bcdd_sat_count_uint64(_func, vars);
  }
  double sat_count_double(oxidd_level_no_t vars) const noexcept {
    assert(_func.__p);
    return oxidd_bcdd_sat_count_double(_func, vars);
  }

  assignment pick_cube() const noexcept {
    assert(_func.__p);
    return assignment(oxidd_bcdd_pick_cube(_func));
  }
};

inline bcdd_function bcdd_manager::new_var() noexcept {
  assert(_manager.__p);
  return bcdd_function(oxidd_bcdd_new_var(_manager));
}
inline bcdd_function bcdd_manager::top() const noexcept {
  assert(_manager.__p);
  return oxidd_bcdd_true(_manager);
}
inline bcdd_function bcdd_manager::bot() const noexcept {
  assert(_manager.__p);
  return oxidd_bcdd_false(_manager);
}

} // namespace oxidd
