/// @file Zero-suppressed binary decision diagrams

#pragma once

#include <oxidd/utils.hpp>

namespace oxidd {

class zbdd_set;

class zbdd_manager {
  capi::oxidd_zbdd_manager_t _manager;

  zbdd_manager(capi::oxidd_zbdd_manager_t manager) noexcept
      : _manager(manager) {}

public:
  zbdd_manager() noexcept { _manager._p = nullptr; }
  zbdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept {
    _manager = capi::oxidd_zbdd_manager_new(inner_node_capacity,
                                            apply_cache_capacity, threads);
  }
  zbdd_manager(const zbdd_manager &other) noexcept : _manager(other._manager) {
    oxidd_zbdd_manager_ref(_manager);
  }
  zbdd_manager(zbdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~zbdd_manager() noexcept {
    if (_manager._p != nullptr)
      oxidd_zbdd_manager_unref(_manager);
  }

  zbdd_manager &operator=(const zbdd_manager &rhs) noexcept {
    if (_manager._p != nullptr)
      oxidd_zbdd_manager_unref(_manager);
    _manager = rhs._manager;
    oxidd_zbdd_manager_ref(_manager);
    return *this;
  }

  friend bool operator==(const zbdd_manager &lhs,
                         const zbdd_manager &rhs) noexcept {
    return lhs._manager._p == rhs._manager._p;
  }
  friend bool operator!=(const zbdd_manager &lhs,
                         const zbdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  bool is_invalid() const noexcept { return !_manager._p; }

  zbdd_set new_singleton() noexcept;
  zbdd_set new_var() noexcept;

  zbdd_set empty() const noexcept;
  zbdd_set base() const noexcept;
  zbdd_set top() const noexcept;
  zbdd_set bot() const noexcept;

  size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return oxidd_zbdd_num_inner_nodes(_manager);
  }
};

class zbdd_set {
  capi::oxidd_zbdd_t _func;

  friend class zbdd_manager;
  friend struct std::hash<zbdd_set>;

  zbdd_set(capi::oxidd_zbdd_t func) noexcept : _func(func) {}

public:
  zbdd_set() noexcept { _func._p = nullptr; }
  zbdd_set(const zbdd_set &other) noexcept : _func(other._func) {
    oxidd_zbdd_ref(_func);
  }
  zbdd_set(zbdd_set &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~zbdd_set() noexcept {
    if (_func._p != nullptr)
      oxidd_zbdd_unref(_func);
  }

  zbdd_set &operator=(const zbdd_set &rhs) noexcept {
    if (_func._p != nullptr)
      oxidd_zbdd_unref(_func);
    _func = rhs._func;
    if (_func._p != nullptr)
      oxidd_zbdd_ref(_func);
    return *this;
  }
  zbdd_set &operator=(zbdd_set &&rhs) noexcept {
    if (_func._p != nullptr)
      oxidd_zbdd_unref(_func);
    _func = rhs._func;
    rhs._func._p = nullptr;
    return *this;
  }

  bool is_invalid() const noexcept { return _func._p == nullptr; }

  friend bool operator==(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  friend bool operator!=(const zbdd_set &lhs, const zbdd_set &rhs) noexcept {
    return !(lhs == rhs);
  }

  zbdd_set operator~() const noexcept {
    return zbdd_set(oxidd_zbdd_not(_func));
  }
  zbdd_set operator&(const zbdd_set &rhs) const noexcept {
    return zbdd_set(oxidd_zbdd_intsec(_func, rhs._func));
  }
  zbdd_set &operator&=(const zbdd_set &rhs) noexcept {
    return (*this = *this & rhs);
  }
  zbdd_set operator|(const zbdd_set &rhs) const noexcept {
    return zbdd_set(oxidd_zbdd_union(_func, rhs._func));
  }
  zbdd_set &operator|=(const zbdd_set &rhs) noexcept {
    return (*this = *this | rhs);
  }
  zbdd_set operator^(const zbdd_set &rhs) const noexcept {
    return zbdd_set(oxidd_zbdd_xor(_func, rhs._func));
  }
  zbdd_set &operator^=(const zbdd_set &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  zbdd_set operator-(const zbdd_set &rhs) const noexcept {
    return zbdd_set(oxidd_zbdd_diff(_func, rhs._func));
  }
  zbdd_set &operator-=(const zbdd_set &rhs) noexcept {
    return (*this = *this - rhs);
  }

  zbdd_set make_zbdd_node(zbdd_set &&hi, zbdd_set &&lo) const noexcept {
    capi::oxidd_zbdd_t h = hi._func, l = lo._func;
    hi._func._p = nullptr;
    lo._func._p = nullptr;
    return zbdd_set(oxidd_zbdd_make_node(_func, h, l));
  }

  uint64_t node_count() const noexcept {
    assert(_func._p);
    return oxidd_zbdd_node_count(_func);
  }

  uint64_t sat_count_uint64(level_no_t vars) const noexcept {
    assert(_func._p);
    return oxidd_zbdd_sat_count_uint64(_func, vars);
  }
  double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return oxidd_zbdd_sat_count_double(_func, vars);
  }

  assignment pick_cube() const noexcept {
    assert(_func._p);
    return assignment(oxidd_zbdd_pick_cube(_func));
  }
};

inline zbdd_set zbdd_manager::new_singleton() noexcept {
  assert(_manager._p);
  return zbdd_set(oxidd_zbdd_new_singleton(_manager));
}
inline zbdd_set zbdd_manager::new_var() noexcept {
  assert(_manager._p);
  return zbdd_set(oxidd_zbdd_new_var(_manager));
}
inline zbdd_set zbdd_manager::empty() const noexcept {
  assert(_manager._p);
  return oxidd_zbdd_empty(_manager);
}
inline zbdd_set zbdd_manager::base() const noexcept {
  assert(_manager._p);
  return oxidd_zbdd_base(_manager);
}
inline zbdd_set zbdd_manager::top() const noexcept {
  assert(_manager._p);
  return oxidd_zbdd_true(_manager);
}
inline zbdd_set zbdd_manager::bot() const noexcept {
  assert(_manager._p);
  return oxidd_zbdd_false(_manager);
}

} // namespace oxidd

namespace std {

template <> struct std::hash<oxidd::zbdd_set> {
  std::size_t operator()(const oxidd::zbdd_set &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

} // namespace std
