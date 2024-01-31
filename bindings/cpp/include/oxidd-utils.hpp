/// Primitives and utilities

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <oxidd.h>

namespace oxidd {

using capi::oxidd_level_no_t;

enum class opt_bool : int8_t {
  NONE = -1,
  FALSE = 0,
  TRUE = 1,
};

class assignment {
  const capi::oxidd_assignment_t _assignment;

  assignment(capi::oxidd_assignment_t a) : _assignment(a) {}
  assignment(assignment &other) = delete; // use as_vector() to copy
  friend class bdd_function;
  friend class bcdd_function;
  friend class zbdd_set;

public:
  ~assignment() noexcept { oxidd_assignment_free(_assignment); }

  const opt_bool *data() const noexcept { return (opt_bool *)_assignment.data; }
  size_t size() const noexcept { return _assignment.len; }

  const opt_bool *begin() const noexcept { return data(); }
  const opt_bool *end() const noexcept { return data() + size(); }

  const opt_bool *cbegin() const noexcept { return data(); }
  const opt_bool *cend() const noexcept { return data() + size(); }

  const opt_bool &operator[](size_t idx) const noexcept {
    assert(idx < size());
    return data()[idx];
  }

  std::vector<opt_bool> as_vector() const noexcept {
    return std::vector<opt_bool>(begin(), end());
  }
};

} // namespace oxidd
