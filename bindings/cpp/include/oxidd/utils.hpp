/// @file  utils.hpp
/// @brief Primitives and utilities

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <oxidd.h>

/// OxiDD's main namespace
namespace oxidd {

/// Level number type
using level_no_t = capi::oxidd_level_no_t;

/// Optional `bool`
enum class opt_bool : int8_t {
  NONE = -1,
  FALSE = 0,
  TRUE = 1,
};

/// Assignment, similar to a std::vector<opt_bool>
///
/// This class has no copy constructor to avoid additional calls into the FFI.
/// Use as_vector() instead.
class assignment {
  capi::oxidd_assignment_t _assignment;

  assignment(capi::oxidd_assignment_t a) : _assignment(a) {}
  assignment(assignment &other) = delete; // use as_vector() to copy
  friend class bdd_function;
  friend class bcdd_function;
  friend class zbdd_set;

public:
  /// Move constructor: empties `other`
  assignment(assignment &&other) : _assignment(other._assignment) {
    other._assignment.data = nullptr;
    other._assignment.len = 0;
  }

  ~assignment() noexcept { oxidd_assignment_free(_assignment); }

  /// Get a pointer to the underlying data
  const opt_bool *data() const noexcept {
    return (const opt_bool *)_assignment.data;
  }
  /// Get the number of elements in the assignment
  size_t size() const noexcept { return _assignment.len; }

  /// Returns an iterator to the beginning
  const opt_bool *begin() const noexcept { return data(); }
  /// Returns an iterator to the end
  const opt_bool *end() const noexcept { return data() + size(); }

  /// Returns an iterator to the beginning
  const opt_bool *cbegin() const noexcept { return data(); }
  /// Returns an iterator to the end
  const opt_bool *cend() const noexcept { return data() + size(); }

  /// Access the element at `idx`
  ///
  /// @param  idx  The index, must be less than size(), otherwise the call
  ///              results in undefined behavior
  ///
  /// @returns  A reference to the element
  const opt_bool &operator[](size_t idx) const noexcept {
    assert(idx < size());
    return data()[idx];
  }

  /// Copy the data into a std::vector<opt_bool>
  std::vector<opt_bool> as_vector() const noexcept {
    return std::vector<opt_bool>(begin(), end());
  }
};

} // namespace oxidd
