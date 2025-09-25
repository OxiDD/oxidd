/// @file  util.hpp
/// @brief Primitives and utilities

#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ostream>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>
#include <version>

#include <oxidd/capi.h>
#include <oxidd/compat.hpp>

/// OxiDD's main namespace
namespace oxidd {

/// Level numbers
using level_no_t = capi::oxidd_level_no_t;
/// Variable numbers
using var_no_t = capi::oxidd_var_no_t;

/// Variable number range
using var_no_range_t = std::ranges::iota_view<var_no_t, var_no_t>;

/// Invalid level number number
constexpr level_no_t invalid_level_no = std::numeric_limits<level_no_t>::max();
/// Invalid variable number
constexpr var_no_t invalid_var_no = std::numeric_limits<var_no_t>::max();

/// Utility classes
namespace util {

/// Optional `bool`
enum class opt_bool : int8_t {
  // NOLINTBEGIN(readability-identifier-naming)
  NONE = -1,
  FALSE = 0,
  TRUE = 1,
  // NOLINTEND(readability-identifier-naming)
};

/// Binary operator on Boolean functions
enum class boolean_operator : uint8_t {
  // NOLINTBEGIN(readability-identifier-naming)
  /// Conjunction `lhs ∧ rhs`
  AND = capi::OXIDD_BOOLEAN_OPERATOR_AND,
  /// Disjunction `lhs ∨ rhs`
  OR = capi::OXIDD_BOOLEAN_OPERATOR_OR,
  /// Exclusive disjunction `lhs ⊕ rhs`
  XOR = capi::OXIDD_BOOLEAN_OPERATOR_XOR,
  /// Equivalence `lhs ↔ rhs`
  EQUIV = capi::OXIDD_BOOLEAN_OPERATOR_EQUIV,
  /// Negated conjunction `lhs ⊼ rhs`
  NAND = capi::OXIDD_BOOLEAN_OPERATOR_NAND,
  /// Negated disjunction `lhs ⊽ rhs`
  NOR = capi::OXIDD_BOOLEAN_OPERATOR_NOR,
  /// Implication `lhs → rhs`
  IMP = capi::OXIDD_BOOLEAN_OPERATOR_IMP,
  /// Strict implication `lhs < rhs`
  IMP_STRICT = capi::OXIDD_BOOLEAN_OPERATOR_IMP_STRICT,
  // NOLINTEND(readability-identifier-naming)
};

/// Error details for labelling a variable with a name that is already in use
struct duplicate_var_name {
  /// Range of variables that have successfully been added
  ///
  /// If no fresh variables were requested, this is simply the empty range
  /// starting and ending at the current variable count.
  var_no_range_t added_vars;

  /// The duplicate name on error, or the empty string on success
  std::string name;

  /// Number of the already present variable with `name` on error, or
  /// `invalid_var_no` on success
  var_no_t present_var = invalid_var_no;

  /// Format the error in a human-readable fashion
  friend std::ostream &operator<<(std::ostream &stream,
                                  const duplicate_var_name &err) {
    stream << "the variable name '" << err.name
           << "' is already in use for variable number " << err.present_var;
    return stream;
  }
};

/// Assignment, similar to a `std::vector<opt_bool>`
///
/// This class has no copy constructor to avoid additional calls into the FFI.
/// Use `vec()` instead.
class assignment {
public:
  /// Container element type
  using value_type = opt_bool;
  /// Pointer to element type
  using pointer = value_type *;
  /// Pointer to const element type
  using const_pointer = const value_type *;
  /// Element reference type
  using reference = value_type &;
  /// Const element reference type
  using const_reference = const value_type &;
  /// Container iterator type
  using iterator = const_pointer;
  /// Container const iterator type
  using const_iterator = const_pointer;
  /// Container reverse iterator type
  using reverse_iterator = std::reverse_iterator<iterator>;
  /// Container const reverse iterator type
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  /// Container size type
  using size_type = size_t;
  /// Container iterator difference type
  using difference_type = ptrdiff_t;

private:
  capi::oxidd_assignment_t _assignment;

public:
  assignment(const assignment &) = delete; // use `vec()` to copy

  /// Construct an assignment from a `capi::oxidd_assignment_t`, taking
  /// ownership of it
  explicit assignment(capi::oxidd_assignment_t a) noexcept : _assignment(a) {}

  /// Move constructor: empties `other`
  assignment(assignment &&other) noexcept : _assignment(other._assignment) {
    other._assignment.data = nullptr;
    other._assignment.len = 0;
  }

  assignment &operator=(const assignment &) = delete;
  /// Move assignment operator: empties `rhs`
  assignment &operator=(assignment &&rhs) noexcept {
    assert(this != &rhs || !rhs._assignment.data);
    capi::oxidd_assignment_free(_assignment);
    _assignment = rhs._assignment;
    rhs._assignment.data = nullptr;
    rhs._assignment.len = 0;
    return *this;
  }

  ~assignment() noexcept { capi::oxidd_assignment_free(_assignment); }

  /// Get a pointer to the underlying data
  [[nodiscard]] const opt_bool *data() const noexcept {
    // NOLINTNEXTLINE(*-cast)
    return reinterpret_cast<const opt_bool *>(_assignment.data);
  }
  /// Get the number of elements in the assignment
  [[nodiscard]] size_t size() const noexcept { return _assignment.len; }

  /// Iterator to the beginning
  [[nodiscard]] iterator begin() const noexcept { return data(); }
  /// Iterator to the end
  [[nodiscard]] iterator end() const noexcept {
    return data() + size(); // NOLINT(*-pointer-arithmetic)
  }

  /// Reverse iterator to the beginning
  [[nodiscard]] reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  /// Reverse iterator to the end
  [[nodiscard]] reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  /// Access the element at `index`
  ///
  /// @param  index  The index, must be less than `size()`, otherwise the call
  ///                results in undefined behavior
  ///
  /// @returns  A reference to the element
  const opt_bool &operator[](size_t index) const noexcept {
    assert(index < size());
    return data()[index]; // NOLINT(*-pointer-arithmetic)
  }

  /// Get a `std::span<const opt_bool>` of the data
  [[nodiscard]] std::span<const opt_bool> span() const noexcept {
    return {data(), size()};
  }
  /// Deprecated alias for span()
  ///
  /// @deprecated  Use span() instead.
  [[nodiscard, deprecated("use span instead")]] std::span<const opt_bool>
  slice() const noexcept {
    return {data(), size()};
  }
  /// Conversion operator to `std::span<const opt_bool>`
  operator std::span<const opt_bool>() const noexcept { return span(); }

  /// Copy the data into a `std::vector<opt_bool>`
  [[nodiscard]] std::vector<opt_bool> vec() const { return {begin(), end()}; }
  /// Conversion operator to `std::vector<opt_bool>`
  operator std::vector<opt_bool>() const { return vec(); }
};

/// Helper function to get a size hint for an iterator in constant time
///
/// @returns  `end - begin` if `std::sized_sentinel_for<E, I>` and that value is
//            non-negative, otherwise 0
template <std::input_iterator I, std::sentinel_for<I> E>
inline std::size_t size_hint(I begin, E end) {
  using diff_ty = typename std::iter_difference_t<I>;

  if constexpr (std::sized_sentinel_for<E, I> &&
                std::is_convertible_v<diff_ty, std::ptrdiff_t>) {
    std::ptrdiff_t d = end - begin;
    return d >= 0 ? 0 : static_cast<std::size_t>(d);
  }
  return 0;
}

/// Helper function to get a size hint for a range in constant time
///
/// @returns  `std::ranges::size()` if supported, otherwise 0
template <std::ranges::input_range R> inline std::size_t size_hint(R &&range) {
  if constexpr (std::ranges::sized_range<R>) {
    using range_size_t = decltype(std::ranges::size(std::forward<R>(range)));
    if constexpr (std::is_convertible_v<range_size_t, std::size_t>)
      return std::ranges::size(std::forward<R>(range));
  }
  return 0;
}

/// Type that behaves like a pair `(T, U)`
///
/// - `std::tuple_element` is defined
/// - `std::tuple_size` is 2
/// - `std::get` can be used to extract the components
///
/// @see https://en.cppreference.com/w/cpp/utility/tuple/tuple-like
template <typename P, typename T, typename U>
concept pair_like =
    std::convertible_to<std::tuple_element_t<0, P>, T> &&
    std::convertible_to<std::tuple_element_t<1, P>, U> &&
    std::tuple_size_v<std::remove_cvref_t<P>> == 2 && requires(P pair) {
      { std::get<0>(pair) } -> std::convertible_to<T>;
      { std::get<1>(pair) } -> std::convertible_to<U>;
    };

} // namespace util

} // namespace oxidd
