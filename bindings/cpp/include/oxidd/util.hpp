/// @file  util.hpp
/// @brief Primitives and utilities

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>
#include <version>

#ifdef __cpp_lib_concepts
#include <concepts>
#endif // __cpp_lib_concepts

#ifdef __cpp_lib_ranges
#include <ranges>
#endif // __cpp_lib_ranges

#include <oxidd/capi.h>

/// OxiDD's main namespace
namespace oxidd {

/// Level number type
using level_no_t = capi::oxidd_level_no_t;

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

/// View into a contiguous sequence, roughly equivalent to Rust's `&[T]` type
///
/// This type is trivially copyable and should be passed by value.
// Inspired by LLVM's ArrayRef class
template <typename T> class slice {
public:
  /// Container element type
  using value_type = T;
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
  const T *_data = nullptr;
  size_t _len = 0;

public:
  /// @name Constructors
  /// @{

  /// Construct an empty slice
  constexpr slice() = default;

  /// Construct an empty slice from `std::nullopt`
  constexpr slice(std::nullopt_t /*unused*/) {}

  /// Construct a slice from a pointer and length
  constexpr slice(const T *data, size_t len) : _data(data), _len(len) {}

  /// Construct a slice from a range
  constexpr slice(const T *begin, const T *end)
      : _data(begin), _len(end - begin) {
    assert(begin <= end);
  }

  /// Construct a slice from a single element
  constexpr slice(const T &element) : _data(&element), _len(1) {}

  /// Construct a slice from a `std::vector<T>`
  template <typename A>
  slice(const std::vector<T, A> &vec) : _data(vec.data()), _len(vec.size()) {}

  /// Construct a slice from a `std::array<T>`
  template <size_t N>
  constexpr slice(const std::array<T, N> &array)
      : _data(array.data()), _len(array.size()) {}

  /// Construct a slice from a C array
  template <size_t N>
  constexpr slice(const T (&array)[N]) // NOLINT(*-c-arrays)
      : _data(array), _len(N) {}

  /// Construct a slice from a `std::initializer_list<T>`
  constexpr slice(const std::initializer_list<T> &list)
      : _data(list.begin() == list.end() ? (const T *)nullptr : list.begin()),
        _len(list.size()) {}

  /// @}

  /// Get this slice's length
  [[nodiscard]] constexpr size_t size() const noexcept { return _len; }
  /// Check if this slice is empty
  [[nodiscard]] constexpr bool empty() const noexcept { return _len == 0; }

  /// Get the data pointer
  [[nodiscard]] constexpr const T *data() const noexcept { return _data; }

  /// Iterator to the beginning
  [[nodiscard]] constexpr iterator begin() const noexcept { return _data; }
  /// Iterator to the end
  [[nodiscard]] constexpr iterator end() const noexcept { return _data + _len; }

  /// Reverse iterator to the beginning
  [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator(end());
  }
  /// Reverse iterator to the end
  [[nodiscard]] constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator(begin());
  }

  /// Get the first element of a non-empty slice
  [[nodiscard]] constexpr const T &front() const noexcept {
    assert(_len != 0 && "slice must be non-empty");
    return _data[0]; // NOLINT(*-pointer-arithmetic)
  }
  /// Get the last element of a non-empty slice
  [[nodiscard]] constexpr const T &back() const noexcept {
    assert(_len != 0 && "slice must be non-empty");
    return _data[_len - 1]; // NOLINT(*-pointer-arithmetic)
  }

  /// Access the element at `index`
  ///
  /// @param  index  The index, must be less than `size()`, otherwise the call
  ///                results in undefined behavior
  ///
  /// @returns  A reference to the element
  constexpr const T &operator[](size_t index) const noexcept {
    assert(index < _len && "index out of bounds");
    return _data[index]; // NOLINT(*-pointer-arithmetic)
  }

  /// Get a subslice starting at index `start` (inclusively) and ending at
  /// index `end` (exclusively)
  [[nodiscard]] constexpr slice subslice(size_t start,
                                         size_t end) const noexcept {
    assert(start <= end);
    assert(end <= _len);
    return slice(_data + start, end - start);
  }

  /// Copy the data into a `std::vector<T>`
  [[nodiscard]] std::vector<T> vec() const {
    return std::vector<T>(begin(), end());
  }
  /// Conversion operator to `std::vector<T>`
  operator std::vector<T>() const { return vec(); }
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

  /// Get a `slice<opt_bool>` of the data
  [[nodiscard]] util::slice<opt_bool> slice() const noexcept {
    return {data(), size()};
  }
  /// Conversion operator to `slice<opt_bool>`
  operator util::slice<opt_bool>() const noexcept { return slice(); }

  /// Copy the data into a `std::vector<opt_bool>`
  [[nodiscard]] std::vector<opt_bool> vec() const { return {begin(), end()}; }
  /// Conversion operator to `std::vector<opt_bool>`
  operator std::vector<opt_bool>() const { return vec(); }
};

/// @see `size_hint(IT, IT, std::input_iterator_tag)`
template <typename IT>
inline std::size_t size_hint(IT begin, IT end,
                             std::forward_iterator_tag /*tag*/) {
  return std::distance(begin, end);
}

/// Helper function to get a size hint for an iterator
///
/// This function uses `std::distance()` to compute the size hint if the
/// iterator allows for multiple passes (i.e., it is a `std::forward_iterator`),
/// and otherwise simply returns 0.
///
/// This function uses tag dispatch. The tag argument must be
/// `std::iterator_traits<IT>::iterator_category()`.
///
/// Behavior is undefined if `end` is not reachable from `begin`. Moreover, if
/// `IT` is a random access iterator, then behavior is also undefined if `begin`
/// is not reachable from `end`.
template <typename IT>
inline std::size_t size_hint(IT /*begin*/, IT /*end*/,
                             std::input_iterator_tag /*tag*/) {
  return 0;
}

#ifdef __cpp_lib_ranges

/// Helper function to get a size hint for a range
///
/// This function uses `std::ranges::distance()` to compute the size hint if the
/// iterator allows for multiple passes (i.e., it is a
/// `std::ranges::forward_range` or `std::ranges::sized_range`), and otherwise
/// simply returns 0.
template <std::ranges::input_range R> inline std::size_t size_hint(R &&range) {
  if constexpr (std::ranges::forward_range<R> || std::ranges::sized_range<R>) {
    const auto d = std::ranges::distance(std::forward<R>(range));
    return d >= 0 ? d : 0; // d is signed
  } else {
    return 0;
  }
}

#endif // __cpp_lib_ranges

#ifdef __cpp_lib_concepts

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

#endif // __cpp_lib_concepts

/// @cond

namespace detail {

extern "C" inline void *oxidd_callback_helper(void *f) {
  return (*static_cast<std::function<void *()> *>(f))();
}

template <typename M, typename R>
R run_in_worker_pool(void *(*run_fn)(M, void *(void *), void *), M manager,
                     const std::function<R()> f) {
  if constexpr (std::is_void_v<R>) {
    const std::function<void *()> wrapper([f = std::move(f)]() -> void * {
      f();
      return nullptr;
    });
    run_fn(manager, oxidd_callback_helper, (void *)(&wrapper));
  } else if constexpr (std::is_reference_v<R>) {
    const std::function<void *()> wrapper(
        [f = std::move(f)]() -> void * { return (void *)(&f()); });
    return *static_cast<std::remove_reference_t<R> *>(
        run_fn(manager, oxidd_callback_helper, (void *)(&wrapper)));
  } else if constexpr (std::is_pointer_v<R>) {
    const std::function<void *()> wrapper(
        [f = std::move(f)]() -> void * { return (void *)f(); });
    return static_cast<R>(
        run_fn(manager, oxidd_callback_helper, (void *)(&wrapper)));
  } else {
    // union type to not call the R's default constructor (or even require one)
    union {
      R v;
    } return_val;
    const std::function<void *()> wrapper(
        [f = std::move(f), &return_val]() -> void * {
          return_val.v = f();
          return nullptr;
        });
    run_fn(manager, oxidd_callback_helper, (void *)(&wrapper));
    return return_val.v;
  }
}

} // namespace detail

/// @endcond

} // namespace util

} // namespace oxidd
