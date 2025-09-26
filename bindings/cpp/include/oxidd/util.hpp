/// @file   util.hpp
/// @brief  Primitives and utilities

#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ostream>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>
#include <version>

#include <oxidd/capi.h>
#include <oxidd/compat.hpp>

// spell-checker:dictionaries dddmp

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

/// General-purpose error type with human-readable error messages
class error {
  capi::oxidd_error_t _error;

  error(capi::oxidd_error_t error) : _error(error) {}

public:
  /// Construct an `error` from a C API struct
  ///
  /// This will take ownership of `error`, i.e., this object will take care of
  /// calling `capi::oxidd_error_free()` (unless ownership is moved away via a
  /// move constructor, …).
  ///
  /// Time complexity: O(1)
  ///
  /// @param  c_error  The C API struct to be wrapped
  ///
  /// @returns  The error object wrapping `c_error`
  [[nodiscard]] static error from_c_api(capi::oxidd_error_t c_error) noexcept {
    return c_error;
  }

  /// Get the wrapped C API struct
  ///
  /// This method keeps ownership in `this`. To transfer ownership away, use
  //  `release_to_c_api()`.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The wrapped manager struct
  [[nodiscard]] constexpr capi::oxidd_error_t to_c_api() const noexcept {
    return _error;
  }
  /// Get the wrapped C API struct and release ownership of it
  ///
  /// You should eventually call `capi::oxidd_error_free()` on the result.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The wrapped manager struct
  ///
  /// @see  `to_c_api()` for a version that does not transfer ownership.
  constexpr capi::oxidd_error_t release_to_c_api() noexcept {
    const capi::oxidd_error_t tmp = _error;
    _error._msg_cap = 0;
    return _error;
  }

  /// Copy constructor
  error(const error &other) noexcept
      : _error(capi::oxidd_error_clone(&other._error)) {}
  /// Move constructor
  ///
  /// Note that calling the move constructor will transfer ownership of the
  /// error message to the move-constructed object, but not clear the message in
  /// `other`. This means that you will still be able to retrieve the error
  /// message from `other`, but when the object having ownership gets
  /// destructed, the `std::string_view` returned by `message` becomes a
  /// dangling reference.
  constexpr error(error &&other) noexcept : _error(other._error) {
    other._error._msg_cap = 0;
  }
  ~error() noexcept {
    if (_error._msg_cap != 0)
      capi::oxidd_error_free(_error);
  }

  /// Copy assignment operator
  error &operator=(const error &other) noexcept {
    _error = capi::oxidd_error_clone(&other._error);
    return *this;
  }
  /// Move assignment operator
  ///
  /// Note that executing a move assignment will transfer ownership of the error
  /// message to `this`, but not clear the message in `other`. This means that
  /// you will still be able to retrieve the error message from `other`, but
  /// when the object having ownership gets destructed, the `std::string_view`
  /// returned by `message` becomes a dangling reference.
  constexpr error &operator=(error &&other) noexcept {
    _error = other._error;
    other._error._msg_cap = 0;
    return *this;
  }

  /// Get the human-readable error message
  ///
  /// Note that the returned string view is only valid until this object gets
  /// destructed.
  [[nodiscard]] constexpr std::string_view message() const noexcept {
    return {_error.msg, _error.msg_len};
  }
  /// Conversion to a `std::string_view`, returns the same as `message()`
  constexpr operator std::string_view() const noexcept { return message(); }

  friend constexpr bool operator==(error &x, error &y) {
    return x.message() == y.message();
  }

  /// Write the error message to `stream`
  friend std::ostream &operator<<(std::ostream &stream, const error &err) {
    return stream << err.message();
  }
};

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
  /// Deprecated alias for `span()`
  ///
  /// @deprecated  Use `span()` instead.
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

/// DDDMP header loaded as the first step of an import process
class dddmp_file {
  capi::oxidd_dddmp_file_t *_file = nullptr;

  explicit dddmp_file(capi::oxidd_dddmp_file_t *file) noexcept : _file(file) {}

  template <typename S,
            capi::oxidd_str_t Fn(const capi::oxidd_dddmp_file_t *, S)>
  class str_iter {
    const capi::oxidd_dddmp_file_t *_file = nullptr;
    S _i = 0;

    explicit str_iter(const capi::oxidd_dddmp_file_t *file, S i)
        : _file(file), _i(i) {}

  public:
    explicit str_iter() {} // NOLINT(*-default)
    explicit str_iter(const dddmp_file &file, S i)
        : _file(file.to_c_api()), _i(i) {}

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::string_view;
    using reference = std::string_view;

    reference operator*() const {
      capi::oxidd_str_t s(Fn(_file, _i));
      return {s.ptr, s.len};
    }

    str_iter &operator++() {
      ++_i;
      return *this;
    }
    str_iter operator++(int) {
      str_iter tmp = *this;
      ++(*this);
      return tmp;
    }
    str_iter &operator--() {
      --_i;
      return *this;
    }
    str_iter operator--(int) {
      str_iter tmp = *this;
      --(*this);
      return tmp;
    }

    friend str_iter operator+(const str_iter &lhs, difference_type rhs) {
      return str_iter(lhs._file, lhs._i + rhs);
    }
    friend str_iter operator+(difference_type lhs, const str_iter &rhs) {
      return rhs + lhs;
    }
    str_iter &operator+=(difference_type i) { return *this = *this + i; }

    friend difference_type operator-(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i - rhs._i;
    }
    friend str_iter operator-(const str_iter &lhs, difference_type rhs) {
      return str_iter(lhs._file, lhs._i - rhs);
    }
    friend str_iter operator-(difference_type lhs, const str_iter &rhs) {
      return str_iter(rhs._file, lhs - rhs._i);
    }
    str_iter &operator-=(difference_type i) { return *this = *this + i; }

    std::string_view operator[](difference_type i) const {
      return *(*this + i);
    }

    friend bool operator==(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i == rhs._i;
    }
    friend bool operator!=(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i != rhs._i;
    }
    friend bool operator<=(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i <= rhs._i;
    }
    friend bool operator<(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i < rhs._i;
    }
    friend bool operator>=(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i >= rhs._i;
    }
    friend bool operator>(const str_iter &lhs, const str_iter &rhs) {
      assert(lhs._file == rhs._file);
      return lhs._i > rhs._i;
    }
  };

  using var_name_iterator = str_iter<var_no_t, &capi::oxidd_dddmp_var_name>;
  using root_name_iterator = str_iter<size_t, &capi::oxidd_dddmp_root_name>;
  static_assert(std::random_access_iterator<var_name_iterator>);
  static_assert(std::random_access_iterator<root_name_iterator>);

public:
  dddmp_file(const dddmp_file &other) = delete;
  /// Move constructor, invalidates `other`
  dddmp_file(dddmp_file &&other) noexcept : _file(other._file) {
    other._file = nullptr;
  }

  ~dddmp_file() {
    if (_file != nullptr)
      capi::oxidd_dddmp_close(_file);
  }

  dddmp_file &operator=(const dddmp_file &other) = delete;
  /// Move assignment operator, invalidates `other`
  dddmp_file &operator=(dddmp_file &&other) noexcept {
    _file = other._file;
    other._file = nullptr;
    return *this;
  }

  /// Load a DDDMP header from file from the file at `path`
  [[nodiscard]] static compat::expected<dddmp_file, error>
  open(const std::string_view path) noexcept {
    capi::oxidd_error_t err;
    capi::oxidd_dddmp_file_t *file =
        capi::oxidd_dddmp_open(path.data(), path.size(), &err);
    if (file != nullptr)
      return dddmp_file(file);
    return compat::unexpected(error::from_c_api(err));
  }

  /// Construct a `dddmp_file` from a C API struct
  ///
  /// This will take ownership of `error`, i.e., this object will take care of
  /// calling `capi::oxidd_error_free()` (unless ownership is moved away via a
  /// move constructor, …).
  ///
  /// Time complexity: O(1)
  ///
  /// @param  c_file  The C API struct to be wrapped
  ///
  /// @returns  The error object wrapping `c_error`
  [[nodiscard]] static dddmp_file
  from_c_api(capi::oxidd_dddmp_file_t *c_file) noexcept {
    return dddmp_file(c_file);
  }
  /// Get the wrapped C API struct
  ///
  /// This method keeps ownership in `this`. To transfer ownership away, use
  //  `release_to_c_api()`.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The wrapped manager struct
  [[nodiscard]] constexpr capi::oxidd_dddmp_file_t *to_c_api() const noexcept {
    return _file;
  }
  /// Get the wrapped C API struct and release ownership of it
  ///
  /// You should eventually call `capi::oxidd_error_free()` on the result.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The wrapped manager struct
  ///
  /// @see  `to_c_api()` for a version that does not transfer ownership.
  [[nodiscard]] constexpr capi::oxidd_dddmp_file_t *
  release_to_c_api() noexcept {
    capi::oxidd_dddmp_file_t *f = _file;
    _file = nullptr;
    return f;
  }

  /// Get the name of the decision diagram
  ///
  /// Corresponds to the DDDMP `.dd` field.
  ///
  /// @returns  The name, or `std::nullopt` if the field is not present.
  [[nodiscard]] std::optional<std::string_view> diagram_name() const noexcept {
    assert(_file);
    const capi::oxidd_str_t name = capi::oxidd_dddmp_diagram_name(_file);
    return name.len != 0 ? std::optional(std::string_view(name.ptr, name.len))
                         : std::nullopt;
  }

  /// Get the number of nodes in the dumped decision diagram
  ///
  /// Corresponds to the DDDMP `.nnodes` field.
  [[nodiscard]] size_t num_nodes() const noexcept {
    assert(_file);
    return capi::oxidd_dddmp_num_nodes(_file);
  }

  /// Get the number of all variables in the exported decision diagram
  ///
  /// Corresponds to the DDDMP `.nvars` field.
  [[nodiscard]] var_no_t num_vars() const noexcept {
    assert(_file);
    return capi::oxidd_dddmp_num_vars(_file);
  }

  /// Get the number of variables in the true support of the decision diagram
  ///
  /// Corresponds to the DDDMP `.nsuppvars` field.
  [[nodiscard]] var_no_t num_support_vars() const noexcept {
    assert(_file);
    return capi::oxidd_dddmp_num_support_vars(_file);
  }

  /// Get the variables in the true support of the decision diagram
  ///
  /// Concretely, these are indices of the original variable numbering. Hence,
  /// the returned slice contains [`DumpHeader::num_support_vars()`] integers in
  /// strictly ascending order.
  ///
  /// Example: Consider a decision diagram that was created with the variables
  /// `x`, `y`, and `z`, in this order (`x` is the top-most variable). Suppose
  /// that only `y` and `z` are used by the dumped functions. Then, the
  /// returned slice is `[1, 2]`, regardless of any subsequent reordering.
  ///
  /// Corresponds to the DDDMP `.ids` field.
  [[nodiscard]] std::span<const var_no_t> support_vars() const noexcept {
    assert(_file);
    const capi::oxidd_slice_var_no vars = capi::oxidd_dddmp_support_vars(_file);
    return {vars.ptr, vars.len};
  }

  /// Get the support variables' order
  ///
  /// The returned slice is always [`DumpHeader::num_support_vars()`] elements
  /// long and represents a mapping from positions to variable numbers.
  ///
  /// Example: Consider a decision diagram that was created with the variables
  /// `x`, `y`, and `z` (`x` is the top-most variable). The variables were
  /// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
  /// the dumped functions. Then, the returned slice is `[2, 1]`.
  [[nodiscard]] std::span<const var_no_t> support_var_order() const noexcept {
    assert(_file);
    const capi::oxidd_slice_var_no vars =
        capi::oxidd_dddmp_support_var_order(_file);
    return {vars.ptr, vars.len};
  }

  /// Get the mapping from support variables to levels
  ///
  /// The returned slice is always [`DumpHeader::num_support_vars()`]
  /// elements long. If the value at the `i`th index is `l`, then the `i`th
  /// support variable is at level `l` in the dumped decision diagram. By the
  /// `i`th support variable, we mean the variable `header.support_vars()[i]`
  /// in the original numbering.
  ///
  /// Example: Consider a decision diagram that was created with the variables
  /// `x`, `y`, and `z` (`x` is the top-most variable). The variables were
  /// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
  /// the dumped functions. Then, the returned slice is `[2, 0]`.
  ///
  /// Corresponds to the DDDMP `.permids` field.
  [[nodiscard]] std::span<const level_no_t>
  support_var_to_level() const noexcept {
    assert(_file);
    const capi::oxidd_slice_level_no levels =
        capi::oxidd_dddmp_support_var_to_level(_file);
    return {levels.ptr, levels.len};
  }

  /// Get whether the header contains variable names
  [[nodiscard]] bool has_var_names() const noexcept {
    assert(_file);
    return capi::oxidd_dddmp_has_var_names(_file);
  }

  /// Get an iterator referencing the first variable name
  ///
  /// @returns  The beginning `std::random_access_iterator`
  ///
  /// @see  `var_names()` for more details
  [[nodiscard]] var_name_iterator var_names_begin() const {
    return var_name_iterator(*this, 0);
  }
  /// Get the sentinel iterator for the variable names
  ///
  /// @returns  The sentinel `std::random_access_iterator`
  ///
  /// @see  `var_names()` for more details
  [[nodiscard]] var_name_iterator var_names_end() const {
    return var_name_iterator(*this, num_vars());
  }
  /// Get the variable names
  ///
  /// The variable names are read from the DDDMP `.varnames` field, but
  /// `.orderedvarnames` and `.suppvarnames` are also considered if one of the
  /// fields is missing. All variable names are non-empty unless only
  /// `.suppvarnames` is given in the input (in which case only the names of
  /// support variables are non-empty). The return value is only `None` if
  /// neither of `.varnames`, `.orderedvarnames`, and `.suppvarnames` is present
  /// in the input.
  ///
  /// @returns  A `std::ranges::random_access_range` over the variable names
  [[nodiscard]] std::ranges::subrange<var_name_iterator> var_names() const {
    return {var_names_begin(), var_names_end()};
  }

  /// Get the number of roots
  ///
  /// The `manager::import_dddmp()` method returns this number of roots on
  /// success. Corresponds to the DDDMP `.nroots` field.
  [[nodiscard]] size_t num_roots() const noexcept {
    assert(_file);
    return capi::oxidd_dddmp_num_roots(_file);
  }

  /// Get whether the header contains functions names
  [[nodiscard]] bool has_root_names() const noexcept {
    assert(_file);
    return capi::oxidd_dddmp_has_root_names(_file);
  }

  /// Get an iterator referencing the first root name
  ///
  /// @returns  The beginning `std::random_access_iterator`
  ///
  /// @see  `root_names()` for more details
  [[nodiscard]] root_name_iterator root_names_begin() const {
    return root_name_iterator(*this, 0);
  }
  /// Get the sentinel iterator for the root names
  ///
  /// @returns  The sentinel `std::random_access_iterator`
  ///
  /// @see  `root_names()` for more details
  [[nodiscard]] root_name_iterator root_names_end() const {
    return root_name_iterator(*this, num_vars());
  }
  /// Get the root names
  ///
  /// The names are read from the DDDMP `.rootnames` field.
  ///
  /// @returns  A `std::ranges::random_access_range` over the root names
  [[nodiscard]] std::ranges::subrange<root_name_iterator> root_names() const {
    return {root_names_begin(), root_names_end()};
  }
};

/// DDDMP format version version
enum class dddmp_version : uint8_t {
  /// Version 2.0, bundled with [CUDD] 3.0
  ///
  /// [CUDD]: https://github.com/cuddorg/cudd
  v2_0 = capi::oxidd_dddmp_version::OXIDD_DDDMP_VERSION_2_0,
  /// Version 3.0, used by [BDDSampler] and [Logic2BDD]
  ///
  /// [BDDSampler]: https://github.com/davidfa71/BDDSampler
  /// [Logic2BDD]: https://github.com/davidfa71/Extending-Logic
  v3_0 = capi::oxidd_dddmp_version::OXIDD_DDDMP_VERSION_3_0,
};

/// Settings for exporting a decision diagram in the DDDMP format
struct dddmp_export_settings {
  /// DDDMP format version for the export
  dddmp_version version = dddmp_version::v2_0;
  /// Whether to enforce the human-readable ASCII mode
  ///
  /// If `false`, the more compact binary format will be used, provided that
  /// it is supported for the respective decision diagram kind. Currently,
  /// binary mode is supported for BCDDs only.
  bool ascii = false;
  /// Whether to enable strict mode
  ///
  /// The DDDMP format imposes some restrictions on diagram, variable, and
  /// function names:
  ///
  /// - None of them may contain ASCII control characters (e.g., line breaks).
  /// - Variable and function names must not contain spaces either.
  /// - Variable and function names must not be empty. (However, it is
  ///   possible to not export any variable or function names at all.)
  ///
  /// In the diagram name, control characters will be replaced by spaces. In
  /// variable and function names, an underscore (`_`) is used as the
  /// replacement character. When using any of the `*_dddmp_export_with_names`
  /// functions, empty function names are replaced by `_f{i}`, where `{i}`
  /// stands for the position in the iterator. Empty variable names are
  /// replaced by `_x{i}`. To retain uniqueness, as many underscores are
  /// added to the prefix as there are in the longest prefix over all
  /// present variable names.
  ///
  /// However, since such relabelling may lead to unexpected results, there is
  /// a strict mode. In strict mode, no variable names will be exported unless
  /// all variables are named. Additionally, an error will be generated upon
  /// any replacement. The error will not be propagated immediately but only
  /// after the file was written. This should simplify inspecting the error
  /// and also serve as a checkpoint for very long-running computations.
  bool strict = true;
  /// Name of the decision diagram
  std::string_view diagram_name;
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
