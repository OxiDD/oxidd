/// @file   bridge.hpp
/// @brief  Bridge between the C and the C++ APIs

#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <oxidd/compat.hpp>
#include <oxidd/util.hpp>

#ifndef OXIDD_ENABLE_CXX
#error "OxiDD must be built with the `cpp` feature to use the C++ bindings"
#endif

/// Bridge between OxiDD's C and C++ APIs
///
/// This names contains multiple class templates as building blocks for an
/// object-oriented API. The templates leverage the
/// <a href="https://en.cppreference.com/w/cpp/language/crtp.html">curiously
/// recurring template pattern (CRTP)</a>, where the derived class provides a
/// "table" listing the respective functions of the C API.
namespace oxidd::bridge {

/// @cond

/// Implementation details
///
/// This mainly contains classes to allow the conversion from C++ iterators to
/// C ABI iterators (as defined by OxiDD) and C++ callables to C ABI callbacks.
namespace detail {

template <bool Cond, typename T>
using empty_if = std::conditional_t<Cond, std::monostate, T>;

// Note: We need to pass an `extern "C"` function pointer to the Rust side since
// the ABI (calling convention) of functions with C++ linkage may differ from
// those with C linkage. In particular, the C++ standard states: “Calling a
// function through an expression whose function type is different from the
// function type of the called function's definition results in undefined
// behavior” (https://timsong-cpp.github.io/cppwp/n4868/expr.call#6, more recent
// versions allow conversions from `noexcept` functions pointers to regular
// function pointers). “Two function types with different language linkages are
// distinct types even if they are otherwise identical”
// (https://timsong-cpp.github.io/cppwp/n4868/dcl.link#1).
//
// Unfortunately, `extern "C"` can only appear in namespace scope, and we cannot
// template functions with C linkage. Our workaround is to have some helper
// functions with C linkage that call the function passed via the data pointer
// with C++ linkage.

extern "C" capi::oxidd_size_hint_t oxidd_size_hint_callback_helper(void *data);

template <typename T> struct iter_adapter {};

struct c_iter_vtable_base {
  capi::oxidd_size_hint_t (*size_hint)(c_iter_vtable_base *);
};

template <typename T> struct c_iter_vtable : public c_iter_vtable_base {
  iter_adapter<T>::c_opt_value_t (*next)(c_iter_vtable *);
};

/// Bridge between a C++ iterator and `iter_adapter<T>::c_iter_t`
template <typename T, std::input_iterator I, std::sentinel_for<I> E = I,
          bool NoExcept = noexcept(std::declval<I>() != std::declval<E>()) &&
                          noexcept(++(std::declval<I &>())) &&
                          noexcept(*(std::declval<I>())) &&
                          (std::is_lvalue_reference_v<std::iter_value_t<I>> ||
                           std::is_nothrow_move_constructible_v<
                               std::remove_cvref_t<std::iter_value_t<I>>>)>
class c_iter : c_iter_vtable<T> {
  using adapter = iter_adapter<T>;

  static adapter::c_opt_value_t _next_fn(c_iter_vtable<T> *ctx_base) {
    auto *ctx = static_cast<c_iter *>(ctx_base);
    try {
      if (ctx->_current.has_value()) {
        ctx->_current.reset();
        ++(ctx->_begin);
      }
      if (ctx->_begin != ctx->_end) {
        ctx->_current.emplace(*ctx->_begin);
        if constexpr (std::is_lvalue_reference_v<iter_t>)
          return {.is_some = true, .value = adapter::map(ctx->_current->get())};
        else
          return {.is_some = true, .value = adapter::map(*ctx->_current)};
      }
    } catch (...) {
      if constexpr (NoExcept) {
        std::terminate();
      } else {
        ctx->_exception = std::current_exception();
      }
    }
    return {.is_some = false}; // value is allowed to be uninitialized
  }

  constexpr static decltype(c_iter_vtable_base::size_hint)
  _make_size_hint_fn() {
    using diff_ty = typename std::iter_difference_t<I>;
    if constexpr (std::sized_sentinel_for<E, I> &&
                  std::is_convertible_v<diff_ty, std::ptrdiff_t>) {
      return
          [](c_iter_vtable_base *ctx_base) noexcept -> capi::oxidd_size_hint_t {
            auto *ctx = static_cast<c_iter *>(ctx_base);
            try {
              const std::ptrdiff_t d = ctx->_end - ctx->_begin;
              if (d >= 0) {
                const auto u = static_cast<size_t>(d);
                return {.lower = u, .upper = u};
              }
            } catch (...) { // NOLINT(*-empty-catch)
            }
            return {.lower = 0, .upper = std::numeric_limits<size_t>::max()};
          };
    }
    return nullptr;
  }

private:
  [[no_unique_address]] empty_if<NoExcept, std::exception_ptr> _exception;

  // We store the current C++ iterator value in the `_current` field because
  // the values returned by `operator*()` of the iterator may be not trivially
  // destructible (e.g., a `bdd_function` returned from
  // `std::views::transform`). In this case, we need to make sure that the value
  // lives at least until the next *next* operation. Also, for moving iterators
  // (i.e., ones that return rvalue references), we need to store the temporary
  // (hence the `std::remove_cvref_t` in the *else* case of `current_t`). In
  // case the iterator values are lvalue references, we could use them directly
  // considering their lifetimes. To store them in a `std::optional`, however,
  // we need to wrap them in a `std::reference_wrapper`.
  using iter_t = std::iter_reference_t<I>;
  using current_t = std::conditional_t<
      std::is_lvalue_reference_v<iter_t>,
      std::reference_wrapper<std::remove_reference_t<iter_t>>,
      std::remove_cvref_t<iter_t>>;
  std::optional<current_t> _current;

  I _begin;
  E _end;

public:
  c_iter() = delete;
  c_iter(I begin, E end) : _begin(std::move(begin)), _end(std::move(end)) {
    // NOLINTNEXTLINE(*-member-initializer)
    c_iter_vtable_base::size_hint = _make_size_hint_fn();
    c_iter_vtable<T>::next = &_next_fn;
  }

  template <typename F, typename... Args>
  std::invoke_result_t<F, Args..., typename adapter::c_iter_t>
  use_in_invoke(F &&f, Args &&...args)
    requires std::invocable<F, Args..., typename adapter::c_iter_t>
  {
    typename adapter::c_iter_t c_iter{
        .next = adapter::helper,
        .size_hint = c_iter_vtable_base::size_hint == nullptr
                         ? nullptr
                         : oxidd_size_hint_callback_helper,
        .context = this};
    const auto result =
        std::invoke(std::forward<F>(f), std::forward<Args>(args)..., c_iter);

    if constexpr (!NoExcept) {
      if (_exception)
        // Note that moving out of `_exception` does not necessarily reset
        // `_exception`. At least, there is no statement on this aspect on
        // cppreference.com, and, e.g., libc++ also does not reset the pointer
        // upon move. Hence, we use `std::exchange` instead.
        std::rethrow_exception(std::exchange(_exception, {}));
    }
    return result;
  }

  /// Returns `true` iff a *next* operation has been executed and the iterator
  /// has not reached its end
  bool has_current() noexcept { return _current.has_value(); }

  /// Get the element that has been returned by the most recent *next* operation
  ///
  /// Undefined behavior if `has_current()` is `false`.
  const std::remove_cvref_t<iter_t> &current() noexcept { return *_current; }
};

extern "C" inline capi::oxidd_opt_str_t
oxidd_iter_str_callback_helper(void *data);

template <> struct iter_adapter<capi::oxidd_str_t> {
  using c_iter_t = capi::oxidd_iter_str_t;
  using c_opt_value_t = capi::oxidd_opt_str_t;

  static constexpr auto helper = oxidd_iter_str_callback_helper;

  static constexpr capi::oxidd_str_t map(std::string_view val) noexcept {
    return {.ptr = val.data(), .len = val.size()};
  }
};

extern "C" inline capi::oxidd_opt_str_t
oxidd_iter_str_callback_helper(void *data) {
  auto *ctx = static_cast<c_iter_vtable<capi::oxidd_str_t> *>(data);
  return ctx->next(ctx);
}

extern "C" inline capi::oxidd_size_hint_t
oxidd_size_hint_callback_helper(void *data) {
  auto *ctx = static_cast<c_iter_vtable_base *>(data);
  return ctx->size_hint(ctx);
}

extern "C" void *oxidd_callback_helper(void *data);

struct c_callback_vtable {
  void *(*func_wrapper)(c_callback_vtable *);
};

/// Bridge between a C++ callable and a C ABI callback
template <std::invocable C, bool NoExcept = std::is_nothrow_invocable_v<C>>
  requires std::movable<std::invoke_result_t<C>> ||
           std::is_void_v<std::invoke_result_t<C>>
class c_callback : c_callback_vtable {
  static void *_func_wrapper_fn(c_callback_vtable *ctx_base) {
    auto *ctx = static_cast<c_callback *>(ctx_base);
    try {
      if constexpr (std::is_void_v<result_t>) {
        std::invoke(ctx->_func);
      } else if constexpr (std::is_reference_v<result_t>) {
        return static_cast<void *>(&std::invoke(ctx->_func));
      } else if constexpr (std::is_pointer_v<result_t>) {
        return static_cast<void *>(std::invoke(ctx->_func));
      } else {
        std::construct_at(&ctx->_result.value, std::invoke(ctx->_func));
      }
    } catch (...) {
      if constexpr (NoExcept) {
        std::terminate();
      } else {
        ctx->_exception = std::current_exception();
      }
    }
    return nullptr;
  };

  using callback_helper_t = decltype(oxidd_callback_helper);
  using result_t = std::invoke_result_t<C>;
  static constexpr bool direct_return = std::is_void_v<result_t> ||
                                        std::is_pointer_v<result_t> ||
                                        std::is_reference_v<result_t>;

  union result_container { // NOLINT(*-special-member-functions)
    empty_if<direct_return, result_t> value;
    // have a default constructor even if return_t doesn't
    result_container() {}
    ~result_container() {}
  };

  [[no_unique_address]] empty_if<NoExcept, std::exception_ptr> _exception;
  [[no_unique_address]] empty_if<direct_return, result_container> _result;

  C _func;

public:
  c_callback() : c_callback_vtable(&_func_wrapper_fn) {
    if constexpr (std::is_pointer_v<C>)
      _func = nullptr;
  }
  c_callback(const C &func) noexcept(std::is_nothrow_copy_constructible_v<C>)
    requires std::is_copy_constructible_v<C>
      : c_callback_vtable{&_func_wrapper_fn}, _func(func) {}
  c_callback(C &&func) noexcept(std::is_nothrow_move_constructible_v<C>)
      : c_callback_vtable{&_func_wrapper_fn}, _func(std::move(func)) {}

  template <typename F, typename... Args>
  result_t use_in_invoke(F &&f, Args &&...args) noexcept(
      NoExcept &&
      std::is_nothrow_invocable_r_v<void *, F, Args..., callback_helper_t,
                                    void *> &&
      std::is_nothrow_move_constructible_v<result_t>)
    requires std::is_invocable_r_v<void *, F, Args..., callback_helper_t,
                                   void *>
  {
    void *result_ptr =
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...,
                    oxidd_callback_helper, this);

    if constexpr (!NoExcept) {
      if (_exception)
        std::rethrow_exception(std::exchange(_exception, {}));
    }
    if constexpr (std::is_void_v<result_t>) {
      return;
    } else if constexpr (std::is_reference_v<result_t>) {
      return *static_cast<std::remove_reference_t<result_t> *>(result_ptr);
    } else if constexpr (std::is_pointer_v<result_t>) {
      return static_cast<result_t>(result_ptr);
    } else {
      const result_t r = std::move(_result.value);
      if constexpr (!std::is_trivially_destructible_v<result_t>) {
        std::destroy_at(&_result.value);
      }
      return r;
    }
  }
};

extern "C" inline void *oxidd_callback_helper(void *data) {
  auto *ctx = static_cast<c_callback_vtable *>(data);
  return ctx->func_wrapper(ctx);
}

/// Cast a `const std::pair<var_no_t, bool> *` into a
/// `const capi::oxidd_var_no_bool_pair_t *`
///
/// Note that this is type punning, and directly reading/writing the pair
/// elements via the returned pointer from C/C++ is Undefined Behavior due to
/// their strict aliasing rules (specifically the ones backing type-based alias
/// analysis). Accessing the elements from Rust should be fine, however, as its
/// (proposed) aliasing model does not rely on type information.
inline const capi::oxidd_var_no_bool_pair_t *
to_c_var_bool_pair_ptr(const std::pair<var_no_t, bool> *args) {
  // From a C++ perspective, it is nicer to have elements of type
  // `std::pair<var_no_t, bool>` than `capi::oxidd_var_no_bool_pair_t`. In the
  // following we ensure that their layouts are compatible such that the pointer
  // cast below is safe.
  using c_pair = capi::oxidd_var_no_bool_pair_t;
  using cpp_pair = std::pair<var_no_t, bool>;
  static_assert(std::is_standard_layout_v<cpp_pair>);
  static_assert(sizeof(cpp_pair) == sizeof(c_pair));
  static_assert(offsetof(cpp_pair, first) == offsetof(c_pair, var));
  static_assert(offsetof(cpp_pair, second) == offsetof(c_pair, val));
  static_assert(alignof(cpp_pair) == alignof(c_pair));

  return reinterpret_cast<const c_pair *>(args); // NOLINT(*-cast)
}

inline const capi::oxidd_dddmp_export_settings_t *
to_c_export_settings(const std::optional<util::dddmp_export_settings> settings,
                     capi::oxidd_dddmp_export_settings_t &target) {
  if (!settings.has_value())
    return nullptr;

  target.version = static_cast<capi::oxidd_dddmp_version>(settings->version);
  target.ascii = settings->ascii;
  target.strict = settings->strict;
  target.diagram_name = {.ptr = settings->diagram_name.data(),
                         .len = settings->diagram_name.size()};

  return &target;
}

} // namespace detail

/// @endcond

/// Base class for managers
///
/// A manager is the data structure responsible for storing nodes and ensuring
/// their uniqueness. It also defines the variable order.
///
/// ### Reference Counting
///
/// This class is similar to a `std::shared_ptr`: copying instances just
/// increments an atomic reference count, moving out of an instance invalidates
/// it. Also, the default constructor yields an invalid manager instance.
/// Checking if an instance is invalid is possible via the `is_invalid()`
/// method.
///
/// ### Concurrency
///
/// Implementations supporting concurrency have an internal read/write lock.
/// Many operations acquire this lock for reading (shared) or writing
/// (exclusive).
template <class Derived, class DDFunc, typename CManager> class manager {
  friend Derived;

  /// Wrapped C API manager
  CManager _manager = {._p = nullptr};

public:
  /// Associated function type
  using function = DDFunc;

private:
  /// Default constructor, yields an invalid manager
  manager() noexcept = default;
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Time complexity: O(1)
  manager(const manager &other) noexcept : _manager(other._manager) {
    Derived::_c_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  ///
  /// Time complexity: O(1)
  manager(manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  /// Create a new DD manager from a manager instance of the C API
  explicit manager(CManager c_manager) noexcept : _manager(c_manager) {}

public:
  ~manager() noexcept { Derived::_c_unref(_manager); }

  /// Copy assignment operator
  ///
  /// Time complexity: O(1)
  manager &operator=(const manager &rhs) noexcept {
    if (this != &rhs) {
      Derived::_c_unref(_manager);
      _manager = rhs._manager;
      Derived::_c_ref(_manager);
    }
    return *this;
  }
  /// Move assignment operator: invalidates `rhs`
  ///
  /// Time complexity: O(1)
  manager &operator=(manager &&rhs) noexcept {
    assert(this != &rhs || rhs.is_invalid());
    Derived::_c_unref(_manager);
    _manager = rhs._manager;
    rhs._manager._p = nullptr;
    return *this;
  }

  /// Compare two managers for referential equality
  ///
  /// Time complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same manager
  friend bool operator==(const manager &lhs, const manager &rhs) noexcept {
    return lhs._manager._p == rhs._manager._p;
  }
  /// Same as `!(lhs == rhs)` (see `operator==()`)
  friend bool operator!=(const manager &lhs, const manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// @name C API Bridging
  /// @{

  /// Underlying C API type
  using c_api_t = CManager;

  /// Construct a manager from a C API struct
  ///
  /// This does not modify any reference counters.
  ///
  /// @param  c_manager  The C API struct to be wrapped
  ///
  /// @returns  The manager object wrapping `c_manager`
  [[nodiscard]] static Derived from_c_api(CManager c_manager) noexcept {
    return Derived(c_manager);
  }

  /// Get the wrapped C API struct
  ///
  /// This does not modify any reference counters.
  ///
  /// @returns  The wrapped manager struct
  [[nodiscard]] CManager to_c_api() const noexcept { return _manager; }

  /// @}

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor is invalid as well
  /// as a `manager` instance that has been moved (via the move constructor or
  /// assignment operator).
  ///
  /// @returns  `true` iff this manager reference is invalid
  [[nodiscard]] bool is_invalid() const noexcept {
    return _manager._p == nullptr;
  }

  /// Execute `f()` in the worker thread pool of `manager`
  ///
  /// Recursive calls in the multithreaded apply algorithms are always executed
  /// within the manager's thread pool, requiring a rather expensive context
  /// switch if the apply algorithm is not called from within the thread pool.
  /// If the algorithm takes long to execute anyway, this may not be important,
  /// but for many small operations, this may easily make a difference by
  /// factors.
  ///
  /// This method blocks until `f()` has finished.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// @returns  The result of calling `f()`
  template <std::invocable F>
  std::invoke_result_t<F> run_in_worker_pool(F &&f) const
    requires std::movable<std::invoke_result_t<F>> ||
             std::is_void_v<std::invoke_result_t<F>>
  {
    assert(!is_invalid());
    return detail::c_callback(std::forward<F>(f))
        .use_in_invoke(Derived::_c_run_in_worker_pool, _manager);
  }

  /// @name Variable management
  /// @{

  /// Get the number of variables in this manager
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The number of variables
  [[nodiscard]] var_no_t num_vars() const noexcept {
    return Derived::_c_num_vars(_manager);
  }
  /// Get the number of named variables in this manager
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The number of named variables
  [[nodiscard]] var_no_t num_named_vars() const noexcept {
    return Derived::_c_num_named_vars(_manager);
  }

  /// Add `additional` unnamed variables to this decision diagram
  ///
  /// The new variables are added at the bottom of the variable order. More
  /// precisely, the level number equals the variable number for each new
  /// variable.
  ///
  /// Note that some algorithms may assume that the domain of a function
  /// represented by a decision diagram is just the set of all variables, e.g.,
  /// `zbdd_function::operator~()`. In this regard, adding variables can change
  /// the semantics of decision diagram nodes.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @param  additional  Count of variables to add. Adding this to the current
  ///                     number of variables (`num_vars()`) must not overflow.
  ///
  /// @returns  The range of new variable numbers
  var_no_range_t add_vars(var_no_t additional) noexcept {
    assert(!is_invalid());
    capi::oxidd_var_no_range_t range =
        Derived::_c_add_vars(_manager, additional);
    // The standard specifies that the constructor is `explicit`, with an
    // initializer list it does not compile on Windows/MSVC.
    // NOLINTNEXTLINE(*-braced-init-list)
    return std::ranges::iota_view(range.start, range.end);
  }

  /// Add named variables to this decision diagram
  ///
  /// This is a shorthand for `add_vars()` and respective `set_var_name()`
  /// calls. More details can be found there.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @param  names  Span of variable names. Each name must be a null-terminated
  ///                UTF-8 string or `nullptr`. Both an empty string and
  ///                `nullptr` mean that the variable is unnamed.
  ///                Adding `names.size()` to the current number of variables
  ///                (`num_vars()`) must not overflow.
  ///
  /// @returns  Result indicating whether renaming was successful or which name
  ///           is already in use
  compat::expected<var_no_range_t, util::duplicate_var_name>
  add_named_vars(std::span<const char *const> names) noexcept {
    assert(!is_invalid());
    const capi::oxidd_duplicate_var_name_result_t res =
        Derived::_c_add_named_vars(_manager, names.data(), names.size());
    const var_no_range_t added_vars =
        std::ranges::iota_view(res.added_vars.start, res.added_vars.end);
    if (res.present_var == invalid_var_no)
      return added_vars;
    return compat::unexpected(util::duplicate_var_name{
        .added_vars = added_vars,
        .name = std::string(names[res.added_vars.end - res.added_vars.start]),
        .present_var = res.present_var,
    });
  }
  /// Add named variables to this decision diagram from an iterator
  ///
  /// See `add_named_vars(range)` for more details.
  template <std::input_iterator I, std::sentinel_for<I> E>
  compat::expected<var_no_range_t, util::duplicate_var_name>
  add_named_vars(I begin, E end)
    requires std::convertible_to<std::iter_value_t<I>, std::string_view>
  {
    assert(!is_invalid());
    detail::c_iter<capi::oxidd_str_t, I, E> iter(std::move(begin),
                                                 std::move(end));
    const capi::oxidd_duplicate_var_name_result_t res =
        iter.use_in_invoke(Derived::_c_add_named_vars_iter, _manager);
    const var_no_range_t added_vars =
        std::ranges::iota_view(res.added_vars.start, res.added_vars.end);
    if (res.present_var == invalid_var_no)
      return added_vars;
    return compat::unexpected(util::duplicate_var_name{
        .added_vars = added_vars,
        .name = std::string(iter.current()),
        .present_var = res.present_var,
    });
  }
  /// Add named variables to this decision diagram
  ///
  /// This is a shorthand for `add_vars()` and respective `set_var_name()`
  /// calls. More details can be found there.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @param  range  Range yielding UTF-8 encoded `string_view`s. An empty
  ///                string means that the variable is unnamed.
  ///                The range must not yield so many strings that the variable
  ///                count (`num_vars()`) overflows.
  ///
  /// @returns  Result indicating whether renaming was successful or which name
  ///           is already in use
  template <std::ranges::input_range R>
  compat::expected<var_no_range_t,
                   util::duplicate_var_name>
  add_named_vars(R &&range) // NOLINT(*-missing-std-forward)
    requires std::convertible_to<std::ranges::range_value_t<R>,
                                 std::string_view> &&
             // using the std::span version will be cheaper
             (!std::convertible_to<R &&, std::span<const char *const>>)
  {
    return add_named_vars(std::ranges::begin(range), std::ranges::end(range));
  }

  /// Get `var`'s name
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  var  The variable number. Must be less than the variable count
  ///              (`num_vars()`).
  ///
  /// @returns  The name, or an empty string for unnamed variables
  [[nodiscard]] std::string var_name(var_no_t var) const noexcept {
    assert(!is_invalid());
    std::string name;
    Derived::_c_var_name_cpp(_manager, var, &name);
    return name;
  }

  /// Label `var` as `name`
  ///
  /// An empty name means that the variable will become unnamed, and cannot be
  /// retrieved via `name_to_var()` anymore.
  ///
  /// Note that variable names are required to be unique. If labelling `var` as
  /// `name` would violate uniqueness, then `var`'s name is left unchanged.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @param  var   The variable number. Must be less than the variable count
  ///               (`num_vars()`).
  /// @param  name  A UTF-8 string to be used as the variable name.
  ///
  /// @returns  `std::nullopt` on success, otherwise the variable which already
  ///           uses `name`
  std::optional<var_no_t> set_var_name(var_no_t var,
                                       std::string_view name) noexcept {
    assert(!is_invalid());
    var_no_t present =
        Derived::_c_set_var_name(_manager, var, name.data(), name.size());
    return present == invalid_var_no ? std::nullopt : std::optional(present);
  }

  /// Get the variable number for the given variable name, if present
  ///
  /// Note that you cannot retrieve unnamed variables. Calling this function
  /// with an empty name will always result in `std::nullopt`.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  name  UTF-8 string to look up
  ///
  /// @returns  The variable number if found, otherwise `std::nullopt`
  [[nodiscard]] std::optional<var_no_t>
  name_to_var(std::string_view name) const noexcept {
    assert(!is_invalid());
    var_no_t var = Derived::_c_name_to_var(_manager, name.data(), name.size());
    return var == invalid_var_no ? std::nullopt : std::optional(var);
  }

  /// Get the level for the given variable
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  var  The variable number. Must be less than the variable count
  ///              (`num_vars()`).
  ///
  /// @returns  The corresponding level number
  [[nodiscard]] level_no_t var_to_level(var_no_t var) const noexcept {
    assert(!is_invalid());
    return Derived::_c_var_to_level(_manager, var);
  }

  /// Get the variable for the given level
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  level  The level number. Must be less than the level/variable
  ///                count (`num_vars()`).
  ///
  /// @returns  The corresponding variable number
  [[nodiscard]] var_no_t level_to_var(level_no_t level) const noexcept {
    assert(!is_invalid());
    return Derived::_c_level_to_var(_manager, level);
  }

  /// @}
  /// @name Statistics
  /// @{

  /// Get the count of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The number of inner nodes
  ///
  /// @see  `approx_num_inner_nodes()`
  [[nodiscard]] std::size_t num_inner_nodes() const noexcept {
    assert(!is_invalid());
    return Derived::_c_num_inner_nodes(_manager);
  }
  /// Get an approximate count of inner nodes
  ///
  /// For concurrent implementations, it may be much less costly to determine an
  /// approximation of the inner node count that an accurate count
  /// (`num_inner_nodes()`).
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  An approximate count of inner nodes
  [[nodiscard]] std::size_t approx_num_inner_nodes() const noexcept {
    assert(!is_invalid());
    return Derived::_c_approx_num_inner_nodes(_manager);
  }

  /// @}
  /// @name Garbage collection
  /// @{

  /// Perform garbage collection
  ///
  /// This method looks for nodes that are neither referenced by a `function`
  /// nor another node and removes them. The method works from top to bottom, so
  /// if a node is only referenced by nodes that can be removed, this node will
  /// be removed as well.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The count of nodes removed
  std::size_t gc() noexcept {
    assert(!is_invalid());
    return Derived::_c_gc(_manager);
  }

  /// Get the count of garbage collections
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The garbage collection count
  [[nodiscard]] std::uint64_t gc_count() const noexcept {
    assert(!is_invalid());
    return Derived::_c_gc_count(_manager);
  }

  /// @}
  /// @name Import and export
  /// @{

  /// Import the decision diagram from the DDDMP `file` into `manager`
  ///
  /// Note that the support variables must also be ordered by their current
  /// level (lower level numbers first). To this end, you can use
  /// `set_var_order()` with `support_vars` (or
  /// `oxidd_dddmp_support_var_order(file)`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  file          The DDDMP file handle
  /// @param  support_vars  Optional mapping from support variables of the DDDMP
  ///                       file to variable numbers in this manager. By
  ///                       default, `oxidd_dddmp_support_var_order(file)` will
  ///                       be used. If non-null, the pointer must reference an
  ///                       array with `oxidd_dddmp_num_support_vars(file)`
  ///                       elements.
  ///
  /// @returns  The imported DD functions or an error
  compat::expected<std::vector<DDFunc>, util::error>
  import_dddmp(util::dddmp_file &file,
               std::span<const var_no_t> support_vars = {}) noexcept {
    static_assert(sizeof(DDFunc) == sizeof(typename DDFunc::c_api_t) &&
                  alignof(DDFunc) == alignof(typename DDFunc::c_api_t));
    assert(!is_invalid());
    assert(support_vars.empty() ||
           support_vars.size() == file.num_support_vars());

    std::vector<DDFunc> result(file.num_roots());
    capi::oxidd_error_t err;
    const bool success = Derived::_c_import_dddmp(
        _manager, file.to_c_api(),
        support_vars.empty() ? nullptr : support_vars.data(),
        // `DDFunc` contains precisely one member of type `DDFunc::c_api_t`
        // NOLINTNEXTLINE(*-cast)
        reinterpret_cast<typename DDFunc::c_api_t *>(result.data()), &err);
    if (success)
      return result;
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Export the given decision diagram functions as DDDMP file at `path`
  ///
  /// If a file at `path` exists, it will be truncated, otherwise a new one will
  /// be created.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  path       Path at which the DOT file should be written
  /// @param  functions  Span of DD functions in this manager to be exported
  /// @param  settings   Optional export settings. If omitted, the default
  ///                    settings will be used.
  compat::expected<void, util::error> export_dddmp(
      std::string_view path, std::span<const DDFunc> functions,
      std::optional<util::dddmp_export_settings> settings = {}) noexcept {
    static_assert(sizeof(DDFunc) == sizeof(typename DDFunc::c_api_t) &&
                  alignof(DDFunc) == alignof(typename DDFunc::c_api_t));
    assert(!is_invalid());
    capi::oxidd_dddmp_export_settings_t c_settings;
    capi::oxidd_error_t err;
    const bool success = Derived::_c_export_dddmp(
        _manager, path.data(), path.size(),
        // NOLINTNEXTLINE(*-cast) // see reinterpret_cast above
        reinterpret_cast<const DDFunc::c_api_t *>(functions.data()),
        functions.size(), nullptr,
        detail::to_c_export_settings(settings, c_settings), &err);
    if (success)
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Export the given decision diagram functions as DDDMP file at `path`
  ///
  /// See `export_dddmp(path, functions, settings)` for more details. This
  /// overload allows specifying `functions` via iterators.
  template <std::input_iterator I, std::sentinel_for<I> E>
  compat::expected<void, util::error>
  export_dddmp(std::string_view path, I begin, E end,
               std::optional<util::dddmp_export_settings> settings = {})
    requires std::convertible_to<std::iter_value_t<I>, const DDFunc &>
  {
    assert(!is_invalid());
    capi::oxidd_dddmp_export_settings_t c_settings;
    const auto *settings_ptr =
        detail::to_c_export_settings(settings, c_settings);
    capi::oxidd_error_t err;
    detail::c_iter<typename DDFunc::c_api_t, I, E> iter(std::move(begin),
                                                        std::move(end));
    const bool success = iter.use_in_invoke( //
        [this, path, settings_ptr, &err](auto iter) noexcept -> bool {
          return Derived::_c_export_dddmp_iter(
              _manager, path.data(), path.size(), iter, settings_ptr, &err);
        });
    if (success)
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Export the given decision diagram functions as DDDMP file at `path`
  ///
  /// See `export_dddmp(path, functions, settings)` for more details. This
  /// overload allows specifying `functions` as a range.
  template <std::ranges::input_range R>
  compat::expected<void, util::error>
  export_dddmp(std::string_view path,
               R &&range, // NOLINT(*-missing-std-forward)
               std::optional<util::dddmp_export_settings> settings = {})
    requires std::convertible_to<std::ranges::range_value_t<R>,
                                 const DDFunc &> &&
             // using the std::span version will be cheaper
             (!std::convertible_to<R &&, std::span<const DDFunc>>)
  {
    return export_dddmp(path, std::ranges::begin(range),
                        std::ranges::end(range), settings);
  }

  /// Export the given decision diagram functions as DDDMP file at `path`
  ///
  /// See `export_dddmp_with_names(path, functions, settings)` for more details.
  /// This overload allows specifying `functions` as iterators.
  template <std::input_iterator I, std::sentinel_for<I> E>
  compat::expected<void, util::error> export_dddmp_with_names(
      std::string_view path, I begin, E end,
      std::optional<util::dddmp_export_settings> settings = {})
    requires util::pair_like<std::iter_value_t<I>, const DDFunc &,
                             std::string_view>
  {
    assert(!is_invalid());
    capi::oxidd_dddmp_export_settings_t c_settings;
    const auto *settings_ptr =
        detail::to_c_export_settings(settings, c_settings);
    capi::oxidd_error_t err;
    detail::c_iter<typename DDFunc::c_named_t, I, E> iter(std::move(begin),
                                                          std::move(end));
    const bool success = iter.use_in_invoke(
        [this, path, settings_ptr, &err](auto iter) noexcept -> bool {
          return Derived::_c_export_dddmp_with_names_iter(
              _manager, path.data(), path.size(), iter, settings_ptr, &err);
        });
    if (success)
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Export the given decision diagram functions as DDDMP file at `path`
  ///
  /// If a file at `path` exists, it will be truncated, otherwise a new one will
  /// be created.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  path       Path at which the DOT file should be written
  /// @param  functions  Range yielding DD functions of this manager to be
  ///                    exported, paired with a name each
  /// @param  settings   Optional export settings. If omitted, the default
  ///                    settings will be used.
  template <std::ranges::input_range R>
  compat::expected<void, util::error> export_dddmp_with_names(
      std::string_view path,
      R &&functions, // NOLINT(*-missing-std-forward)
      std::optional<util::dddmp_export_settings> settings = {})
    requires util::pair_like<std::ranges::range_value_t<R>, const DDFunc &,
                             std::string_view>
  {
    return export_dddmp_with_names(path, std::ranges::begin(functions),
                                   std::ranges::end(functions), settings);
  }

  /// Serve the given decision diagram functions for visualization
  ///
  /// Blocks until the visualization has been fetched by
  /// <a href="https://oxidd.net/vis">OxiDD-vis</a> (or another compatible
  /// tool).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  diagram_name  Name of the decision diagram
  /// @param  functions     Span of functions to visualize
  /// @param  port          The port to provide the data on. When passing `0`,
  ///                       the default port 4000 will be used.
  compat::expected<void, util::error>
  visualize(std::string_view diagram_name, std::span<const DDFunc> functions,
            uint16_t port = 0) noexcept {
    static_assert(sizeof(DDFunc) == sizeof(typename DDFunc::c_api_t) &&
                  alignof(DDFunc) == alignof(typename DDFunc::c_api_t));
    assert(!is_invalid());
    capi::oxidd_error_t err;
    const bool success = Derived::_c_visualize(
        _manager, diagram_name.data(), diagram_name.size(),
        // NOLINTNEXTLINE(*-cast) // see reinterpret_cast above
        reinterpret_cast<const DDFunc::c_api_t *>(functions.data()),
        functions.size(), nullptr, port, &err);
    if (success)
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Serve the given decision diagram functions for visualization
  ///
  /// See `visualize(diagram_name, functions, port)` for more details. This
  /// overload allows specifying `functions` via iterators.
  template <std::input_iterator I, std::sentinel_for<I> E>
  compat::expected<void, util::error>
  visualize(std::string_view diagram_name, I begin, E end, uint16_t port = 0)
    requires std::convertible_to<std::iter_value_t<I>, const DDFunc &>
  {
    assert(!is_invalid());
    capi::oxidd_error_t err;
    detail::c_iter<typename DDFunc::c_api_t, I, E> iter(std::move(begin),
                                                        std::move(end));
    const bool success = iter.use_in_invoke( //
        [this, diagram_name, port, &err](auto iter) noexcept -> bool {
          return Derived::_c_visualize_iter(_manager, diagram_name.data(),
                                            diagram_name.size(), iter, port,
                                            &err);
        });
    if (success)
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Serve the given decision diagram functions for visualization
  ///
  /// See `visualize(diagram_name, functions, port)` for more details. This
  /// overload allows specifying `functions` as a range.
  template <std::ranges::input_range R>
  compat::expected<void, util::error>
  visualize(std::string_view diagram_name,
            R &&range, // NOLINT(*-missing-std-forward)
            uint16_t port = 0)
    requires std::convertible_to<std::ranges::range_value_t<R>,
                                 const DDFunc &> &&
             // using the std::span version will be cheaper
             (!std::convertible_to<R &&, std::span<const DDFunc>>)
  {
    return visualize(diagram_name, std::ranges::begin(range),
                     std::ranges::end(range), port);
  }

  /// Serve the given decision diagram functions for visualization
  ///
  /// See `visualize_with_names(diagram_name, functions, port)` for more
  /// details. This overload allows specifying `functions` via iterators.
  template <std::input_iterator I, std::sentinel_for<I> E>
  compat::expected<void, util::error>
  visualize_with_names(std::string_view diagram_name, I begin, E end,
                       uint16_t port = 0)
    requires util::pair_like<std::iter_value_t<I>, const DDFunc &,
                             std::string_view>
  {
    assert(!is_invalid());
    capi::oxidd_error_t err;
    detail::c_iter<typename DDFunc::c_named_t, I, E> iter(std::move(begin),
                                                          std::move(end));
    const bool success = iter.use_in_invoke( //
        [this, diagram_name, port, &err](auto iter) noexcept -> bool {
          return Derived::_c_visualize_with_names_iter(
              _manager, diagram_name.data(), diagram_name.size(), iter, port,
              &err);
        });
    if (success)
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Serve the given decision diagram functions for visualization
  ///
  /// Blocks until the visualization has been fetched by
  /// <a href="https://oxidd.net/vis">OxiDD-vis</a> (or another compatible
  /// tool).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  diagram_name  Name of the decision diagram
  /// @param  functions     Range yielding DD functions of this manager to be
  ///                       visualized, paired with a name each
  /// @param  port          The port to provide the data on. When passing `0`,
  ///                       the default port 4000 will be used.
  template <std::ranges::input_range R>
  compat::expected<void, util::error>
  visualize_with_names(std::string_view diagram_name,
                       R &&functions, // NOLINT(*-missing-std-forward)
                       uint16_t port = 0)
    requires util::pair_like<std::ranges::range_value_t<R>, const DDFunc &,
                             std::string_view>
  {
    return visualize_with_names(diagram_name, std::ranges::begin(functions),
                                std::ranges::end(functions), port);
  }

  /// Dump the entire decision diagram represented by `manager` as Graphviz DOT
  /// code to a file at `path`
  ///
  /// If a file at `path` exists, it will be truncated, otherwise a new one will
  /// be created.
  ///
  /// The other overloads of this method allow naming selected DD functions.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  compat::expected<void, util::error> dump_all_dot(std::string_view path) {
    assert(!is_invalid());
    capi::oxidd_error_t err;
    if (Derived::_c_dump_all_dot_path(_manager, path.data(), path.size(),
                                      nullptr, nullptr, 0, &err))
      return {};
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Dump the entire decision diagram represented by `manager` as Graphviz DOT
  /// code to a file at `path`
  ///
  /// For more details, see `dump_all_dot(path, functions)`
  template <std::input_iterator I, std::sentinel_for<I> E>
  compat::expected<void, util::error> dump_all_dot(std::string_view path,
                                                   I begin, E end)
    requires util::pair_like<std::iter_value_t<I>, const DDFunc &,
                             std::string_view>
  {
    assert(!is_invalid());
    capi::oxidd_error_t err;
    detail::c_iter<typename DDFunc::c_named_t, I, E> iter(std::move(begin),
                                                          std::move(end));
    const bool success =
        iter.use_in_invoke([this, path, &err](auto iter) noexcept -> bool {
          return Derived::_c_dump_all_dot_path_iter(_manager, path.data(),
                                                    path.size(), iter, &err);
        });
    if (success) {
      return {};
    }
    return compat::unexpected(util::error::from_c_api(err));
  }

  /// Dump the entire decision diagram represented by `manager` as Graphviz DOT
  /// code to a file at `path`
  ///
  /// If a file at `path` exists, it will be truncated, otherwise a new one will
  /// be created.
  ///
  /// `functions` is a range yielding pairs of a DD function and a name. The DD
  /// functions will be marked with their names in the output. Nonetheless, the
  /// output will include all nodes stored in this manager, even if they are not
  /// reachable from the DD functions of the range.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  template <std::ranges::input_range R>
  compat::expected<void, util::error>
  dump_all_dot(std::string_view path,
               R &&functions) // NOLINT(*-missing-std-forward)
    requires util::pair_like<std::ranges::range_value_t<R>, const DDFunc &,
                             std::string_view>
  {
    return dump_all_dot(path, std::ranges::begin(functions),
                        std::ranges::end(functions));
  }

  /// @}
};

/// Manager that supports variable reordering
template <class Derived> class reordering_manager {
  friend Derived;
  reordering_manager() = default;

public:
  /// Reorder the variables in this manager according to `order`
  ///
  /// If a variable `x` occurs before variable `y` in `order`, then `x` will be
  /// above `y` in the decision diagram when this function returns. Variables
  /// not mentioned in `order` will be placed in a position such that the least
  /// number of level swaps need to be performed.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  void set_var_order(std::span<const var_no_t> order) noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    Derived::_c_set_var_order(self.to_c_api(), order.data(), order.size());
  }
};

/// Manager for Boolean functions
template <class Derived, class DDFunc> class boolean_function_manager {
  friend Derived;
  boolean_function_manager() = default;

public:
  /// Get the Boolean function that is true if and only if `var` is true
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  var  The variable number. Must be less than the variable count
  ///              (`num_vars()`).
  ///
  /// @returns  The DD function representing the variable
  [[nodiscard]] DDFunc var(var_no_t var) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_var(self.to_c_api(), var);
  }
  /// Get the Boolean function that is true if and only if `var` is false
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  var  The variable number. Must be less than the variable count
  ///              (`num_vars()`).
  ///
  /// @returns  The DD function representing the negated variable
  [[nodiscard]] DDFunc not_var(var_no_t var) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_not_var(self.to_c_api(), var);
  }

  /// Get the constant true DD function ⊤
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  ⊤ as DD function
  [[nodiscard]] DDFunc t() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_true(self.to_c_api());
  }
  /// Get the constant false DD function ⊥
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  ⊥ as DD function
  [[nodiscard]] DDFunc f() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_false(self.to_c_api());
  }
};

/// Base class for functions represented as decision diagrams
///
/// A function is the combination of a reference to a `manager` and a (possibly
/// tagged) edge pointing to a node.
///
/// ### Reference Counting
///
/// This class is similar to a `std::shared_ptr`: copying instances just
/// increments an atomic reference count, moving out of an instance invalidates
/// it. Invalid function instances are also produced by the default constructor
/// and decision diagram operations that run out of memory. Checking if an
/// instance is invalid is possible via the `is_invalid()` method. Unless
/// explicitly stated otherwise, invalid functions may be passed to a method. If
/// one operand of a decision diagram operation is invalid, then the operation
/// produces an invalid function. This permits chaining of these operations
/// without checking for out of memory issues in between all the time.
///
/// ### Ordering, Hashing
///
/// Functions are hashable and totally ordered according to an arbitrary order.
/// Two functions `f`, `g` are equal `f == g` iff they are stored in the same
/// manager and are structurally equal, i.e., they have the same underlying
/// edge. In the following, we denote structural equality as `f ≅ g`.
///
/// By “ordered according to an arbitrary order” we mean: Assuming two functions
/// `f`, `g` with `f < g`, then if either `f` or `g` is deleted and recreated
/// later on, `f < g` does not necessarily hold anymore. Similarly, the hash
/// value of `f` may change if deleting `f` and recreating it later on.
/// Moreover, assume `f < g` and two structurally equivalent functions `f2`,
/// `g2` in different managers (i.e., `f ≅ f2` but `f != f2`, and `g ≅ g2` but
/// `g != g2`), then `f2 < g2` does not necessarily hold.
///
/// In general, structural equality on decision diagrams implies semantic
/// equality. The comparisons and the hash implementation typically operate on
/// underlying pointer or index values, making the implementations very
/// efficient. However, such implementations also lead to the aforementioned
/// restrictions.
template <class Derived, class Manager, typename CFunction> class function {
  friend Derived;

  friend Manager;
  friend struct std::hash<function>;

  /// Wrapped C API function
  CFunction _func = {._p = nullptr};

public:
  /// Associated manager type
  using manager = Manager;

private:
  /// Default constructor, yields an invalid function
  function() noexcept = default;
  /// Copy constructor: increments the internal reference counters
  ///
  /// Time complexity: O(1)
  function(const function &other) noexcept : _func(other._func) {
    Derived::_c_ref(_func);
  }
  /// Move constructor: invalidates `other`
  ///
  /// Time complexity: O(1)
  function(function &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  /// Create a new `function` from a C API function instance
  explicit function(CFunction func) noexcept : _func(func) {}

public:
  ~function() noexcept { Derived::_c_unref(_func); }

  /// Copy assignment operator
  function &operator=(const function &rhs) noexcept {
    if (this != &rhs) {
      Derived::_c_unref(_func);
      _func = rhs._func;
      Derived::_c_ref(_func);
    }
    return *this;
  }
  /// Move assignment operator: invalidates `rhs`
  function &operator=(function &&rhs) noexcept {
    assert(this != &rhs || !rhs._func._p);
    Derived::_c_unref(_func);
    _func = rhs._func;
    rhs._func._p = nullptr;
    return *this;
  }

  /// Compare two functions for referential equality
  ///
  /// Time complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same node and have the
  ///           same edge tag
  friend bool operator==(const function &lhs, const function &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  /// Same as `!(lhs == rhs)` (see `operator==()`)
  friend bool operator!=(const function &lhs, const function &rhs) noexcept {
    return !(lhs == rhs);
  }
  /// Check if `lhs` is less than `rhs` according to an arbitrary total order
  ///
  /// Time complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` is less than `rhs` according to the total order
  friend bool operator<(const function &lhs, const function &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  /// `operator<()` with arguments swapped
  friend bool operator>(const function &lhs, const function &rhs) noexcept {
    return rhs < lhs;
  }
  /// Same as `!(rhs < lhs)` (see `operator<()`)
  friend bool operator<=(const function &lhs, const function &rhs) noexcept {
    return !(rhs < lhs);
  }
  /// Same as `!(lhs < rhs)` (see `operator<()`)
  friend bool operator>=(const function &lhs, const function &rhs) noexcept {
    return !(lhs < rhs);
  }

  /// @name C API Bridging
  /// @{

  /// Underlying C API type
  using c_api_t = CFunction;

  /// Construct a manager from a C API struct
  ///
  /// This does not modify any reference counters.
  ///
  /// @param  c_func  The C API struct to be wrapped
  ///
  /// @returns  The manager object wrapping `c_func`
  [[nodiscard]] static Derived from_c_api(CFunction c_func) noexcept {
    return Derived(c_func);
  }

  /// Get the wrapped C API struct
  ///
  /// This does not modify any reference counters.
  ///
  /// @returns  The wrapped manager struct
  [[nodiscard]] CFunction to_c_api() const noexcept { return _func; }

  /// @}

  /// Check if this DD function is invalid
  ///
  /// A DD function created by the default constructor is invalid as well as a
  /// `function` instance that has been moved (via
  /// `function(function &&other)`). Moreover, if an operation tries to allocate
  /// new nodes but runs out of memory, then it returns an invalid function.
  ///
  /// @returns  `true` iff this DD function is invalid
  [[nodiscard]] bool is_invalid() const noexcept { return _func._p == nullptr; }

  /// Get the containing manager
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// @returns  The `manager`
  [[nodiscard]] manager containing_manager() const noexcept {
    assert(!is_invalid());
    return Derived::_c_containing_manager(_func);
  }

  /// Count descendant nodes
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  Node count including the terminal node
  [[nodiscard]] std::size_t node_count() const noexcept {
    assert(!is_invalid());
    return Derived::_c_node_count(_func);
  }
};

/// Extension for `function` where the underlying nodes carry their level number
template <class Derived> class has_level {
  friend Derived;
  has_level() = default;

public:
  /// Get the level of the underlying node
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The level of the underlying inner node or `std::nullopt` for
  ///           terminals and invalid functions
  [[nodiscard]] std::optional<level_no_t> node_level() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    const level_no_t level = Derived::_c_node_level(self.to_c_api());
    return level == invalid_level_no ? std::nullopt : std::optional(level);
  }
  /// Get the level of the underlying node
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The level of the underlying inner node or `std::nullopt` for
  ///           terminals and invalid functions
  ///
  /// @deprecated  use `node_level()` instead
  [[deprecated("use node_level instead"), nodiscard]] level_no_t
  level() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_node_level(self.to_c_api());
  }

  /// Get the variable number for the underlying node
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The variable number of the underlying inner node or
  ///           `std::nullopt` for terminals and invalid functions
  [[nodiscard]] std::optional<var_no_t> node_var() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    const var_no_t var = Derived::_c_node_var(self.to_c_api());
    return var == invalid_var_no ? std::nullopt : std::optional(var);
  }
};

template <class Derived> class function_subst;

/// Substitution mapping variables to replacement functions
///
/// The intent behind this class is to optimize the case where the same
/// substitution is applied multiple times. We would like to re-use apply cache
/// entries across these operations, and therefore, we need a compact identifier
/// for the substitution.
///
/// This behaves like a `std::unique_ptr` with respect to default/move
/// construction and assignment.
template <class DDFunc> class substitution {
  static_assert(std::is_base_of_v<function_subst<DDFunc>, DDFunc>);

  /// Wrapped C API substitution pointer
  DDFunc::c_substitution_t *_subst = nullptr;

  explicit substitution(DDFunc::c_substitution_t *c_substitution)
      : _subst(c_substitution) {}

public:
  /// Associated function type
  using function = DDFunc;

  /// Create an invalid substitution
  substitution() = default;
  substitution(const substitution &other) = delete;
  /// Move constructor: invalidates `other`
  substitution(substitution &&other) noexcept : _subst(other._subst) {
    other._subst = nullptr;
  }

  substitution &operator=(const substitution &) = delete;
  /// Move assignment operator: invalidates `rhs`
  substitution &operator=(substitution &&rhs) noexcept {
    assert(this != &rhs || !rhs._subst);
    DDFunc::_c_substitution_free(_subst);
    _subst = rhs._subst;
    rhs._subst = nullptr;
    return *this;
  }

  ~substitution() noexcept { DDFunc::_c_substitution_free(_subst); }

  /// Create a substitution from an iterator of pairs
  /// `(var_no_t variable, const function &replacement)`
  template <std::input_iterator I, std::sentinel_for<I> E>
    requires(util::pair_like<std::iter_value_t<I>, var_no_t, const function &>)
  substitution(I begin, E end)
      : _subst(DDFunc::_c_substitution_new(util::size_hint(begin, end))) {
    for (; begin != end; ++begin) {
      const auto &pair = *begin;
      const var_no_t var = std::get<0>(pair);
      const function &replacement = std::get<1>(pair);
      assert(!replacement.is_invalid());
      DDFunc::_c_substitution_add_pair(_subst, var, replacement.to_c_api());
    }
  }

  /// Create a substitution from a range of pairs
  /// `(const function &var, const function &replacement)`
  template <std::ranges::input_range R>
    requires(util::pair_like<std::ranges::range_value_t<R>, var_no_t,
                             const function &>)
  substitution(R &&range)
      : _subst(DDFunc::_c_substitution_new(util::size_hint(range))) {
    for (const auto &[var, replacement] : std::forward<R>(range)) {
      assert(!replacement.is_invalid());
      DDFunc::_c_substitution_add_pair(_subst, var, replacement.to_c_api());
    }
  }

  /// Check if this substitution is invalid
  [[nodiscard]] bool is_invalid() const { return _subst == nullptr; }

  /// @name C API Bridging
  /// @{

  /// Underlying C API type
  using c_api_t = DDFunc::c_substitution_t;

  /// Construct a manager from a C API struct
  ///
  /// This does not modify any reference counters.
  ///
  /// @param  c_subst  The C API struct to be wrapped
  ///
  /// @returns  The manager object wrapping `c_func`
  [[nodiscard]] static substitution from_c_api(c_api_t *c_subst) noexcept {
    return substitution(c_subst);
  }

  /// Get the wrapped C API struct
  ///
  /// This does not modify any reference counters.
  ///
  /// @returns  The wrapped substitution struct
  [[nodiscard]] c_api_t *to_c_api() const noexcept { return _subst; }

  /// @}
};

/// Substitution extension for `function`
///
/// @see  `substitution`
template <class Derived> class function_subst {
  friend Derived;
  function_subst() = default;

public:
  /// Associated substitution type
  using substitution = bridge::substitution<Derived>;

  /// Substitute `vars` in the DD function `f` by `replacement`
  ///
  /// The substitution is performed in a parallel fashion, e.g.:
  /// `(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived substitute(const substitution &subst) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!subst.is_invalid());
    return Derived::_c_substitute(self.to_c_api(), subst.to_c_api());
  }
};

/// Boolean `function` 𝔹ⁿ → 𝔹 represented as decision diagram
///
/// Instances may equivalently be viewed as a set of Boolean vectors 𝔹ⁿ.
///
/// @see  `function`, `boolean_function_manager`
template <class Derived> class boolean_function {
  friend Derived;
  boolean_function() = default;

  // Helper functions that are allowed to access private members of `Derived`.
  // MSVC does not permit accessing these private members in the friend
  // functions below (`operator&`, etc.).
  [[nodiscard]] Derived _and(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_and(self.to_c_api(), rhs.to_c_api());
  }
  [[nodiscard]] Derived _or(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_or(self.to_c_api(), rhs.to_c_api());
  }
  [[nodiscard]] Derived _xor(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_xor(self.to_c_api(), rhs.to_c_api());
  }

public:
  /// @name Construction
  /// @{

  /// Compute the negation `¬this`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function
  [[nodiscard]] Derived operator~() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_not(self.to_c_api());
  }

  /// Compute the conjunction `lhs ∧ rhs`
  ///
  /// The conjunction on Boolean functions may equivalently be viewed as an
  /// intersection of sets `lhs ∩ rhs`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend Derived operator&(const Derived &lhs,
                                         const Derived &rhs) noexcept {
    return lhs._and(rhs);
  }
  /// Assignment version of `operator&()`
  Derived &operator&=(const Derived &rhs) noexcept {
    Derived &self = *static_cast<Derived *>(this);
    return (self = self & rhs);
  }

  /// Compute the disjunction `lhs ∨ rhs`
  ///
  /// The disjunction on Boolean functions may equivalently be viewed as a
  /// union of sets `lhs ∪ rhs`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend Derived operator|(const Derived &lhs,
                                         const Derived &rhs) noexcept {
    return lhs._or(rhs);
  }
  /// Assignment version of `operator|()`
  Derived &operator|=(const Derived &rhs) noexcept {
    Derived &self = *static_cast<Derived *>(this);
    return (self = self | rhs);
  }

  /// Compute the exclusive disjunction `lhs ⊕ rhs`
  ///
  /// The exclusive disjunction on Boolean functions may equivalently be viewed
  /// as a symmetric difference on sets.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend Derived operator^(const Derived &lhs,
                                         const Derived &rhs) noexcept {
    return lhs._xor(rhs);
  }
  /// Assignment version of `operator^()`
  Derived &operator^=(const Derived &rhs) noexcept {
    Derived &self = *static_cast<Derived *>(this);
    return (self = self ^ rhs);
  }

  /// Compute the set difference `lhs ∖ rhs`
  ///
  /// This is equivalent to the strict implication `rhs < lhs` on Boolean
  /// functions.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The DD set/function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend Derived operator-(const Derived &lhs,
                                         const Derived &rhs) noexcept {
    return rhs.imp_strict(lhs);
  }
  /// Assignment version of @ref operator-
  Derived &operator-=(const Derived &rhs) noexcept {
    Derived &self = *static_cast<Derived *>(this);
    return (self = self - rhs);
  }

  /// Compute the negated conjunction `this ⊼ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|this| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived nand(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_nand(self.to_c_api(), rhs.to_c_api());
  }

  /// Compute the negated disjunction `this ⊽ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|this| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived nor(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_nor(self.to_c_api(), rhs.to_c_api());
  }

  /// Compute the equivalence `this ↔ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|this| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived equiv(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_equiv(self.to_c_api(), rhs.to_c_api());
  }

  /// Compute the implication `this → rhs` (or `this ≤ rhs`)
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|this| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived imp(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_imp(self.to_c_api(), rhs.to_c_api());
  }

  /// Compute the strict implication `this < rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|this| · |rhs|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived imp_strict(const Derived &rhs) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_imp_strict(self.to_c_api(), rhs.to_c_api());
  }

  /// Compute the conditional “if `this` then `t` else `e`”
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(|this| · |t| · |e|)
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived ite(const Derived &t, const Derived &e) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_ite(self.to_c_api(), t.to_c_api(), e.to_c_api());
  }

  /// @}
  /// @name Traversal
  /// @{

  /// Get the cofactors `(f_true, f_false)` of `f`
  ///
  /// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the
  /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
  /// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
  ///
  /// Structurally, the cofactors are children with edge tags are adjusted
  /// accordingly. If you only need one of the cofactors, then use
  /// `cofactor_true()` or `cofactor_false()`. These functions are slightly more
  /// efficient then.
  ///
  /// Note that the domain of f is 𝔹<sup>n+1</sup> while the domain of
  /// f<sub>true</sub> and f<sub>false</sub> is 𝔹<sup>n</sup>. (Remember that,
  /// e.g., g(x₀) = x₀ and g'(x₀, x₁) = x₀ have different representations as
  /// ZBDDs.)
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  The pair `{f_true, f_false}` if `f` is valid and references an
  ///           inner node, otherwise a pair of invalid functions.
  [[nodiscard]] std::pair<Derived, Derived> cofactors() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    const auto p = Derived::_c_cofactors(self.to_c_api());
    return {p.first, p.second};
  }
  /// Get the cofactor `f_true` of `f`
  ///
  /// This function is slightly more efficient than `cofactors()` in case
  /// `f_false` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  `f_true` if `f` is valid and references an inner node, otherwise
  ///           an invalid function.
  [[nodiscard]] Derived cofactor_true() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_cofactor_true(self.to_c_api());
  }
  /// Get the cofactor `f_false` of `f`
  ///
  /// This function is slightly more efficient than `cofactors()` in case
  /// `f_true` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @returns  `f_false` if `f` is valid and references an inner node,
  ///           otherwise an invalid function.
  [[nodiscard]] Derived cofactor_false() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_cofactor_false(self.to_c_api());
  }

  /// @}
  /// @name Query Operations
  /// @{

  /// Check for satisfiability
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is a satisfying assignment
  [[nodiscard]] bool satisfiable() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_satisfiable(self.to_c_api());
  }

  /// Check for validity
  ///
  /// `this` must not be invalid (in the technical, not the mathematical sense).
  /// Check via `is_invalid()`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is are only satisfying assignment
  [[nodiscard]] bool valid() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_valid(self.to_c_api());
  }

  /// Count the number of satisfying assignments
  ///
  /// This method assumes that the function's domain of has `vars` many
  /// variables.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  vars  Number of variables in the function's domain
  ///
  /// @returns  Count of satisfying assignments
  [[nodiscard]] double sat_count_double(var_no_t vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_sat_count_double(self.to_c_api(), vars);
  }

  /// Pick a satisfying assignment
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. If this function is
  ///           unsatisfiable, the assignment is empty.
  [[nodiscard]] util::assignment pick_cube() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return util::assignment(Derived::_c_pick_cube(self.to_c_api()));
  }

  /// Pick a satisfying assignment, represented as DD
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
  ///           `f` is ⊥), ⊥ is returned.
  [[nodiscard]] Derived pick_cube_dd() const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_pick_cube_dd(self.to_c_api());
  }

  /// Pick a satisfying assignment, represented as DD, using the literals in
  /// `literal_set` if there is a choice
  ///
  /// `literal_set` is represented as a conjunction of literals. Whenever there
  /// is a choice for a variable, it will be set to true if the variable has a
  /// positive occurrence in `literal_set`, and set to false if it occurs
  /// negated in `literal_set`. If the variable does not occur in `literal_set`,
  /// then it will be left as don't care if possible, otherwise an arbitrary
  /// (not necessarily random) choice will be performed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
  ///           `f` is ⊥), ⊥ is returned.
  [[nodiscard]] Derived
  pick_cube_dd_set(const Derived &literal_set) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_pick_cube_dd_set(self.to_c_api(),
                                        literal_set.to_c_api());
  }

  /// Evaluate this Boolean function with arguments `args`
  ///
  /// `args` determines the valuation for all variables in the function's
  /// domain. The order is irrelevant (except that if the valuation for a
  /// variable is given multiple times, the last value counts).
  ///
  /// Note that the domain of the Boolean function represented by `this` is
  /// implicit and may comprise a strict subset of the variables in the manager
  /// only. This method assumes that the function's domain corresponds the set
  /// of variables in `args`. Remember that there are kinds of decision diagrams
  /// (e.g., ZBDDs) where the domain plays a crucial role for the interpretation
  /// of decision diagram nodes as a Boolean function. On the other hand,
  /// extending the domain of, e.g., ordinary BDDs does not affect the
  /// evaluation result.
  ///
  /// Should there be a decision node for a variable not part of the domain,
  /// then `false` is used as the decision value.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  args  Span of pairs `(variable, value)`, where each variable
  ///               number must be less than the number of variables in the
  ///               manager
  ///
  /// @returns  Result of the evaluation with `args`
  [[nodiscard]] bool
  eval(std::span<const std::pair<var_no_t, bool>> args) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    assert(!self.is_invalid());
    return Derived::_c_eval(self.to_c_api(),
                            detail::to_c_var_bool_pair_ptr(args.data()),
                            args.size());
  }

  /// @}
};

/// Quantification extension for `boolean_function`
template <class Derived> class boolean_function_quant {
  friend Derived;
  boolean_function_quant() = default;

public:
  /// Compute the universal quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// universal quantification. Universal quantification `∀x. f(…, x, …)` of a
  /// Boolean function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ∧ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived forall(const Derived &vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_forall(self.to_c_api(), vars.to_c_api());
  }
  /// Compute the existential quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// existential quantification. Existential quantification `∃x. f(…, x, …)` of
  /// a Boolean function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ∨ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived exists(const Derived &vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_exists(self.to_c_api(), vars.to_c_api());
  }
  /// Deprecated alias for `exists()`
  ///
  /// @deprecated  use `exists()` instead
  [[nodiscard, deprecated]]
  Derived exist(const Derived &vars) const noexcept {
    return exists(vars);
  }
  /// Compute the unique quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// unique quantification. Unique quantification `∃!x. f(…, x, …)` of a
  /// Boolean function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ⊕ f(…, 1, …)`.
  ///
  /// Unique quantification is also known as the
  /// [Boolean
  /// difference](https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors)
  /// or
  /// [Boolean
  /// derivative](https://en.wikipedia.org/wiki/Boolean_differential_calculus).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] Derived unique(const Derived &vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_unique(self.to_c_api(), vars.to_c_api());
  }

  /// Combined application of `op` and `forall()`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function `∀ vars. this <op> rhs` (may be invalid if the
  ///           operation runs out of memory)
  [[nodiscard]] Derived apply_forall(const util::boolean_operator op,
                                     const Derived &rhs,
                                     const Derived &vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_apply_forall(
        static_cast<capi::oxidd_boolean_operator>(op), self.to_c_api(),
        rhs.to_c_api(), vars.to_c_api());
  }

  /// Combined application of `op` and `exists()`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function `∃ vars. this <op> rhs` (may be invalid if the
  ///           operation runs out of memory)
  [[nodiscard]] Derived apply_exists(const util::boolean_operator op,
                                     const Derived &rhs,
                                     const Derived &vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_apply_exists(
        static_cast<capi::oxidd_boolean_operator>(op), self.to_c_api(),
        rhs.to_c_api(), vars.to_c_api());
  }
  /// Deprecated alias for `apply_exists()`
  ///
  /// @deprecated  use `apply_exists()` instead
  [[nodiscard, deprecated]]
  Derived apply_exist(const util::boolean_operator op, const Derived &rhs,
                      const Derived &vars) const noexcept {
    return apply_exists(op, rhs, vars);
  }

  /// Combined application of `op` and `unique()`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD function `∃! vars. this <op> rhs` (may be invalid if the
  ///           operation runs out of memory)
  [[nodiscard]] Derived apply_unique(const util::boolean_operator op,
                                     const Derived &rhs,
                                     const Derived &vars) const noexcept {
    const Derived &self = *static_cast<const Derived *>(this);
    return Derived::_c_apply_unique(
        static_cast<capi::oxidd_boolean_operator>(op), self.to_c_api(),
        rhs.to_c_api(), vars.to_c_api());
  }
};

} // namespace oxidd::bridge

/// @cond

/// Partial specialization for `oxidd::bridge::function` derivatives
template <class Derived>
  requires std::is_base_of_v<
      oxidd::bridge::function<Derived, typename Derived::manager,
                              typename Derived::c_api_t>,
      Derived>
struct std::hash<Derived> {
  [[nodiscard]] std::size_t operator()(const Derived &f) const noexcept {
    const typename Derived::c_api_t c = f.to_c_api();
    return std::hash<const void *>{}(c._p) ^ c._i;
  }
};

/// @endcond
