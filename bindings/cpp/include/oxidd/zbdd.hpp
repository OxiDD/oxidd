/// @file   zbdd.hpp
/// @brief  Zero-suppressed binary decision diagrams

#pragma once

#include <oxidd/bridge.hpp>

namespace oxidd {

class zbdd_function;

/// Manager for zero-suppressed binary decision diagrams
///
/// Instances can safely be sent to other threads.
///
/// Models `oxidd::concepts::boolean_function_manager`
class zbdd_manager
    : public bridge::manager<zbdd_manager, zbdd_function,
                             capi::oxidd_zbdd_manager_t>,
      public bridge::reordering_manager<zbdd_manager>,
      public bridge::boolean_function_manager<zbdd_manager, zbdd_function> {
  // friends
  friend class bridge::manager<zbdd_manager, zbdd_function,
                               capi::oxidd_zbdd_manager_t>;
  friend class bridge::reordering_manager<zbdd_manager>;
  friend class bridge::boolean_function_manager<zbdd_manager, zbdd_function>;

  friend class bridge::function<zbdd_function, zbdd_manager,
                                capi::oxidd_zbdd_t>;

  // C API functions
#define OXIDD_LINK_C(x)                                                        \
  static constexpr auto _c_##x = capi::oxidd_zbdd_manager_##x;

  // manager
  OXIDD_LINK_C(ref)
  OXIDD_LINK_C(unref)
  OXIDD_LINK_C(run_in_worker_pool)

  OXIDD_LINK_C(num_vars)
  OXIDD_LINK_C(num_named_vars)
  OXIDD_LINK_C(add_vars)
  OXIDD_LINK_C(add_named_vars)
  OXIDD_LINK_C(add_named_vars_iter)
  OXIDD_LINK_C(var_name_cpp)
  OXIDD_LINK_C(set_var_name)
  OXIDD_LINK_C(name_to_var)
  OXIDD_LINK_C(var_to_level)
  OXIDD_LINK_C(level_to_var)

  OXIDD_LINK_C(set_var_order)

  OXIDD_LINK_C(num_inner_nodes)
  OXIDD_LINK_C(approx_num_inner_nodes)

  OXIDD_LINK_C(gc)
  OXIDD_LINK_C(gc_count)

  OXIDD_LINK_C(import_dddmp)
  OXIDD_LINK_C(export_dddmp)
  OXIDD_LINK_C(export_dddmp_iter)
  OXIDD_LINK_C(export_dddmp_with_names_iter)
  OXIDD_LINK_C(visualize)
  OXIDD_LINK_C(visualize_iter)
  OXIDD_LINK_C(visualize_with_names_iter)
  OXIDD_LINK_C(dump_all_dot_path)
  OXIDD_LINK_C(dump_all_dot_path_iter)

#undef OXIDD_LINK_C
#define OXIDD_LINK_C(x) static constexpr auto _c_##x = capi::oxidd_zbdd_##x;
  // boolean_function_manager
  OXIDD_LINK_C(var)
  OXIDD_LINK_C(not_var)
  OXIDD_LINK_C(true)
  OXIDD_LINK_C(false)
#undef OXIDD_LINK_C

  /// Create a new ZBDD manager from a manager instance of the C API
  zbdd_manager(capi::oxidd_zbdd_manager_t c_manager) noexcept
      : manager(c_manager) {}

public:
  /// Default constructor, yields an invalid ZBDD manager
  zbdd_manager() noexcept = default;

  /// Create a new ZBDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  zbdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
               uint32_t threads) noexcept
      : manager(capi::oxidd_zbdd_manager_new(inner_node_capacity,
                                             apply_cache_capacity, threads)) {}

  /// @name Function creation
  /// @{

  /// Get the ZBDD set ∅
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  ∅ as ZBDD set
  [[nodiscard]] zbdd_function empty() const noexcept;
  /// Get the ZBDD set {∅}
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  {∅} as ZBDD set
  [[nodiscard]] zbdd_function base() const noexcept;

  /// Get the singleton set {var}
  ///
  /// `this` must not be invalid (check via is_invalid()).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Time complexity: O(1)
  ///
  /// @param  var  The variable number. Must be less than the variable count
  ///              (num_vars()).
  ///
  /// @returns  The ZBDD function representing the set
  [[nodiscard]] zbdd_function singleton(var_no_t var) const noexcept;

  /// @}
};

/// Boolean function 𝔹ⁿ → 𝔹 (or set of Boolean vectors 𝔹ⁿ) represented as
/// zero-suppressed binary decision diagram
///
/// Models `oxidd::concepts::boolean_function`
class zbdd_function
    : public bridge::function<zbdd_function, zbdd_manager, capi::oxidd_zbdd_t>,
      public bridge::has_level<zbdd_function>,
      public bridge::boolean_function<zbdd_function> {
  // friends
  friend class bridge::function<zbdd_function, zbdd_manager,
                                capi::oxidd_zbdd_t>;
  friend class bridge::boolean_function_manager<zbdd_manager, zbdd_function>;
  friend class bridge::boolean_function<zbdd_function>;
  friend class bridge::boolean_function_quant<zbdd_function>;
  friend class bridge::has_level<zbdd_function>;

  friend class zbdd_manager;
  friend class bridge::manager<zbdd_manager, zbdd_function,
                               capi::oxidd_zbdd_manager_t>;

  // C API functions
  using c_named_t = capi::oxidd_named_zbdd_t;
#define OXIDD_LINK_C(x) static constexpr auto _c_##x = capi::oxidd_zbdd_##x;
  // function
  OXIDD_LINK_C(ref)
  OXIDD_LINK_C(unref)
  OXIDD_LINK_C(containing_manager)
  OXIDD_LINK_C(node_count)

  // has_level
  OXIDD_LINK_C(node_level)
  OXIDD_LINK_C(node_var)

  // boolean_function
  OXIDD_LINK_C(not)
  OXIDD_LINK_C(and)
  OXIDD_LINK_C(or)
  OXIDD_LINK_C(xor)
  OXIDD_LINK_C(nand)
  OXIDD_LINK_C(nor)
  OXIDD_LINK_C(equiv)
  OXIDD_LINK_C(imp)
  OXIDD_LINK_C(imp_strict)
  OXIDD_LINK_C(ite)

  OXIDD_LINK_C(cofactors)
  OXIDD_LINK_C(cofactor_true)
  OXIDD_LINK_C(cofactor_false)

  OXIDD_LINK_C(satisfiable)
  OXIDD_LINK_C(valid)
  OXIDD_LINK_C(sat_count_double)
  OXIDD_LINK_C(pick_cube)
  OXIDD_LINK_C(pick_cube_dd)
  OXIDD_LINK_C(pick_cube_dd_set)
  OXIDD_LINK_C(eval)
#undef OXIDD_LINK_C

  /// Create a ZBDD function from a function instance of the C API
  zbdd_function(capi::oxidd_zbdd_t c_function) noexcept
      : function(c_function) {}

public:
  /// Default constructor, yields an invalid ZBDD function
  zbdd_function() noexcept = default;

  /// Get the ZBDD Boolean function v for the singleton set {v}
  ///
  /// `this` must be a singleton set.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The ZBDD Boolean function v
  [[deprecated("use zbdd_manager::var instead"), nodiscard]] zbdd_function
  var_boolean_function() const noexcept {
    const std::optional<var_no_t> var = node_var();
    if (!var)
      return {};
    return containing_manager().var(*var);
  }

  /// Get the subset of `self` not containing `var`, formally
  /// `{s ∈ self | {var} ∉ s}`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD set (may be invalid if the operation runs out of memory)
  [[nodiscard]] zbdd_function subset0(var_no_t var) const noexcept {
    return capi::oxidd_zbdd_subset0(_func, var);
  }
  /// Get the subsets of `set` containing `var` with `var` removed afterwards,
  /// formally `{s ∖ {{var}} | s ∈ set ∧ var ∈ s}`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD set (may be invalid if the operation runs out of memory)
  [[nodiscard]] zbdd_function subset1(var_no_t var) const noexcept {
    return capi::oxidd_zbdd_subset1(_func, var);
  }
  /// Swap `subset0` and `subset1` with respect to `var`, formally
  /// `{s ∪ {{var}} | s ∈ set ∧ {var} ∉ s} ∪
  /// {s ∖ {{var}} | s ∈ set ∧ {var} ∈ s}`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The DD set (may be invalid if the operation runs out of memory)
  [[nodiscard]] zbdd_function change(var_no_t var) const noexcept {
    return capi::oxidd_zbdd_change(_func, var);
  }

  /// Create a new ZBDD node at the level of `this` with the given `hi` and `lo`
  /// edges
  ///
  /// `this` must be a singleton set.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The ZBDD set representing referencing the new node (may be
  ///           invalid if the operation runs out of memory)
  // NOLINTBEGIN(*-moved)
  [[nodiscard]] zbdd_function make_node(zbdd_function &&hi,
                                        zbdd_function &&lo) const noexcept {
    const capi::oxidd_zbdd_t h = hi._func, l = lo._func;
    hi._func._p = nullptr;
    lo._func._p = nullptr;
    return capi::oxidd_zbdd_make_node(_func, h, l);
  }
  // NOLINTEND(*-moved)
};

inline zbdd_function zbdd_manager::empty() const noexcept {
  assert(!is_invalid());
  return capi::oxidd_zbdd_empty(_manager);
}
inline zbdd_function zbdd_manager::base() const noexcept {
  assert(!is_invalid());
  return capi::oxidd_zbdd_base(_manager);
}

inline zbdd_function zbdd_manager::singleton(var_no_t var) const noexcept {
  assert(!is_invalid());
  return capi::oxidd_zbdd_singleton(_manager, var);
}

/// @cond
namespace bridge::detail {

extern "C" inline capi::oxidd_opt_zbdd_t
oxidd_iter_zbdd_callback_helper(void *data);

template <> struct iter_adapter<capi::oxidd_zbdd_t> {
  using c_iter_t = capi::oxidd_iter_zbdd_t;
  using c_opt_value_t = capi::oxidd_opt_zbdd_t;

  static constexpr auto helper = oxidd_iter_zbdd_callback_helper;

  static capi::oxidd_zbdd_t map(const zbdd_function &val) noexcept {
    return val.to_c_api();
  }
};

extern "C" inline capi::oxidd_opt_zbdd_t
oxidd_iter_zbdd_callback_helper(void *data) {
  auto *ctx = static_cast<c_iter_vtable<capi::oxidd_zbdd_t> *>(data);
  return ctx->next(ctx);
}

extern "C" inline capi::oxidd_opt_named_zbdd_t
oxidd_iter_named_zbdd_callback_helper(void *data);

template <> struct iter_adapter<capi::oxidd_named_zbdd_t> {
  using c_iter_t = capi::oxidd_iter_named_zbdd_t;
  using c_opt_value_t = capi::oxidd_opt_named_zbdd_t;

  static constexpr auto helper = oxidd_iter_named_zbdd_callback_helper;

  static capi::oxidd_named_zbdd_t map(const auto &val) noexcept {
    const std::string_view str(std::get<1>(val));
    const capi::oxidd_str_t name{.ptr = str.data(), .len = str.size()};
    return {.func = std::get<0>(val).to_c_api(), .name = name};
  }
};

extern "C" inline capi::oxidd_opt_named_zbdd_t
oxidd_iter_named_zbdd_callback_helper(void *data) {
  auto *ctx = static_cast<c_iter_vtable<capi::oxidd_named_zbdd_t> *>(data);
  return ctx->next(ctx);
}

} // namespace bridge::detail
/// @endcond

} // namespace oxidd
