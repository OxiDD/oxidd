/// @file   bdd.hpp
/// @brief  Reduced ordered binary decision diagrams (without complement edges)

#pragma once

#include <oxidd/bridge.hpp>

namespace oxidd {

class bdd_function;

/// Manager for binary decision diagrams (without complement edges)
///
/// Instances can safely be sent to other threads.
///
/// Models `oxidd::concepts::boolean_function_manager`
class bdd_manager
    : public bridge::manager<bdd_manager, bdd_function,
                             capi::oxidd_bdd_manager_t>,
      public bridge::reordering_manager<bdd_manager>,
      public bridge::boolean_function_manager<bdd_manager, bdd_function> {
  // friends
  friend class bridge::manager<bdd_manager, bdd_function,
                               capi::oxidd_bdd_manager_t>;
  friend class bridge::reordering_manager<bdd_manager>;
  friend class bridge::boolean_function_manager<bdd_manager, bdd_function>;

  friend class bridge::function<bdd_function, bdd_manager, capi::oxidd_bdd_t>;

  // C API functions
#define OXIDD_LINK_C(x)                                                        \
  static constexpr auto _c_##x = capi::oxidd_bdd_manager_##x;

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
#define OXIDD_LINK_C(x) static constexpr auto _c_##x = capi::oxidd_bdd_##x;
  // boolean_function_manager
  OXIDD_LINK_C(var)
  OXIDD_LINK_C(not_var)
  OXIDD_LINK_C(true)
  OXIDD_LINK_C(false)
#undef OXIDD_LINK_C

  /// Create a new BDD manager from a manager instance of the C API
  bdd_manager(capi::oxidd_bdd_manager_t c_manager) noexcept
      : manager(c_manager) {}

public:
  /// Default constructor, yields an invalid BDD manager
  bdd_manager() noexcept = default;

  /// Create a new BDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  bdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
              uint32_t threads) noexcept
      : manager(capi::oxidd_bdd_manager_new(inner_node_capacity,
                                            apply_cache_capacity, threads)) {}
};

/// Boolean function represented as a binary decision diagram (BDD, without
/// complement edges)
///
/// This is essentially a reference to a BDD node.
///
/// Instances can safely be sent to other threads.
///
/// Models `oxidd::concepts::boolean_function_quant` and
/// `oxidd::concepts::function_subst`
class bdd_function
    : public bridge::function<bdd_function, bdd_manager, capi::oxidd_bdd_t>,
      public bridge::has_level<bdd_function>,
      public bridge::function_subst<bdd_function>,
      public bridge::boolean_function<bdd_function>,
      public bridge::boolean_function_quant<bdd_function> {
  // friends
  friend class bridge::function<bdd_function, bdd_manager, capi::oxidd_bdd_t>;
  friend class bridge::function_subst<bdd_function>;
  friend class bridge::boolean_function_manager<bdd_manager, bdd_function>;
  friend class bridge::boolean_function<bdd_function>;
  friend class bridge::boolean_function_quant<bdd_function>;
  friend class bridge::has_level<bdd_function>;

  friend class bdd_manager;
  friend class bridge::manager<bdd_manager, bdd_function,
                               capi::oxidd_bdd_manager_t>;
  friend class bridge::substitution<bdd_function>;

  // C API functions
  using c_named_t = capi::oxidd_named_bdd_t;
#define OXIDD_LINK_C(x) static constexpr auto _c_##x = capi::oxidd_bdd_##x;
  // function
  OXIDD_LINK_C(ref)
  OXIDD_LINK_C(unref)
  OXIDD_LINK_C(containing_manager)
  OXIDD_LINK_C(node_count)

  // has_level
  OXIDD_LINK_C(node_level)
  OXIDD_LINK_C(node_var)

  // substitution
  using c_substitution_t = capi::oxidd_bdd_substitution_t;
  OXIDD_LINK_C(substitution_new)
  OXIDD_LINK_C(substitution_free)
  OXIDD_LINK_C(substitution_add_pair)
  // function_subst
  OXIDD_LINK_C(substitute)

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

  // boolean_function_quant
  OXIDD_LINK_C(forall)
  OXIDD_LINK_C(exists)
  OXIDD_LINK_C(unique)
  OXIDD_LINK_C(apply_forall)
  OXIDD_LINK_C(apply_exists)
  OXIDD_LINK_C(apply_unique)
#undef OXIDD_LINK_C

  /// Create a BDD function from a function instance of the C API
  bdd_function(capi::oxidd_bdd_t c_function) noexcept : function(c_function) {}

public:
  /// Default constructor, yields an invalid BDD function
  bdd_function() noexcept = default;
};

/// Substituion type for BDDs
///
/// @see  `bdd_function::substitute()`, `bridge::substitution`
using bdd_substitution = bridge::substitution<bdd_function>;

/// @cond
namespace bridge::detail {

extern "C" inline capi::oxidd_opt_bdd_t
oxidd_iter_bdd_callback_helper(void *data);

template <> struct iter_adapter<capi::oxidd_bdd_t> {
  using c_iter_t = capi::oxidd_iter_bdd_t;
  using c_opt_value_t = capi::oxidd_opt_bdd_t;

  static constexpr auto helper = oxidd_iter_bdd_callback_helper;

  static capi::oxidd_bdd_t map(const bdd_function &val) noexcept {
    return val.to_c_api();
  }
};

extern "C" inline capi::oxidd_opt_bdd_t
oxidd_iter_bdd_callback_helper(void *data) {
  auto *ctx = static_cast<c_iter_vtable<capi::oxidd_bdd_t> *>(data);
  return ctx->next(ctx);
}

extern "C" inline capi::oxidd_opt_named_bdd_t
oxidd_iter_named_bdd_callback_helper(void *data);

template <> struct iter_adapter<capi::oxidd_named_bdd_t> {
  using c_iter_t = capi::oxidd_iter_named_bdd_t;
  using c_opt_value_t = capi::oxidd_opt_named_bdd_t;

  static constexpr auto helper = oxidd_iter_named_bdd_callback_helper;

  static capi::oxidd_named_bdd_t map(const auto &val) noexcept {
    const std::string_view str(std::get<1>(val));
    const capi::oxidd_str_t name{.ptr = str.data(), .len = str.size()};
    return {.func = std::get<0>(val).to_c_api(), .name = name};
  }
};

extern "C" inline capi::oxidd_opt_named_bdd_t
oxidd_iter_named_bdd_callback_helper(void *data) {
  auto *ctx = static_cast<c_iter_vtable<capi::oxidd_named_bdd_t> *>(data);
  return ctx->next(ctx);
}

} // namespace bridge::detail
/// @endcond

} // namespace oxidd
