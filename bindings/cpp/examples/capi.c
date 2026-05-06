/// @file   capi.c
/// @brief  Example of using OxiDD's C API

#include <assert.h>
#include <stdio.h>

#include <oxidd/capi.h>

#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))

int main(void) {
  const size_t inner_node_capacity = 65536;
  const size_t apply_cache_capacity = 1024;
  oxidd_bdd_manager_t manager =
      oxidd_bdd_manager_new(inner_node_capacity, apply_cache_capacity, 1);

  const char *var_names[] = {"x", "y", "z"};
  oxidd_var_no_range_t added_vars =
      oxidd_bdd_manager_add_named_vars(manager, var_names, ARRAY_LEN(var_names))
          .added_vars;
  assert(added_vars.start == 0 && added_vars.end == ARRAY_LEN(var_names));

  oxidd_bdd_t x = oxidd_bdd_var(manager, 0);
  oxidd_bdd_t z = oxidd_bdd_var(manager, 2);

  oxidd_bdd_t f = oxidd_bdd_and(x, z);
  // NOLINTNEXTLINE(*-bool-conversion) // bug llvm/llvm-project#195913
  puts(oxidd_bdd_satisfiable(f) ? "SAT" : "UNSAT");

  oxidd_bdd_unref(x);
  oxidd_bdd_unref(z);
  oxidd_bdd_unref(f);

  oxidd_bdd_manager_unref(manager);
}
