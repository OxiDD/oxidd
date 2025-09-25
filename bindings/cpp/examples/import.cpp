/// @file   import.cpp
/// @brief  Example importing a DDDMP file and counting the nodes of each
///         present function

#include <cstddef>
#include <iostream>
#include <ranges>
#include <span>
#include <string_view>

#include <oxidd/bcdd.hpp>
#include <oxidd/bdd.hpp>
#include <oxidd/concepts.hpp>
#include <oxidd/zbdd.hpp>

namespace {

template <oxidd::concepts::boolean_function_manager M>
  requires(oxidd::concepts::reordering_manager<M>)
int import(M &manager, oxidd::util::dddmp_file &f) {
  if (f.has_var_names()) {
    auto result = manager.add_named_vars(f.var_names());
    if (!result.has_value()) {
      std::cerr << result.error() << '\n';
      return 1;
    }
  } else {
    manager.add_vars(f.num_vars());
  }

  manager.set_var_order(f.support_var_order());
  auto result = manager.import_dddmp(f);
  if (!result.has_value()) {
    std::cerr << "error during import: " << result.error() << '\n';
    return 1;
  }
  const std::vector<typename M::function> functions(std::move(result).value());

  for (std::ptrdiff_t i = 0; i < functions.size(); ++i) {
    std::cout << "function " << i << " '" << f.root_names()[i] << "' has "
              << f.num_nodes() << " nodes\n";
  }

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::span args(argv, argc);
  if (args.size() != 3) {
    std::cerr << "usage: " << args[0] << " bdd|bcdd|zbdd <input.dddmp>\n";
    return 1;
  }
  const std::string_view dd_kind(args[1]);
  const std::string_view file_name(args[2]);

  auto res = oxidd::util::dddmp_file::open(file_name);
  if (!res.has_value()) {
    std::cerr << "Could not open '" << file_name << "': " << res.error()
              << "\n";
    return 1;
  }

  constexpr size_t inner_node_capacity = 1024L * 1024L,
                   apply_cache_capacity = 65536;

  if (dd_kind == std::string_view("bdd")) {
    oxidd::bdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);
    return import(manager, *res);
  }
  if (dd_kind == std::string_view("bcdd")) {
    oxidd::bcdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);
    return import(manager, *res);
  }
  if (dd_kind == std::string_view("zbdd")) {
    oxidd::zbdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);
    return import(manager, *res);
  }

  std::cerr << "Unknown DD kind '" << dd_kind << "'\n";
  return 1;
}
