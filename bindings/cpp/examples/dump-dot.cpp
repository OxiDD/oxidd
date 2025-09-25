/// @file   dump-dot.cpp
/// @brief  Example of dumping all nodes in a manager as a DDDMP file

#include <array>
#include <cstdlib>
#include <iostream>
#include <span>

#include <oxidd/bcdd.hpp>
#include <oxidd/bdd.hpp>
#include <oxidd/concepts.hpp>
#include <oxidd/zbdd.hpp>

namespace {

template <oxidd::concepts::boolean_function_manager M>
int dump_dot_main(M &manager, std::string_view path, bool names) {
  using func = M::function;

  // We are sure that the function names are distinct, so we ignore the result.
  (void)manager.add_named_vars(std::array{"x", "y", "z"});

  const func f = manager.var(0) & manager.var(1);
  const func g = ~f | manager.not_var(2);

  auto result =
      names ? manager.dump_all_dot(path)
            : manager.dump_all_dot(path, std::array{std::make_pair(f, "f")});

  if (!result.has_value()) {
    std::cerr << result.error() << '\n';
    return 1;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::span args(argv, argc);
  if (args.size() < 3) {
    std::cerr << "usage: " << args[0] << " bdd|bcdd|zbdd <path> [--names]\n";
    return 1;
  }
  const std::string_view dd_kind(args[1]);
  const std::string_view path(args[2]);
  const bool names = args.size() > 3;

  constexpr size_t inner_node_capacity = 1024L * 1024L,
                   apply_cache_capacity = 65536;

  if (dd_kind == std::string_view("bdd")) {
    oxidd::bdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);
    return dump_dot_main(manager, path, names);
  }
  if (dd_kind == std::string_view("bcdd")) {
    oxidd::bcdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);
    return dump_dot_main(manager, path, names);
  }
  if (dd_kind == std::string_view("zbdd")) {
    oxidd::zbdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);
    return dump_dot_main(manager, path, names);
  }

  std::cerr << "Unknown DD kind '" << dd_kind << "'\n";
  return 1;
}
