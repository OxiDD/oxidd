/// @file   visualize.cpp
/// @brief  Example of visualizing a BDD using OxiDD-vis

#include <array>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <span>

#include <oxidd/bdd.hpp>

int main(int argc, char **argv) {
  const std::span args(argv, argc);

  constexpr size_t inner_node_capacity = 65536, apply_cache_capacity = 65536;
  oxidd::bdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);

  // We are sure that the function names are distinct, so we ignore the result.
  (void)manager.add_named_vars(std::array{"x", "y", "z"});

  const std::array functions{manager.var(0) & manager.var(1)};

  oxidd::compat::expected<void, oxidd::util::error> result;
  // There are multiple versions of the visualization functionality. In this
  // example program, you can select which one is executed using the first
  // command line argument. (We include all these variants here to test that
  // they actually compile. In principle, there are two more variants allowing
  // to pass the begin and end iterator as two arguments rather than as a range.
  // The range versions call them internally.)
  switch (args.size() >= 2 ? std::atoi(args[1]) : 0) {
  case 1:
    // Visualize functions without giving them a name (I)
    //
    // From a technical perspective, this uses the overload with a `std::span`
    // argument.
    result = manager.visualize("foo", functions);
    break;

  case 2:
    // Visualize functions without giving them a name (II)
    //
    // There is a more general version, which takes a range over DD functions.
    // Here, we use `std::view::transform` to view the negated `functions`.
    result = manager.visualize(
        "foo (neg)",
        functions | std::views::transform(&oxidd::bdd_function::operator~));
    break;

  default:
    // This version accepts a range over pairs of a function and its name
    result = manager.visualize_with_names(
        "foo", std::array{std::make_pair(functions[0], "x & y"),
                          std::make_pair(manager.var(2), "z")});
  }

  if (!result.has_value()) {
    std::cerr << result.error() << '\n';
    return 1;
  }

  return 0;
}
