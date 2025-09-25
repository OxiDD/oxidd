/// @file   export.cpp
/// @brief  Example of exporting a BDD as DDDMP file

#include <array>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <span>

#include <oxidd/bdd.hpp>

int main(int argc, char **argv) {
  const std::span args(argv, argc);
  if (args.size() < 2) {
    std::cerr << "usage: " << args[0] << " <path> [<variant>]\n";
    return 1;
  }
  const std::string_view path(args[1]);

  constexpr size_t inner_node_capacity = 65536, apply_cache_capacity = 65536;
  oxidd::bdd_manager manager(inner_node_capacity, apply_cache_capacity, 1);

  // We are sure that the function names are distinct, so we ignore the result.
  (void)manager.add_named_vars(std::array{"x", "y", "z"});

  const std::array functions{manager.var(0) & manager.var(1)};

  oxidd::util::dddmp_export_settings settings;
  settings.diagram_name = "foo";
  settings.version = oxidd::util::dddmp_version::v3_0;

  oxidd::compat::expected<void, oxidd::util::error> result;
  // There are multiple versions of the export functionality. In this example
  // program, you can select which one is executed using the second command line
  // argument. (We include all these variants here to test that they actually
  // compile. In principle, there are two more variants allowing to pass the
  // begin and end iterator as two arguments rather than as a range. The range
  // versions call them internally.)
  switch (args.size() >= 3 ? std::atoi(args[2]) : 0) {
  case 1:
    // Export functions without giving them a name (I)
    //
    // From a technical perspective, this uses the overload with a `std::span`
    // argument.
    result = manager.export_dddmp(path, functions);
    break;
  case 2:
    // Same, but with settings
    result = manager.export_dddmp(path, functions, settings);
    break;

  case 3:
    // Export functions without giving them a name (II)
    //
    // There is a more general version, which takes a range over DD functions.
    // Here, we use `std::view::transform` to view the negated `functions`.
    result = manager.export_dddmp(
        path,
        functions | std::views::transform(&oxidd::bdd_function::operator~));
    break;
  case 4:
    result = manager.export_dddmp(
        path,
        functions | std::views::transform(&oxidd::bdd_function::operator~),
        settings);
    break;

  case 5: // NOLINT(*-magic-numbers)
    // This version accepts a range over pairs of a function and its name
    result = manager.export_dddmp_with_names(
        path, std::array{std::make_pair(functions[0], "x&y"),
                         std::make_pair(manager.var(2), "z")});
    break;
  default:
    result = manager.export_dddmp_with_names(
        path,
        std::array{std::make_pair(functions[0], "x&y"),
                   std::make_pair(manager.var(2), "z")},
        settings);
  }

  if (!result.has_value()) {
    std::cerr << result.error() << '\n';
    return 1;
  }

  return 0;
}
