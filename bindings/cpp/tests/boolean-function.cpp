#include <oxidd/bcdd.hpp>
#include <oxidd/bdd.hpp>
#include <oxidd/concepts.hpp>
#include <oxidd/zbdd.hpp>

int main() {
  static_assert(oxidd::boolean_function_manager<oxidd::bdd_manager>);
  static_assert(oxidd::boolean_function_manager<oxidd::bcdd_manager>);
  static_assert(oxidd::boolean_function_manager<oxidd::zbdd_manager>);

  static_assert(oxidd::boolean_function<oxidd::bdd_function>);
  static_assert(oxidd::boolean_function<oxidd::bcdd_function>);
  static_assert(oxidd::boolean_function<oxidd::zbdd_set>);

  // TODO
}
