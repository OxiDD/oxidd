#undef NDEBUG // enable runtime assertions regardless of the build type

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <oxidd/bcdd.hpp>
#include <oxidd/bdd.hpp>
#include <oxidd/concepts.hpp>
#include <oxidd/zbdd.hpp>

using namespace oxidd;
using namespace oxidd::concepts;
using oxidd::util::slice;

// spell-checker:ignore nvars

using explicit_b_func = uint32_t;

/// C++ translation `TestAllBooleanFunctions` form
/// `crates/oxidd/tests/boolean_function.rs`
template <boolean_function_manager M> class test_all_boolean_functions {
  M _mgr;
  slice<typename M::function> _vars;
  /// Stores all possible Boolean functions with `vars.size()` vars
  std::vector<typename M::function> _boolean_functions;
  std::unordered_map<typename M::function, explicit_b_func> _dd_to_boolean_func;

public:
  test_all_boolean_functions(M mgr, slice<typename M::function> vars,
                             slice<typename M::function> var_handles)
      : _mgr(std::move(mgr)), _vars(vars) {
    assert(vars.size() == var_handles.size());
    assert(std::bit_width(
               (unsigned)std::numeric_limits<explicit_b_func>::digits) >=
               vars.size() &&
           "too many variables");
    // actually, only 3 are possible in a feasible amount of time

    const unsigned nvars = vars.size();
    const unsigned num_assignments = 1 << nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    _boolean_functions.reserve(num_functions);
    _dd_to_boolean_func.reserve(num_functions);

    std::vector<std::pair<typename M::function, bool>> args;
    args.reserve(nvars);
    for (const typename M::function &var : var_handles)
      args.emplace_back(var, false);

    for (explicit_b_func explicit_f = 0; explicit_f < num_functions;
         ++explicit_f) {
      // naïve DD construction from the on-set
      typename M::function f = _mgr.f();
      for (unsigned assignment = 0; assignment < num_assignments;
           ++assignment) {
        if ((explicit_f & (1 << assignment)) == 0)
          continue; // not part of the on-set
        typename M::function cube = _mgr.t();

        for (unsigned var = 0; var < nvars; ++var) {
          typename M::function v = _vars[var];
          if ((assignment & (1 << var)) == 0)
            v = ~v;
          cube &= v;
          assert(!cube.is_invalid());
        }

        f |= cube;
        assert(!f.is_invalid());
      }

      // check that evaluating the function yields the desired values
      for (unsigned assignment = 0; assignment < num_assignments;
           ++assignment) {
        const bool expected = (explicit_f & (1 << assignment)) != 0;
        for (unsigned var = 0; var < nvars; ++var)
          args[var].second = (assignment & (1 << var)) != 0;
        const bool actual = f.eval(args);
        assert(actual == expected);
        assert(f.sat_count_double(nvars) == std::popcount(explicit_f));
      }

      _boolean_functions.push_back(f);
      const auto [_, inserted] =
          _dd_to_boolean_func.emplace(std::move(f), explicit_f);
      assert(inserted &&
             "two different Boolean functions have the same representation");
    }
  }

  /// Test basic operations on all Boolean function with the given variable set
  void basic() const {
    const unsigned nvars = _vars.size();
    const unsigned num_assignments = 1 << nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    const explicit_b_func func_mask = num_functions - 1;

    // false & true
    assert(_mgr.f() == _boolean_functions.front());
    assert(_mgr.t() == _boolean_functions.back());

    // vars
    for (unsigned var = 0; var < nvars; ++var) {
      explicit_b_func expected = 0;
      for (unsigned assignment = 0; assignment < num_assignments; ++assignment)
        expected |= ((assignment >> var) & 1) << assignment;
      const explicit_b_func actual = _dd_to_boolean_func.at(_vars[var]);
      assert(actual == expected);
    }

    // arity >= 1
    for (explicit_b_func f_explicit = 0; f_explicit < num_functions;
         ++f_explicit) {
      const typename M::function &f = _boolean_functions[f_explicit];

      /* not */ {
        const explicit_b_func expected = ~f_explicit & func_mask;
        const explicit_b_func actual = _dd_to_boolean_func.at(~f);
        assert(actual == expected);
      }

      // arity >= 2
      for (explicit_b_func g_explicit = 0; g_explicit < num_functions;
           ++g_explicit) {
        const typename M::function &g = _boolean_functions[g_explicit];

        /* and */ {
          const explicit_b_func expected = f_explicit & g_explicit;
          const explicit_b_func actual = _dd_to_boolean_func.at(f & g);
          assert(actual == expected);
        }
        /* or */ {
          const explicit_b_func expected = f_explicit | g_explicit;
          const explicit_b_func actual = _dd_to_boolean_func.at(f | g);
          assert(actual == expected);
        }
        /* xor */ {
          const explicit_b_func expected = f_explicit ^ g_explicit;
          const explicit_b_func actual = _dd_to_boolean_func.at(f ^ g);
          assert(actual == expected);
        }
        /* equiv */ {
          const explicit_b_func expected =
              ~(f_explicit ^ g_explicit) & func_mask;
          const explicit_b_func actual = _dd_to_boolean_func.at(f.equiv(g));
          assert(actual == expected);
        }
        /* nand */ {
          const explicit_b_func expected =
              ~(f_explicit & g_explicit) & func_mask;
          const explicit_b_func actual = _dd_to_boolean_func.at(f.nand(g));
          assert(actual == expected);
        }
        /* nor */ {
          const explicit_b_func expected =
              ~(f_explicit | g_explicit) & func_mask;
          const explicit_b_func actual = _dd_to_boolean_func.at(f.nor(g));
          assert(actual == expected);
        }
        /* implication */ {
          const explicit_b_func expected =
              (~f_explicit | g_explicit) & func_mask;
          const explicit_b_func actual = _dd_to_boolean_func.at(f.imp(g));
          assert(actual == expected);
        }
        /* strict implication */ {
          const explicit_b_func expected = ~f_explicit & g_explicit;
          const explicit_b_func actual =
              _dd_to_boolean_func.at(f.imp_strict(g));
          assert(actual == expected);
        }

        // arity >= 3
        for (explicit_b_func h_explicit = 0; h_explicit < num_functions;
             ++h_explicit) {
          const typename M::function &h = _boolean_functions[h_explicit];

          /* ite */ {
            const explicit_b_func expected =
                (f_explicit & g_explicit) | (~f_explicit & h_explicit);
            const explicit_b_func actual = _dd_to_boolean_func.at(f.ite(g, h));
            assert(actual == expected);
          }
        }
      }
    }
  }

  /// Test quantification operations on all Boolean function with the given
  /// variable set
  void quant() const
    requires(boolean_function_quant<typename M::function>)
  {
    const unsigned nvars = _vars.size();
    const unsigned num_assignments = 1 << nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    const explicit_b_func func_mask = num_functions - 1;

    // Example for 3 vars: [0b01010101, 0b00110011, 0b00001111]
    std::vector<explicit_b_func> var_functions;
    var_functions.reserve(nvars);
    for (unsigned i = 0; i < nvars; ++i) {
      explicit_b_func f = 0;
      for (unsigned assignment = 0; assignment < num_assignments; ++assignment)
        f |= explicit_b_func{(assignment >> i) & 1} << assignment;
      var_functions.push_back(f);
    }

    // TODO: restrict (once we have it in the C/C++ API)

    // quantification
    std::vector<explicit_b_func> assignment_to_mask(num_assignments);
    for (unsigned var_set = 0; var_set < num_assignments; ++var_set) {
      typename M::function dd_var_set = _mgr.t();
      for (unsigned i = 0; i < nvars; ++i) {
        if ((var_set & (1 << i)) != 0)
          dd_var_set &= _vars[i];
      }

      // precompute `assignment_to_mask`
      for (unsigned assignment = 0; assignment < num_assignments;
           ++assignment) {
        explicit_b_func mask = func_mask;
        for (unsigned i = 0; i < nvars; ++i) {
          if (((var_set >> i) & 1) != 0)
            continue;
          const explicit_b_func f = var_functions[i];
          mask &= ((assignment >> i) & 1) != 0 ? f : ~f;
        }
        assignment_to_mask[assignment] = mask;
      }

      for (explicit_b_func f_explicit = 0; f_explicit < num_functions;
           ++f_explicit) {
        const typename M::function &f = _boolean_functions[f_explicit];

        explicit_b_func exist_expected = 0, forall_expected = 0,
                        unique_expected = 0;
        for (unsigned assignment = 0; assignment < num_assignments;
             ++assignment) {
          const explicit_b_func mask = assignment_to_mask[assignment];

          // or of all bits under mask
          exist_expected |= explicit_b_func{(f_explicit & mask) != 0}
                            << assignment;
          // and of all bits under mask
          forall_expected |= explicit_b_func{(f_explicit & mask) == mask}
                             << assignment;
          // xor of all bits under mask
          unique_expected |=
              static_cast<explicit_b_func>(std::popcount(f_explicit & mask) & 1)
              << assignment;
        }

        const explicit_b_func exist_actual =
            _dd_to_boolean_func.at(f.exist(dd_var_set));
        assert(exist_actual == exist_expected);

        const explicit_b_func forall_actual =
            _dd_to_boolean_func.at(f.forall(dd_var_set));
        assert(forall_actual == forall_expected);

        const explicit_b_func unique_actual =
            _dd_to_boolean_func.at(f.unique(dd_var_set));
        assert(unique_actual == unique_expected);
      }
    }
  }
};

void bdd_all_boolean_functions_2vars_t1() {
  // NOLINTNEXTLINE(*-magic-numbers)
  bdd_manager mgr(65536, 1024, 1);
  const std::array vars{mgr.new_var(), mgr.new_var()};
  const test_all_boolean_functions test(mgr, vars, vars);
  test.basic();
  test.quant();
}

void bcdd_all_boolean_functions_2vars_t1() {
  // NOLINTNEXTLINE(*-magic-numbers)
  bcdd_manager mgr(65536, 1024, 1);
  const std::array vars{mgr.new_var(), mgr.new_var()};
  const test_all_boolean_functions test(mgr, vars, vars);
  test.basic();
  test.quant();
}

void zbdd_all_boolean_functions_2vars_t1() {
  // NOLINTNEXTLINE(*-magic-numbers)
  zbdd_manager mgr(65536, 1024, 1);
  const std::array singletons{mgr.new_singleton(), mgr.new_singleton()};
  const std::array vars{singletons[0].var_boolean_function(),
                        singletons[1].var_boolean_function()};
  const test_all_boolean_functions test(mgr, vars, singletons);
  test.basic();
}

int main() {
  static_assert(boolean_function_manager<bdd_manager>);
  static_assert(boolean_function_manager<bcdd_manager>);
  static_assert(boolean_function_manager<zbdd_manager>);

  static_assert(boolean_function_quant<bdd_function>);
  static_assert(boolean_function_quant<bcdd_function>);
  static_assert(boolean_function<zbdd_function>);

  bdd_all_boolean_functions_2vars_t1();
  bcdd_all_boolean_functions_2vars_t1();
  zbdd_all_boolean_functions_2vars_t1();
}
