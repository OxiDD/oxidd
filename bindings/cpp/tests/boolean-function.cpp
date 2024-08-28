#undef NDEBUG // enable runtime assertions regardless of the build type

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include <oxidd/bcdd.hpp>
#include <oxidd/bdd.hpp>
#include <oxidd/concepts.hpp>
#include <oxidd/util.hpp>
#include <oxidd/zbdd.hpp>

using namespace oxidd;
using namespace oxidd::concepts;
using oxidd::util::boolean_operator;
using oxidd::util::slice;

// spell-checker:ignore nvars

using explicit_b_func = uint32_t;

namespace my_enumerate {

template <class V>
concept enumerable_view =
    std::ranges::view<V> && std::ranges::common_range<V> &&
    std::ranges::sized_range<V> && std::ranges::forward_range<V> &&
    std::move_constructible<std::ranges::range_reference_t<V>> &&
    std::move_constructible<std::ranges::range_rvalue_reference_t<V>>;

template <enumerable_view V>
class enumerate_view : public std::ranges::view_interface<enumerate_view<V>> {
  V _base = V();

  class iterator;

public:
  constexpr enumerate_view()
    requires std::default_initializable<V>
  = default;
  constexpr explicit enumerate_view(V base) : _base(std::move(base)) {};

  [[nodiscard]] constexpr iterator begin() const {
    return iterator(std::ranges::begin(_base), 0);
  }
  [[nodiscard]] constexpr iterator end() const {
    return iterator(std::ranges::end(_base), std::ranges::distance(_base));
  }

  [[nodiscard]] constexpr auto size() const { return std::ranges::size(_base); }
};

template <enumerable_view V> class enumerate_view<V>::iterator {
  friend class enumerate_view<V>;

public:
  using iterator_category = std::input_iterator_tag;
  using iterator_concept = std::forward_iterator_tag;
  using difference_type = std::ranges::range_difference_t<V>;
  using value_type = std::tuple<difference_type, std::ranges::range_value_t<V>>;

private:
  std::ranges::iterator_t<V> _current = std::ranges::iterator_t<V>();
  difference_type _pos = 0;

  constexpr explicit iterator(std::ranges::iterator_t<V> current,
                              difference_type pos)
      : _current(std::move(current)), _pos(pos) {}

public:
  iterator()
    requires std::default_initializable<std::ranges::iterator_t<V>>
  = default;

  constexpr auto operator*() const {
    return std::tuple<difference_type, std::ranges::range_reference_t<V>>(
        _pos, *_current);
  }

  constexpr iterator &operator++() {
    ++_current;
    ++_pos;
    return *this;
  }

  constexpr void operator++(int) {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  friend constexpr bool operator==(const iterator &x,
                                   const iterator &y) noexcept {
    return x._pos == y._pos;
  }
};

template <class R> enumerate_view(R &&) -> enumerate_view<std::views::all_t<R>>;

template <std::ranges::common_range R>
  requires(std::ranges::sized_range<R> && std::ranges::forward_range<R>)
auto enumerate(R &&base) {
  return enumerate_view(std::forward<R>(base));
}

} // namespace my_enumerate

// TODO: replace this by `using std::views::enumerate` once enumerate is
// available in libc++ and the respective libc++ version is included in macOS
using my_enumerate::enumerate;

/// C++ translation `TestAllBooleanFunctions` form
/// `crates/oxidd/tests/boolean_function.rs`
template <boolean_function_manager M> class test_all_boolean_functions {
  M _mgr;
  slice<typename M::function> _vars, _var_handles;
  /// Stores all possible Boolean functions over `vars.size()` variables
  std::vector<typename M::function> _boolean_functions;
  /// Map from Boolean functions as decision diagrams to their explicit (truth
  /// table) representations
  std::unordered_map<typename M::function, explicit_b_func> _dd_to_boolean_func;
  /// Map from variables (`0..vars.size()`) to Boolean functions
  ///
  /// Example for three variables: `[0b01010101, 0b00110011, 0b00001111]`
  std::vector<explicit_b_func> _var_functions;

public:
  /// Initialize the test, generating DDs for all Boolean functions for the
  /// given variable set. `vars` are the Boolean functions representing the
  /// variables identified by `var_handles`. For BDDs, the two coincide, but
  /// not for ZBDDs.
  test_all_boolean_functions(M mgr, slice<typename M::function> vars,
                             slice<typename M::function> var_handles)
      : _mgr(std::move(mgr)), _vars(vars), _var_handles(var_handles) {
    assert(vars.size() == var_handles.size());
    assert(std::bit_width(
               (unsigned)std::numeric_limits<explicit_b_func>::digits) >=
               vars.size() &&
           "too many variables");
    // actually, only 2 are possible in a feasible amount of time (with
    // substitution)

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

    _var_functions.reserve(nvars);
    for (unsigned i = 0; i < nvars; ++i) {
      explicit_b_func f = 0;
      for (unsigned assignment = 0; assignment < num_assignments; ++assignment)
        f |= explicit_b_func{(assignment >> i) & 1} << assignment;
      _var_functions.push_back(f);
    }
  }

private:
  typename M::function make_cube(unsigned positive, unsigned negative) const {
    assert((positive & negative) == 0);

    typename M::function cube = _boolean_functions.back(); // ⊤
    for (const auto &[i, var] : enumerate(_vars)) {
      if (((positive >> i) & 1) != 0) {
        cube &= var;
      } else if (((negative >> i) & 1) != 0) {
        cube &= ~var;
      }
    }

    assert(!cube.is_invalid());
    return cube;
  }

public:
  /// Test basic operations on all Boolean functions
  void basic() const {
    const unsigned nvars = _vars.size();
    const unsigned num_assignments = 1 << nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    const explicit_b_func func_mask = num_functions - 1;

    // false & true
    assert(_mgr.f() == _boolean_functions.front());
    assert(_mgr.t() == _boolean_functions.back());

    // vars
    for (const auto &[i, var] : enumerate(_vars)) {
      explicit_b_func expected = 0;
      for (unsigned assignment = 0; assignment < num_assignments; ++assignment)
        expected |= ((assignment >> i) & 1) << assignment;
      const explicit_b_func actual = _dd_to_boolean_func.at(var);
      assert(actual == expected);
    }

    // arity >= 1
    for (const auto &[f_explicit, f] : enumerate(_boolean_functions)) {

      /* not */ {
        const explicit_b_func expected = ~f_explicit & func_mask;
        const explicit_b_func actual = _dd_to_boolean_func.at(~f);
        assert(actual == expected);
      }

      // arity >= 2
      for (const auto &[g_explicit, g] : enumerate(_boolean_functions)) {
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
        for (const auto &[h_explicit, h] : enumerate(_boolean_functions)) {
          /* ite */ {
            const explicit_b_func expected =
                (f_explicit & g_explicit) | (~f_explicit & h_explicit);
            const explicit_b_func actual = _dd_to_boolean_func.at(f.ite(g, h));
            assert(actual == expected);
          }
        }
      }

      /* pick_cube() etc. */ {
        // This is a stripped-down version of the Rust test; we only test that
        // the results of `pick_cube()` and `pick_cube_symbolic()` agree
        // (therefore, both can be represented as a conjunction of literals),
        // and that they imply `f`.
        const util::assignment cube = f.pick_cube();
        const explicit_b_func actual =
            _dd_to_boolean_func.at(f.pick_cube_symbolic());

        if (f_explicit == 0) {
          assert(actual == 0);
          assert(cube.size() == 0);
        } else {
          assert(cube.size() == nvars);
          assert((actual & ~f_explicit) == 0);

          explicit_b_func cube_func = func_mask;
          for (unsigned var = 0; var < nvars; ++var) {
            switch (cube[var]) {
            case util::opt_bool::NONE:
              break;
            case util::opt_bool::FALSE:
              cube_func &= ~_var_functions[var];
              break;
            case util::opt_bool::TRUE:
              cube_func &= _var_functions[var];
              break;
            }
          }

          assert(cube_func == actual);
        }

        for (unsigned pos = 0; pos < num_assignments; ++pos) {
          for (unsigned neg = 0; neg < num_assignments; ++neg) {
            if ((pos & neg) != 0)
              continue;

            const explicit_b_func actual = _dd_to_boolean_func.at(
                f.pick_cube_symbolic_set(make_cube(pos, neg)));

            if (f_explicit == 0) {
              assert(actual == 0);
            } else {
              assert((actual & !f_explicit) == 0);

              for (const auto &[var, var_func] : enumerate(_var_functions)) {
                if ((actual & var_func) >> (1 << var) == (actual & ~var_func))
                  continue; // var is don't care
                explicit_b_func flipped = 0;
                if ((actual & var_func) == 0) { // selected to be false
                  if ((pos & (1 << var)) == 0)
                    continue; // was not requested to be true
                  flipped = actual << (1 << var);
                } else {
                  assert((actual & ~var_func) == 0 &&
                         "not a conjunction of literals");
                  // selected to be false
                  if ((neg & (1 << var)) == 0)
                    continue; // was not requested to be true
                  flipped = actual >> (1 << var);
                }

                // If the variable was selected to be the opposite of the
                // request, then the reason must be that the cube would not have
                // implied the function.
                assert((flipped & ~f_explicit) != 0);
              }
            }
          }
        }
      }
    }
  }

private:
  void subst_rec(std::vector<std::optional<explicit_b_func>> &replacements,
                 uint32_t current_var) const
    requires(function_subst<typename M::function>)
  {
    assert(replacements.size() == _vars.size());
    if (current_var < _vars.size()) {
      replacements[current_var] = {};
      subst_rec(replacements, current_var + 1);
      for (const explicit_b_func f :
           std::views::iota(0U, _boolean_functions.size())) {
        replacements[current_var] = {f};
        subst_rec(replacements, current_var + 1);
      }
    } else {
      const unsigned nvars = _vars.size();
      const unsigned num_assignments = 1 << nvars;

      const typename M::function::substitution subst(
          enumerate(replacements) | std::views::filter([](const auto p) {
            return std::get<1>(p).has_value();
          }) |
          std::views::transform([this](const auto p) {
            const auto [i, repl] = p;
            return std::make_pair(_var_handles[i], _boolean_functions[*repl]);
          }));

      for (const auto [f_explicit, f] : enumerate(_boolean_functions)) {
        explicit_b_func expected = 0;
        // To compute the expected truth table, we first compute a mapped
        // assignment that we look up in the truth table for `f`
        for (const unsigned assignment :
             std::views::iota(0U, num_assignments)) {
          unsigned mapped_assignment = 0;
          for (const auto [var, repl] : enumerate(replacements)) {
            const unsigned val =
                (repl ?
                      // replacement function evaluated for `assignment`
                     *repl >> assignment
                      :
                      // `var` is set in `assignment`?
                     assignment >> var) &
                1;
            mapped_assignment |= val << var;
          }
          expected |= ((f_explicit >> mapped_assignment) & 1) << assignment;
        }

        const explicit_b_func actual =
            _dd_to_boolean_func.at(f.substitute(subst));
        assert(actual == expected);
      }
    }
  }

public:
  /// Test all possible substitutions
  void subst() const
    requires(function_subst<typename M::function>)
  {
    std::vector<std::optional<explicit_b_func>> replacements(_vars.size());
    subst_rec(replacements, 0);
  }

  /// Test quantification operations on all Boolean functions
  void quant() const
    requires(boolean_function_quant<typename M::function>)
  {
    const unsigned nvars = _vars.size();
    const unsigned num_assignments = 1 << nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    const explicit_b_func func_mask = num_functions - 1;

    // TODO: restrict (once we have it in the C/C++ API)

    // quantification
    std::vector<explicit_b_func> assignment_to_mask(num_assignments);
    for (unsigned var_set = 0; var_set < num_assignments; ++var_set) {
      typename M::function dd_var_set = _mgr.t();
      for (const auto &[i, var] : enumerate(_vars)) {
        if ((var_set & (1 << i)) != 0)
          dd_var_set &= var;
      }
      assert(!dd_var_set.is_invalid());

      // precompute `assignment_to_mask`
      for (const auto &[assignment, mask] : enumerate(assignment_to_mask)) {
        explicit_b_func tmp = func_mask;
        for (const auto [i, func] : enumerate(_var_functions)) {
          if (((var_set >> i) & 1) != 0)
            continue;
          const explicit_b_func f = _var_functions[i];
          tmp &= ((assignment >> i) & 1) != 0 ? f : ~f;
        }
        mask = tmp;
      }

      for (const auto &[f_explicit, f] : enumerate(_boolean_functions)) {
        explicit_b_func exist_expected = 0, forall_expected = 0,
                        unique_expected = 0;
        for (unsigned assignment = 0; assignment < num_assignments;
             ++assignment) {
          const explicit_b_func mask = assignment_to_mask[assignment];
          const explicit_b_func bit = 1 << assignment;

          // or of all bits under mask
          if ((f_explicit & mask) != 0)
            exist_expected |= bit;
          // and of all bits under mask
          if ((f_explicit & mask) == mask)
            forall_expected |= bit;
          // xor of all bits under mask
          if ((std::popcount(explicit_b_func(f_explicit) & mask) & 1) != 0)
            unique_expected |= bit;
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

        for (const auto &g : _boolean_functions) {
          for (const boolean_operator op :
               {boolean_operator::AND, boolean_operator::OR,
                boolean_operator::XOR, boolean_operator::EQUIV,
                boolean_operator::NAND, boolean_operator::NOR,
                boolean_operator::IMP, boolean_operator::IMP_STRICT}) {
            typename M::function inner;
            switch (op) {
            case boolean_operator::AND:
              inner = f & g;
              break;
            case boolean_operator::OR:
              inner = f | g;
              break;
            case boolean_operator::XOR:
              inner = f ^ g;
              break;
            case boolean_operator::EQUIV:
              inner = f.equiv(g);
              break;
            case boolean_operator::NAND:
              inner = f.nand(g);
              break;
            case boolean_operator::NOR:
              inner = f.nor(g);
              break;
            case boolean_operator::IMP:
              inner = f.imp(g);
              break;
            case boolean_operator::IMP_STRICT:
              inner = f.imp_strict(g);
              break;
            }

            assert(f.apply_forall(op, g, dd_var_set) ==
                   inner.forall(dd_var_set));
            assert(f.apply_exist(op, g, dd_var_set) == inner.exist(dd_var_set));
            assert(f.apply_unique(op, g, dd_var_set) ==
                   inner.unique(dd_var_set));
          }
        }
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
  test.subst();
  test.quant();
}

void bcdd_all_boolean_functions_2vars_t1() {
  // NOLINTNEXTLINE(*-magic-numbers)
  bcdd_manager mgr(65536, 1024, 1);
  const std::array vars{mgr.new_var(), mgr.new_var()};
  const test_all_boolean_functions test(mgr, vars, vars);
  test.basic();
  test.subst();
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
