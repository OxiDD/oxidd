/// @file   boolean-function.cpp
/// @brief  Test all Boolean functions over a fixed number of variables

#undef NDEBUG // enable runtime assertions regardless of the build type

#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <oxidd/bcdd.hpp>
#include <oxidd/bdd.hpp>
#include <oxidd/concepts.hpp>
#include <oxidd/zbdd.hpp>

using namespace oxidd;
using namespace oxidd::concepts;
using oxidd::util::boolean_operator;

// spell-checker:ignore nvars

namespace {

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
  // clang-format off
  // Ubuntu 24.04. has clang-format 18.1.3, which wants to remove the space
  // before `{}`. clang-format 18.1.8, however, wants the space.
  constexpr explicit enumerate_view(V base) : _base(std::move(base)) {};
  // clang-format on

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
  var_no_t _nvars;
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
  test_all_boolean_functions(M mgr)
      : _mgr(std::move(mgr)), _nvars(_mgr.num_vars()) {
    const var_no_t nvars = _nvars;
    assert(std::bit_width(
               (var_no_t)std::numeric_limits<explicit_b_func>::digits) >=
               nvars &&
           "too many variables");
    // actually, only 2 are possible in a feasible amount of time (with
    // substitution)

    const unsigned num_assignments = 1 << nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    _boolean_functions.reserve(num_functions);
    _dd_to_boolean_func.reserve(num_functions);

    std::vector<std::pair<var_no_t, bool>> args;
    args.reserve(nvars);
    for (var_no_t var = 0; var < nvars; ++var)
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
          typename M::function v = _mgr.var(var);
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
        for (var_no_t var = 0; var < nvars; ++var)
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
    for (var_no_t i = 0; i < nvars; ++i) {
      explicit_b_func f = 0;
      for (unsigned assignment = 0; assignment < num_assignments; ++assignment)
        f |= explicit_b_func{(assignment >> i) & 1} << assignment;
      _var_functions.push_back(f);
    }
  }

private:
  [[nodiscard]] explicit_b_func _make_cube(unsigned positive,
                                           unsigned negative) const {
    assert((positive & negative) == 0);

    explicit_b_func cube = (1 << (1 << _nvars)) - 1; // ⊤
    for (const auto [i, var] : enumerate(_var_functions)) {
      if (((positive >> i) & 1) != 0) {
        cube &= var;
      } else if (((negative >> i) & 1) != 0) {
        cube &= ~var;
      }
    }

    return cube;
  }

public:
  /// Test basic operations on all Boolean functions
  void basic() const {
    const unsigned num_assignments = 1 << _nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    const explicit_b_func func_mask = num_functions - 1;

    // false & true
    assert(_mgr.f() == _boolean_functions.front());
    assert(_mgr.t() == _boolean_functions.back());

    // vars
    for (const auto &[i, expected] : enumerate(_var_functions)) {
      assert(_dd_to_boolean_func.at(_mgr.var(i)) == expected);
      assert(_dd_to_boolean_func.at(_mgr.not_var(i)) == (expected ^ func_mask));
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
          typename M::function tmp = f;
          tmp &= g;
          assert(_dd_to_boolean_func.at(tmp) == actual);
        }
        /* or */ {
          const explicit_b_func expected = f_explicit | g_explicit;
          const explicit_b_func actual = _dd_to_boolean_func.at(f | g);
          assert(actual == expected);
          typename M::function tmp = f;
          tmp |= g;
          assert(_dd_to_boolean_func.at(tmp) == actual);
        }
        /* xor */ {
          const explicit_b_func expected = f_explicit ^ g_explicit;
          const explicit_b_func actual = _dd_to_boolean_func.at(f ^ g);
          assert(actual == expected);
          typename M::function tmp = f;
          tmp ^= g;
          assert(_dd_to_boolean_func.at(tmp) == actual);
        }
        /* set difference (strict implication with operands swapped) */ {
          const explicit_b_func expected = f_explicit & ~g_explicit;
          const explicit_b_func actual = _dd_to_boolean_func.at(f - g);
          assert(actual == expected);
          typename M::function tmp = f;
          tmp -= g;
          assert(_dd_to_boolean_func.at(tmp) == actual);
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
        // the results of `pick_cube()` and `pick_cube_dd()` agree (therefore,
        // both can be represented as a conjunction of literals), and that they
        // imply `f`.
        const util::assignment cube = f.pick_cube();
        const explicit_b_func actual = _dd_to_boolean_func.at(f.pick_cube_dd());

        if (f_explicit == 0) {
          assert(actual == 0);
          assert(cube.size() == 0);
        } else {
          assert(cube.size() == _nvars);
          assert((actual & ~f_explicit) == 0);

          explicit_b_func cube_func = func_mask;
          for (var_no_t var = 0; var < _nvars; ++var) {
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
                f.pick_cube_dd_set(_boolean_functions[_make_cube(pos, neg)]));

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
  void _subst_rec(std::vector<std::optional<explicit_b_func>> &replacements,
                  uint32_t current_var) const
    requires(function_subst<typename M::function>)
  {
    assert(replacements.size() == _nvars);
    if (current_var < _nvars) {
      replacements[current_var] = {};
      _subst_rec(replacements, current_var + 1);
      for (const explicit_b_func f :
           std::views::iota(0U, _boolean_functions.size())) {
        replacements[current_var] = {f};
        _subst_rec(replacements, current_var + 1);
      }
    } else {
      const unsigned num_assignments = 1 << _nvars;

      const typename M::function::substitution subst(
          enumerate(replacements) | std::views::filter([](const auto p) {
            return std::get<1>(p).has_value();
          }) |
          std::views::transform([this](const auto p) {
            const auto [i, repl] = p;
            return std::make_pair(i, _boolean_functions[*repl]);
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
    std::vector<std::optional<explicit_b_func>> replacements(_nvars);
    _subst_rec(replacements, 0);
  }

  /// Test quantification operations on all Boolean functions
  void quant() const
    requires(boolean_function_quant<typename M::function>)
  {
    const unsigned num_assignments = 1 << _nvars;
    const explicit_b_func num_functions = 1 << num_assignments;
    const explicit_b_func func_mask = num_functions - 1;

    // TODO: restrict (once we have it in the C/C++ API)

    // quantification
    std::vector<explicit_b_func> assignment_to_mask(num_assignments);
    for (unsigned var_set = 0; var_set < num_assignments; ++var_set) {
      explicit_b_func explicit_var_set = func_mask;
      for (const auto &[i, var] : enumerate(_var_functions)) {
        if ((var_set & (1 << i)) != 0)
          explicit_var_set &= var;
      }
      const typename M::function dd_var_set =
          _boolean_functions[explicit_var_set];

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
        explicit_b_func exists_expected = 0, forall_expected = 0,
                        unique_expected = 0;
        for (unsigned assignment = 0; assignment < num_assignments;
             ++assignment) {
          const explicit_b_func mask = assignment_to_mask[assignment];
          const explicit_b_func bit = 1 << assignment;

          // or of all bits under mask
          if ((f_explicit & mask) != 0)
            exists_expected |= bit;
          // and of all bits under mask
          if ((f_explicit & mask) == mask)
            forall_expected |= bit;
          // xor of all bits under mask
          if ((std::popcount(explicit_b_func(f_explicit) & mask) & 1) != 0)
            unique_expected |= bit;
        }

        const explicit_b_func exists_actual =
            _dd_to_boolean_func.at(f.exists(dd_var_set));
        assert(exists_actual == exists_expected);

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
            assert(f.apply_exists(op, g, dd_var_set) ==
                   inner.exists(dd_var_set));
            assert(f.apply_unique(op, g, dd_var_set) ==
                   inner.unique(dd_var_set));
          }
        }
      }
    }
  }
};

template <boolean_function_manager M> void test_cofactors(const M &mgr) {
  constexpr bool zbdd = std::is_base_of_v<zbdd_manager, M>;

  typename M::function t0, t1;
  if constexpr (zbdd) {
    t0 = mgr.empty();
    assert(t0 == mgr.f());
    t1 = mgr.base();
  } else {
    t0 = mgr.f();
    t1 = mgr.t();
  }

  const auto [t0_t, t0_e] = t0.cofactors();
  assert(t0_t.is_invalid() && t0_e.is_invalid());
  assert(t0.cofactor_true().is_invalid());
  assert(t0.cofactor_false().is_invalid());

  const auto [t1_t, t1_e] = t1.cofactors();
  assert(t1_t.is_invalid() && t1_e.is_invalid());
  assert(t1.cofactor_true().is_invalid());
  assert(t1.cofactor_false().is_invalid());

  const var_no_t num_vars = mgr.num_vars();
  for (var_no_t i = 0; i < num_vars; ++i) {
    typename M::function v;
    if constexpr (zbdd) {
      v = mgr.singleton(i);
    } else {
      v = mgr.var(i);
    }

    const auto [v_t, v_e] = v.cofactors();
    assert(v_t == v.cofactor_true());
    assert(v_e == v.cofactor_false());
    assert(v_t == t1);
    assert(v_e == t0);
  }
}

/// Sentinel for var_name_iter
struct var_name_iter_sentinel {
  unsigned end = std::numeric_limits<unsigned>::max();
};

/// An input iterator yielding string views like `x_42`. This is a minimal
/// std::input_iterator for testing purposes.
class var_name_iter {
  friend struct var_name_iter_sentinel;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = std::string_view;

private:
  std::ostringstream _current = std::ostringstream("x_");
  unsigned _i = 0;

  void _update_string() {
    _current.seekp(2);
    _current << _i;
  }

public:
  var_name_iter() { _update_string(); };
  var_name_iter(unsigned start) : _i(start) { _update_string(); };

  std::string_view operator*() const { return _current.view(); }

  var_name_iter &operator++() {
    ++_i;
    _update_string();
    return *this;
  }
  void operator++(int) {
    ++_i;
    _update_string();
  }

  bool operator==(const var_name_iter_sentinel &s) const { return _i == s.end; }
};

static_assert(std::input_iterator<var_name_iter>);
static_assert(std::sentinel_for<var_name_iter_sentinel, var_name_iter>);
static_assert(!std::forward_iterator<var_name_iter>);

/// Test addition of vars
///
/// Assumes that two variables called "a" and "b" (in this order) are present so
/// far and that there has not been any reordering operation so far.
template <manager M> void test_add_vars(M &mgr) {
  // The unchecked-optional-access lint is a bit too strict in that it forbids
  // `.value()` as well (which throws an exception and is thus safe).
  // NOLINTBEGIN(*-magic-numbers,*-unchecked-optional-access)
  assert(mgr.num_vars() == 2);
  assert(mgr.num_named_vars() == 2);
  assert(mgr.var_name(0) == "a");
  assert(mgr.var_name(1) == "b");

  const std::ranges::iota_view added = mgr.add_vars(3);
  assert(mgr.num_vars() == 5);
  assert(mgr.num_named_vars() == 2);
  assert(*added.begin() == 2);
  assert(*added.end() == 5);
  for (unsigned i = 2; i < 5; ++i) {
    assert(mgr.var_name(i).empty());
  }

  const var_no_range_t added_vars =
      mgr.add_named_vars(var_name_iter(5), var_name_iter_sentinel{7}).value();
  assert(*added_vars.begin() == 5);
  assert(*added_vars.end() == 7);

  assert(mgr.num_vars() == 7);
  assert(mgr.num_named_vars() == 4);
  assert(mgr.var_name(5) == "x_5");
  assert(mgr.var_name(6) == "x_6");

  assert(mgr.name_to_var("a").value() == 0);
  assert(mgr.name_to_var("b").value() == 1);
  assert(mgr.name_to_var("x_5").value() == 5);
  assert(mgr.name_to_var("x_6").value() == 6);
  assert(!mgr.name_to_var("x4"));
  assert(!mgr.name_to_var(""));

  mgr.set_var_name(4, "x_4");
  assert(mgr.num_vars() == 7);
  assert(mgr.num_named_vars() == 5);
  assert(mgr.var_name(4) == "x_4");
  assert(mgr.name_to_var("x_4").value() == 4);

  // no reordering was enabled so far
  for (unsigned i = 0; i < 7; ++i) {
    assert(mgr.var_to_level(i) == i);
    assert(mgr.level_to_var(i) == i);
  }

  // check that duplicates are handled as intended
  const std::array<std::string, 3> names{"c", "b", "x"};
  const util::duplicate_var_name err = mgr.add_named_vars(names).error();
  assert(mgr.num_vars() == 8);
  assert(mgr.num_named_vars() == 6);
  assert(err.name == "b");
  assert(err.present_var == 1);
  assert(*err.added_vars.begin() == 7);
  assert(*err.added_vars.end() == 8);
  assert(mgr.name_to_var("c").value() == 7);
  assert(mgr.name_to_var("b").value() == 1);
  assert(!mgr.name_to_var("x"));
  // NOLINTEND(*-magic-numbers,*-unchecked-optional-access)
}

template <boolean_function_manager M>
void test_all_boolean_functions_2vars_t1() {
  static_assert(manager<M>);
  static_assert(std::regular<M>);
  static_assert(std::same_as<typename M::function::manager, M>);

  static_assert(boolean_function<typename M::function>);
  static_assert(std::regular<typename M::function>);
  static_assert(std::totally_ordered<typename M::function>);

  constexpr std::size_t inner_node_capacity = 65536;
  constexpr std::size_t apply_cache_capacity = 1024;
  constexpr uint32_t threads = 1;
  M mgr(inner_node_capacity, apply_cache_capacity, threads);

  assert(!mgr.is_invalid());
  assert(mgr.gc_count() == 0);

  assert(mgr.run_in_worker_pool([]() { return 42; }) == 42);
  assert(mgr.run_in_worker_pool([&mgr]() { return mgr; }) == mgr);
  assert(mgr.run_in_worker_pool([&mgr]() { return &mgr; }) == &mgr);

  assert(mgr.run_in_worker_pool([]() noexcept { return 42; }) == 42);
  assert(mgr.run_in_worker_pool([&mgr]() noexcept { return mgr; }) == mgr);
  assert(mgr.run_in_worker_pool([&mgr]() noexcept { return &mgr; }) == &mgr);

#ifdef __cpp_exceptions
  {
    constexpr int expected = 1337;
    int got = 0;
    try {
      mgr.run_in_worker_pool([]() { throw int(expected); });
    } catch (int i) {
      got = i;
    }
    assert(got == expected);
  }
#endif // __cpp_exceptions

  mgr.run_in_worker_pool([&mgr]() {
    const var_no_range_t added_vars =
        mgr.add_named_vars(std::array{"a", "b"}).value();
    assert(*added_vars.begin() == 0);
    assert(*added_vars.end() == 2);

    const test_all_boolean_functions test(mgr);
    test.basic();
    if constexpr (function_subst<typename M::function>)
      test.subst();
    if constexpr (boolean_function_quant<typename M::function>)
      test.quant();

    test_cofactors(mgr);
  });

  const std::uint64_t gc_count = mgr.gc_count();
  mgr.gc();
  assert(mgr.gc_count() == gc_count + 1);

  test_add_vars(mgr);
}

} // namespace

int main() { // NOLINT(*-exception-escape)
  static_assert(boolean_function_quant<bdd_function>);
  static_assert(boolean_function_quant<bcdd_function>);
  static_assert(function_subst<bdd_function>);
  static_assert(function_subst<bcdd_function>);

  test_all_boolean_functions_2vars_t1<bdd_manager>();
  test_all_boolean_functions_2vars_t1<bcdd_manager>();
  test_all_boolean_functions_2vars_t1<zbdd_manager>();
}
