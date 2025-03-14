/// @file  bdd.hpp
/// @brief Binary decision diagrams (without complement edges)

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <version>

#ifdef __cpp_lib_concepts
#include <concepts>
#endif // __cpp_lib_concepts

#include <oxidd/capi.h>
#include <oxidd/util.hpp>

namespace oxidd {

class bdd_function;
class bdd_substitution;

/// Manager for binary decision diagrams (without complement edges)
///
/// Instances can safely be sent to other threads.
///
/// Models `oxidd::concepts::boolean_function_manager`
class bdd_manager {
  /// Wrapped CAPI BDD manager
  capi::oxidd_bdd_manager_t _manager = {._p = nullptr};

  friend class bdd_function;

  /// Create a new BDD manager from a manager instance of the CAPI
  bdd_manager(capi::oxidd_bdd_manager_t manager) noexcept : _manager(manager) {}

public:
  /// Associated function type
  using function = bdd_function;

  /// Default constructor, yields an invalid manager
  bdd_manager() noexcept = default;
  /// Create a new BDD manager
  ///
  /// @param  inner_node_capacity   Maximum count of inner nodes
  /// @param  apply_cache_capacity  Maximum count of apply cache entries
  /// @param  threads               Thread count for the internal thread pool
  bdd_manager(size_t inner_node_capacity, size_t apply_cache_capacity,
              uint32_t threads) noexcept
      : _manager(capi::oxidd_bdd_manager_new(inner_node_capacity,
                                             apply_cache_capacity, threads)) {}
  /// Copy constructor: increments the internal atomic reference counter
  ///
  /// Runtime complexity: O(1)
  bdd_manager(const bdd_manager &other) noexcept : _manager(other._manager) {
    capi::oxidd_bdd_manager_ref(_manager);
  }
  /// Move constructor: invalidates `other`
  bdd_manager(bdd_manager &&other) noexcept : _manager(other._manager) {
    other._manager._p = nullptr;
  }

  ~bdd_manager() noexcept {
    if (_manager._p != nullptr)
      capi::oxidd_bdd_manager_unref(_manager);
  }

  /// Copy assignment operator
  bdd_manager &operator=(const bdd_manager &rhs) noexcept {
    if (this != &rhs) {
      capi::oxidd_bdd_manager_unref(_manager);
      _manager = rhs._manager;
      capi::oxidd_bdd_manager_ref(_manager);
    }
    return *this;
  }
  /// Move assignment operator: invalidates `rhs`
  bdd_manager &operator=(bdd_manager &&rhs) noexcept {
    assert(this != &rhs || !rhs._manager._p);
    capi::oxidd_bdd_manager_unref(_manager);
    _manager = rhs._manager;
    rhs._manager._p = nullptr;
    return *this;
  }

  /// Compare two managers for referential equality
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same manager
  friend bool operator==(const bdd_manager &lhs,
                         const bdd_manager &rhs) noexcept {
    return lhs._manager._p == rhs._manager._p;
  }
  /// Same as `!(lhs == rhs)` (see `operator==()`)
  friend bool operator!=(const bdd_manager &lhs,
                         const bdd_manager &rhs) noexcept {
    return !(lhs == rhs);
  }

  /// Check if this manager reference is invalid
  ///
  /// A manager reference created by the default constructor `bdd_manager()` is
  /// invalid as well as a `bdd_manager` instance that has been moved (via
  /// `bdd_manager(bdd_manager &&other)`).
  ///
  /// @returns  `true` iff this manager reference is invalid
  [[nodiscard]] bool is_invalid() const noexcept {
    return _manager._p == nullptr;
  }

  /// Execute `f()` in the worker thread pool of `manager`
  ///
  /// Recursive calls in the multithreaded apply algorithms are always executed
  /// within the manager's thread pool, requiring a rather expensive context
  /// switch if the apply algorithm is not called from within the thread pool.
  /// If the algorithm takes long to execute anyway, this may not be important,
  /// but for many small operations, this may easily make a difference by
  /// factors.
  ///
  /// This method blocks until `f()` has finished.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// @returns  The result of calling `f()`
  template <typename R> R run_in_worker_pool(std::function<R()> f) const {
    assert(_manager._p);
    return oxidd::util::detail::run_in_worker_pool(
        capi::oxidd_bdd_manager_run_in_worker_pool, _manager, std::move(f));
  }

  /// @name BDD Construction
  /// @{

  /// Get a fresh variable, i.e., a function that is true if and only if the
  /// variable is true. This adds a new level to the decision diagram.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for exclusive access.
  ///
  /// @returns  The BDD function representing the variable
  [[nodiscard]] bdd_function new_var() noexcept;

  /// Get the constant true BCDD function ⊤
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊤ as BDD function
  [[nodiscard]] bdd_function t() const noexcept;
  /// Get the constant false BCDD function ⊥
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  ⊥ as BDD function
  [[nodiscard]] bdd_function f() const noexcept;

  /// @}
  /// @name Statistics
  /// @{

  /// Get the number of inner nodes currently stored
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The number of inner nodes
  [[nodiscard]] size_t num_inner_nodes() const noexcept {
    assert(_manager._p != nullptr);
    return capi::oxidd_bdd_num_inner_nodes(_manager);
  }

  /// @}
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
class bdd_function {
  /// Wrapped BDD function
  capi::oxidd_bdd_t _func = {._p = nullptr};

  friend class bdd_manager;
  friend class bdd_substitution;
  friend struct std::hash<bdd_function>;

  /// Create a new `bdd_function` from a BDD function instance of the CAPI
  bdd_function(capi::oxidd_bdd_t func) noexcept : _func(func) {}

public:
  /// Associated manager type
  using manager = bdd_manager;
  /// Associated substitution type
  using substitution = bdd_substitution;

  /// Default constructor, yields an invalid BCDD function
  bdd_function() noexcept = default;
  /// Copy constructor: increments the internal reference counters
  ///
  /// Runtime complexity: O(1)
  bdd_function(const bdd_function &other) noexcept : _func(other._func) {
    capi::oxidd_bdd_ref(_func);
  }
  /// Move constructor: invalidates `other`
  bdd_function(bdd_function &&other) noexcept : _func(other._func) {
    other._func._p = nullptr;
  }

  ~bdd_function() noexcept {
    if (_func._p != nullptr)
      capi::oxidd_bdd_unref(_func);
  }

  /// Copy assignment operator
  bdd_function &operator=(const bdd_function &rhs) noexcept {
    if (this != &rhs) {
      capi::oxidd_bdd_unref(_func);
      _func = rhs._func;
      capi::oxidd_bdd_ref(_func);
    }
    return *this;
  }
  /// Move assignment operator: invalidates `rhs`
  bdd_function &operator=(bdd_function &&rhs) noexcept {
    assert(this != &rhs || !rhs._func._p);
    capi::oxidd_bdd_unref(_func);
    _func = rhs._func;
    rhs._func._p = nullptr;
    return *this;
  }

  /// Compare two functions for referential equality
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` and `rhs` reference the same node
  friend bool operator==(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return lhs._func._i == rhs._func._i && lhs._func._p == rhs._func._p;
  }
  /// Same as `!(lhs == rhs)` (see `operator==()`)
  friend bool operator!=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(lhs == rhs);
  }
  /// Check if `lhs` is less than `rhs` according to an arbitrary total order
  ///
  /// Runtime complexity: O(1)
  ///
  /// @param  lhs  Left hand side operand
  /// @param  rhs  Right hand side operand
  ///
  /// @returns  `true` iff `lhs` is less than `rhs` according to the total order
  friend bool operator<(const bdd_function &lhs,
                        const bdd_function &rhs) noexcept {
    return std::tie(lhs._func._p, lhs._func._i) <
           std::tie(rhs._func._p, rhs._func._i);
  }
  /// `operator<()` with arguments swapped
  friend bool operator>(const bdd_function &lhs,
                        const bdd_function &rhs) noexcept {
    return rhs < lhs;
  }
  /// Same as `!(rhs < lhs)` (see `operator<()`)
  friend bool operator<=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(rhs < lhs);
  }
  /// Same as `!(lhs < rhs)` (see `operator<()`)
  friend bool operator>=(const bdd_function &lhs,
                         const bdd_function &rhs) noexcept {
    return !(lhs < rhs);
  }

  /// Check if this BDD function is invalid
  ///
  /// A BDD function created by the default constructor `bdd_function()` is
  /// invalid as well as a `bdd_function` instance that has been moved (via
  /// `bdd_function(bdd_function &&other)`). Moreover, if an operation tries to
  /// allocate new nodes but runs out of memory, then it returns an invalid
  /// function.
  ///
  /// @returns  `true` iff this BCDD function is invalid
  [[nodiscard]] bool is_invalid() const noexcept { return _func._p == nullptr; }

  /// Get the containing manager
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// @returns  The bdd_manager
  [[nodiscard]] bdd_manager containing_manager() const noexcept {
    assert(!is_invalid());
    return capi::oxidd_bdd_containing_manager(_func);
  }

  /// @name BDD Construction Operations
  /// @{

  /// Get the cofactors `(f_true, f_false)` of `f`
  ///
  /// Let f(x₀, …, xₙ) be represented by `f`, where x₀ is (currently) the
  /// top-most variable. Then f<sub>true</sub>(x₁, …, xₙ) = f(⊤, x₁, …, xₙ) and
  /// f<sub>false</sub>(x₁, …, xₙ) = f(⊥, x₁, …, xₙ).
  ///
  /// Structurally, the cofactors are the children. If you only need one of the
  /// cofactors, then use `cofactor_true()` or `cofactor_false()`. These
  /// functions are slightly more efficient then.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The pair `{f_true, f_false}` if `f` is valid and references an
  ///           inner node, otherwise a pair of invalid functions.
  [[nodiscard]] std::pair<bdd_function, bdd_function>
  cofactors() const noexcept {
    const capi::oxidd_bdd_pair_t p = capi::oxidd_bdd_cofactors(_func);
    return {p.first, p.second};
  }
  /// Get the cofactor `f_true` of `f`
  ///
  /// This function is slightly more efficient than `cofactors()` in case
  /// `f_false` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_true` if `f` is valid and references an inner node, otherwise
  ///           an invalid function.
  [[nodiscard]] bdd_function cofactor_true() const noexcept {
    return capi::oxidd_bdd_cofactor_true(_func);
  }
  /// Get the cofactor `f_false` of `f`
  ///
  /// This function is slightly more efficient than `cofactors()` in case
  /// `f_true` is not needed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  `f_false` if `f` is valid and references an inner node,
  ///           otherwise an invalid function.
  [[nodiscard]] bdd_function cofactor_false() const noexcept {
    return capi::oxidd_bdd_cofactor_false(_func);
  }

  /// Get the level of the underlying node
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(1)
  ///
  /// @returns  The level of the underlying inner node or
  ///           `std::numeric_limits<oxidd::level_no>::max()` for terminals and
  ///           invalid functions.
  [[nodiscard]] level_no_t level() const noexcept {
    return capi::oxidd_bdd_level(_func);
  }

  /// Compute the BDD for the negation `¬this`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function operator~() const noexcept {
    return capi::oxidd_bdd_not(_func);
  }
  /// Compute the BDD for the conjunction `lhs ∧ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] friend bdd_function
  operator&(const bdd_function &lhs, const bdd_function &rhs) noexcept {
    return capi::oxidd_bdd_and(lhs._func, rhs._func);
  }
  /// Assignment version of `operator&()`
  bdd_function &operator&=(const bdd_function &rhs) noexcept {
    return (*this = *this & rhs);
  }
  /// Compute the BDD for the disjunction `lhs ∨ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  friend bdd_function operator|(const bdd_function &lhs,
                                const bdd_function &rhs) noexcept {
    return capi::oxidd_bdd_or(lhs._func, rhs._func);
  }
  /// Assignment version of `operator|()`
  bdd_function &operator|=(const bdd_function &rhs) noexcept {
    return (*this = *this | rhs);
  }
  /// Compute the BDD for the exclusive disjunction `lhs ⊕ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|lhs| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  friend bdd_function operator^(const bdd_function &lhs,
                                const bdd_function &rhs) noexcept {
    return capi::oxidd_bdd_xor(lhs._func, rhs._func);
  }
  /// Assignment version of `operator^()`
  bdd_function &operator^=(const bdd_function &rhs) noexcept {
    return (*this = *this ^ rhs);
  }
  /// Compute the BDD for the negated conjunction `this ⊼ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function nand(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_nand(_func, rhs._func);
  }
  /// Compute the BDD for the negated disjunction `this ⊽ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function nor(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_nor(_func, rhs._func);
  }
  /// Compute the BDD for the equivalence `this ↔ rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function equiv(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_equiv(_func, rhs._func);
  }
  /// Compute the BDD for the implication `this → rhs` (or `this ≤ rhs`)
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function imp(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_imp(_func, rhs._func);
  }
  /// Compute the BDD for the strict implication `this < rhs`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |rhs|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function
  imp_strict(const bdd_function &rhs) const noexcept {
    return capi::oxidd_bdd_imp_strict(_func, rhs._func);
  }
  /// Compute the BDD for the conditional `this ? t : e`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// Runtime complexity: O(|this| · |t| · |e|)
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function ite(const bdd_function &t,
                                 const bdd_function &e) const noexcept {
    return capi::oxidd_bdd_ite(_func, t._func, e._func);
  }

  /// Substitute `vars` in the BDD `f` by `replacement`
  ///
  /// The substitution is performed in a parallel fashion, e.g.:
  /// `(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function
  substitute(const bdd_substitution &substitution) const noexcept;

  /// Compute the BDD for the universal quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// universal quantification. Universal quantification `∀x. f(…, x, …)` of a
  /// Boolean function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ∧ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function forall(const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_forall(_func, vars._func);
  }
  /// Compute the BDD for the existential quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// existential quantification. Existential quantification `∃x. f(…, x, …)` of
  /// a Boolean function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ∨ f(…, 1, …)`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function exists(const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_exists(_func, vars._func);
  }
  /// Deprecated alias for `exists()`
  [[nodiscard, deprecated]]
  bdd_function exist(const bdd_function &vars) const noexcept {
    return exists(vars);
  }
  /// Compute the BDD for the unique quantification over `vars`
  ///
  /// `vars` is a set of variables, which in turn is just the conjunction of the
  /// variables. This operation removes all occurrences of the variables by
  /// unique quantification. Unique quantification `∃!x. f(…, x, …)` of a
  /// Boolean function `f(…, x, …)` over a single variable `x` is
  /// `f(…, 0, …) ⊕ f(…, 1, …)`.
  ///
  /// Unique quantification is also known as the
  /// [Boolean
  /// difference](https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors)
  /// or
  /// [Boolean
  /// derivative](https://en.wikipedia.org/wiki/Boolean_differential_calculus).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function (may be invalid if the operation runs out of
  ///           memory)
  [[nodiscard]] bdd_function unique(const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_unique(_func, vars._func);
  }

  /// Combined application of `op` and `forall()`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function `∀ vars. this <op> rhs` (may be invalid if the
  ///           operation runs out of memory)
  [[nodiscard]] bdd_function
  apply_forall(const util::boolean_operator op, const bdd_function &rhs,
               const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_apply_forall(
        static_cast<capi::oxidd_boolean_operator>(op), _func, rhs._func,
        vars._func);
  }

  /// Combined application of `op` and `exists()`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function `∃ vars. this <op> rhs` (may be invalid if the
  ///           operation runs out of memory)
  [[nodiscard]] bdd_function
  apply_exists(const util::boolean_operator op, const bdd_function &rhs,
               const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_apply_exists(
        static_cast<capi::oxidd_boolean_operator>(op), _func, rhs._func,
        vars._func);
  }
  /// Deprecated alias for `apply_exists()`
  [[nodiscard, deprecated]]
  bdd_function apply_exist(const util::boolean_operator op,
                           const bdd_function &rhs,
                           const bdd_function &vars) const noexcept {
    return apply_exists(op, rhs, vars);
  }

  /// Combined application of `op` and `unique()`
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  The BDD function `∃! vars. this <op> rhs` (may be invalid if the
  ///           operation runs out of memory)
  [[nodiscard]] bdd_function
  apply_unique(const util::boolean_operator op, const bdd_function &rhs,
               const bdd_function &vars) const noexcept {
    return capi::oxidd_bdd_apply_unique(
        static_cast<capi::oxidd_boolean_operator>(op), _func, rhs._func,
        vars._func);
  }

  /// @}
  /// @name Query Operations on BDDs
  /// @{

  /// Count descendant nodes
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  Node count including the two terminal nodes
  [[nodiscard]] std::size_t node_count() const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_node_count(_func);
  }

  /// Check for satisfiability
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is a satisfying assignment
  [[nodiscard]] bool satisfiable() const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_satisfiable(_func);
  }

  /// Check for validity
  ///
  /// `this` must not be invalid (in the technical, not the mathematical sense).
  /// Check via `is_invalid()`.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  `true` iff there is are only satisfying assignment
  [[nodiscard]] bool valid() const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_valid(_func);
  }

  /// Count the number of satisfying assignments
  ///
  /// This method assumes that the function's domain of has `vars` many
  /// variables.
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  vars  Number of variables in the function's domain
  ///
  /// @returns  Count of satisfying assignments
  [[nodiscard]] double sat_count_double(level_no_t vars) const noexcept {
    assert(_func._p);
    return capi::oxidd_bdd_sat_count_double(_func, vars);
  }

  /// Pick a satisfying assignment
  ///
  /// `this` must not be invalid (check via `is_invalid()`).
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. If this function is
  ///           unsatisfiable, the assignment is empty.
  [[nodiscard]] util::assignment pick_cube() const noexcept {
    assert(_func._p);
    return util::assignment(capi::oxidd_bdd_pick_cube(_func));
  }

  /// Pick a satisfying assignment, represented as BDD
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
  ///           `f` is ⊥), ⊥ is returned.
  [[nodiscard]] bdd_function pick_cube_dd() const noexcept {
    return capi::oxidd_bdd_pick_cube_dd(_func);
  }

  /// Pick a satisfying assignment, represented as BDD, using the literals in
  /// `literal_set` if there is a choice
  ///
  /// `literal_set` is represented as a conjunction of literals. Whenever there
  /// is a choice for a variable, it will be set to true if the variable has a
  /// positive occurrence in `literal_set`, and set to false if it occurs
  /// negated in `literal_set`. If the variable does not occur in `literal_set`,
  /// then it will be left as don't care if possible, otherwise an arbitrary
  /// (not necessarily random) choice will be performed.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @returns  A satisfying assignment if there exists one. Otherwise (i.e., if
  ///           `f` is ⊥), ⊥ is returned.
  [[nodiscard]] bdd_function
  pick_cube_dd_set(const bdd_function &literal_set) const noexcept {
    return capi::oxidd_bdd_pick_cube_dd_set(_func, literal_set._func);
  }

  /// Evaluate this Boolean function with arguments `args`
  ///
  /// `args` determines the valuation for all variables. Missing values are
  /// assumed to be false. The order is irrelevant. All elements must point to
  /// inner nodes.
  ///
  /// Locking behavior: acquires the manager's lock for shared access.
  ///
  /// @param  args  Slice of pairs `(variable, value)`, where all variables are
  ///               not invalid
  ///
  /// @returns  Result of the evaluation with `args`
  [[nodiscard]] bool
  eval(util::slice<std::pair<bdd_function, bool>> args) const noexcept {
    assert(_func._p);

    // From a C++ perspective, it is nicer to have elements of type
    // `std::pair<bdd_function, bool>` than `capi::oxidd_bdd_bool_pair_t`. In
    // the following we ensure that their layouts are compatible such that the
    // pointer cast below is safe.
    using c_pair = capi::oxidd_bdd_bool_pair_t;
    using cpp_pair = std::pair<bdd_function, bool>;
    static_assert(std::is_standard_layout_v<cpp_pair>);
    static_assert(sizeof(cpp_pair) == sizeof(c_pair));
    static_assert(offsetof(cpp_pair, first) == offsetof(c_pair, func));
    static_assert(offsetof(cpp_pair, second) == offsetof(c_pair, val));
    static_assert(alignof(cpp_pair) == alignof(c_pair));

    return capi::oxidd_bdd_eval(
        _func,
        reinterpret_cast<const c_pair *>(args.data()), // NOLINT(*-cast)
        args.size());
  }

  /// @}
};

inline bdd_function bdd_manager::new_var() noexcept {
  assert(_manager._p);
  return capi::oxidd_bdd_new_var(_manager);
}
inline bdd_function bdd_manager::t() const noexcept {
  assert(_manager._p);
  return capi::oxidd_bdd_true(_manager);
}
inline bdd_function bdd_manager::f() const noexcept {
  assert(_manager._p);
  return capi::oxidd_bdd_false(_manager);
}

/// Substitution mapping variables to replacement functions
///
/// Models `oxidd::concepts::substitution`
class bdd_substitution {
  /// Wrapped substitution
  capi::oxidd_bdd_substitution_t *_subst = nullptr;

  friend class bdd_function;

public:
  /// Associated function type
  using function = bdd_function;

  /// Create an invalid substitution
  bdd_substitution() = default;
  bdd_substitution(const bdd_substitution &other) = delete;
  /// Move constructor: invalidates `other`
  bdd_substitution(bdd_substitution &&other) noexcept : _subst(other._subst) {
    other._subst = nullptr;
  }

  bdd_substitution &operator=(const bdd_substitution &) = delete;
  /// Move assignment operator: invalidates `rhs`
  bdd_substitution &operator=(bdd_substitution &&rhs) noexcept {
    assert(this != &rhs || !rhs._subst);
    capi::oxidd_bdd_substitution_free(_subst);
    _subst = rhs._subst;
    rhs._subst = nullptr;
    return *this;
  }

  ~bdd_substitution() noexcept { capi::oxidd_bdd_substitution_free(_subst); }

  /// Create a BDD substitution from an iterator of pairs
  /// `(const bdd_function &var, const bdd_function &replacement)`
#ifdef __cpp_lib_concepts
  template <std::input_iterator IT>
    requires(std::equality_comparable<IT> &&
             util::pair_like<std::iter_value_t<IT>, const bdd_function &,
                             const bdd_function &>)
#else  // __cpp_lib_concepts
  template <typename IT>
#endif // __cpp_lib_concepts
  bdd_substitution(IT begin, IT end)
      : _subst(capi::oxidd_bdd_substitution_new(util::size_hint(
            begin, end,
            typename std::iterator_traits<IT>::iterator_category()))) {
    for (; begin != end; ++begin) {
      const auto &pair = *begin;
      const bdd_function &var = std::get<0>(pair);
      const bdd_function &replacement = std::get<1>(pair);
      assert(var._func._p);
      assert(replacement._func._p);
      capi::oxidd_bdd_substitution_add_pair(_subst, var._func,
                                            replacement._func);
    }
  }

#if defined(__cpp_lib_ranges) && defined(__cpp_lib_concepts)
  /// Create a BDD substitution from a range of pairs
  /// `(const bdd_function &var, const bdd_function &replacement)`
  template <std::ranges::input_range R>
    requires(util::pair_like<std::ranges::range_value_t<R>,
                             const bdd_function &, const bdd_function &>)
  bdd_substitution(R &&range)
      : _subst(capi::oxidd_bdd_substitution_new(util::size_hint(range))) {
    for (const auto &[var, replacement] : std::forward<R>(range)) {
      assert(var._func._p);
      assert(replacement._func._p);
      capi::oxidd_bdd_substitution_add_pair(_subst, var._func,
                                            replacement._func);
    }
  }
#endif // defined(__cpp_lib_ranges) && defined(__cpp_lib_concepts)

  /// Check if this substitution is invalid
  [[nodiscard]] bool is_invalid() const { return _subst == nullptr; }
};

inline bdd_function
bdd_function::substitute(const bdd_substitution &substitution) const noexcept {
  assert(substitution._subst);
  return capi::oxidd_bdd_substitute(_func, substitution._subst);
}

} // namespace oxidd

/// @cond

/// Partial specialization for `oxidd::bdd_function`
template <> struct std::hash<oxidd::bdd_function> {
  std::size_t operator()(const oxidd::bdd_function &f) const noexcept {
    return std::hash<const void *>{}(f._func._p) ^ f._func._i;
  }
};

/// @endcond
