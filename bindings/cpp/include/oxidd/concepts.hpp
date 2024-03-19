/// @file  concepts.hpp
/// @brief C++20 concepts

#pragma once
#if __cplusplus >= 202002L

#include <concepts>
#include <functional>

#include <oxidd/utils.hpp>

namespace oxidd {

/// Manager storing nodes and ensuring their uniqueness
///
/// A manager is the data structure responsible for storing nodes and ensuring
/// their uniqueness. It also defines the variable order.
///
/// Implementations of this concept act as if they were `shared_ptr`s, i.e.,
/// copying them just increments an atomic reference count,
/// moving out of an instance invalidates it.
template <class T>
concept manager = std::regular<T> && requires(const T m) {
  { m.is_invalid() } -> std::same_as<bool>;

  { m.num_inner_nodes() } -> std::same_as<size_t>;
};

/// Function represented as decision diagram
///
/// Functions are hashable and totally ordered according to an arbitrary order.
template <class T>
concept function =
    std::regular<T> && std::totally_ordered<T> && requires(const T f) {
      { f.is_invalid() } -> std::same_as<bool>;

      { std::hash<T>{}(f) } -> std::convertible_to<std::size_t>;

      { f.node_count() } -> std::same_as<std::size_t>;
    };

/// Boolean function 𝔹ⁿ → 𝔹 represented as decision diagram
///
/// @see oxidd::function
template <class T>
concept boolean_function =
    function<T> && requires(const T bf, T mut_bf, level_no_t levels) {
      // negation
      { ~bf } -> std::same_as<T>;
      // conjunction
      { bf &bf } -> std::same_as<T>;
      { mut_bf &= bf } -> std::same_as<T &>;
      // disjunction
      { bf | bf } -> std::same_as<T>;
      { mut_bf |= bf } -> std::same_as<T &>;
      // exclusive disjunction
      { bf ^ bf } -> std::same_as<T>;
      { mut_bf ^= bf } -> std::same_as<T &>;
      // if-then-else
      { bf.ite(bf, bf) } -> std::same_as<T>;

      { bf.sat_count_double(levels) } -> std::same_as<double>;

      { bf.pick_cube() } -> std::same_as<assignment>;
    };

/// Manager for Boolean functions
///
/// @see oxidd::manager, oxidd::boolean_function
template <class T>
concept boolean_function_manager = manager<T> && requires(const T m, T mut_m) {
  { mut_m.new_var() } -> boolean_function;
  { m.top() } -> boolean_function;
  { m.bot() } -> boolean_function;
};

} // namespace oxidd

#endif // __cplusplus >= 202002L
