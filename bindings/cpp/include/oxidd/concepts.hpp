/// @file C++20 concepts

#pragma once
#if __cplusplus >= 202002L

#include <concepts>
#include <functional>

#include <oxidd/utils.hpp>

namespace oxidd {

template <class T>
concept manager = std::regular<T> && requires(const T m) {
  { m.is_invalid() } -> std::same_as<bool>;
};

template <class T>
concept function = std::regular<T> && requires(const T f) {
  { f.is_invalid() } -> std::same_as<bool>;

  { std::hash<T>{}(f) } -> std::convertible_to<std::size_t>;
};

template <class T>
concept boolean_function =
    function<T> && requires(const T bf, T mut_bf, level_no_t levels) {
      { bf.node_count() } -> std::same_as<std::uint64_t>;

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

      { bf.sat_count_double(levels) } -> std::same_as<double>;

      { bf.pick_cube() } -> std::same_as<assignment>;
    };

template <class T>
concept boolean_function_manager = manager<T> && requires(const T m, T mut_m) {
  { mut_m.new_var() } -> boolean_function;
  { m.top() } -> boolean_function;
  { m.bot() } -> boolean_function;
};

} // namespace oxidd

#endif // __cplusplus >= 202002L
