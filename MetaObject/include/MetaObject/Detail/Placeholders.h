#pragma once
#include <type_traits>
#include <functional>
#include <cstddef>
// *************************************************************************
// http://stackoverflow.com/questions/21192659/variadic-templates-and-stdbind
template<int...> struct int_sequence {};
template<int N, int... Is> struct make_int_sequence : make_int_sequence<N - 1, N - 1, Is...> {};
template<int... Is> struct make_int_sequence<0, Is...> : int_sequence<Is...> {};
template<int> struct placeholder_template{};
namespace std {
    template<int N> struct is_placeholder< placeholder_template<N> >: integral_constant<int, N + 1> { };
}
template<class R, class... Args, int... Is>
std::function<R(Args...)> my_bind(R(*p)(Args...), int_sequence<Is...>)
{    return std::bind(p, placeholder_template<Is>{}...);}

template<class R, class C, class... Args, int... Is>
std::function<R(Args...)> my_bind(R(C::*p)(Args...), C* ptr, int_sequence<Is...>)
{    return std::bind(p, ptr, placeholder_template<Is>{}...);}
// *************************************************************************