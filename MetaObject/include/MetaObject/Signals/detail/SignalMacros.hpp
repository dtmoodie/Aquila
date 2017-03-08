#pragma once
#include <boost/preprocessor/facilities/overload.hpp>
#ifdef _MSC_VER
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#endif

#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include <vector>

#define SIGNAL_CALL_1(N, name, ret) \
inline ret sig_##name() \
{ \
	return COMBINE(_sig_##name##_, N)(); \
}

#define SIGNAL_CALL_2(N, name, ret, ARG1) \
inline ret sig_##name(ARG1& arg1) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1); \
}

#define SIGNAL_CALL_3(N, name, ret, ARG1, ARG2) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2); \
}


#define SIGNAL_CALL_4(N, name, ret, ARG1, ARG2, ARG3) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2, ARG3& arg3) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2, arg3); \
}

#define SIGNAL_CALL_5(N, name, ret, ARG1, ARG2, ARG3, ARG4) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2, ARG3& arg3, ARG4& arg4) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2, arg3, arg4); \
}

#define SIGNAL_CALL_6(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2, ARG3& arg3, ARG4& arg4, ARG5& arg5) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2, arg3, arg4, arg5); \
}

#define SIGNAL_CALL_7(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2, ARG3& arg3, ARG4& arg4, ARG5& arg5, ARG6& arg6) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2, arg3, arg4, arg5, arg6); \
}

#define SIGNAL_CALL_8(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2, ARG3& arg3, ARG4& arg4, ARG5& arg5, ARG6& arg6, ARG7& arg7) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7); \
}

#define SIGNAL_CALL_9(N, name, ret, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8) \
inline ret sig_##name(ARG1& arg1, ARG2& arg2, ARG3& arg3, ARG4& arg4, ARG5& arg5, ARG6& arg6, ARG7& arg7, ARG8& arg8) \
{ \
	return COMBINE(_sig_##name##_,N)(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); \
}

#define INIT_SIGNALS_(N, C, RETURN, NAME, ...) \
int init_signals_(bool firstInit, mo::_counter_<C> dummy) \
{ \
    AddSignal(&COMBINE(_sig_##NAME##_, N), #NAME); \
    return init_signals_(firstInit, --dummy) + 1; \
} \
template<class Sig> mo::TypedSignal<RETURN(__VA_ARGS__)>* GetSignal_##NAME(typename std::enable_if<std::is_same<Sig, RETURN(__VA_ARGS__)>::value>::type* = 0) \
{ \
    return &COMBINE(_sig_##NAME##_, N); \
} \
static void list_signal_info_(std::vector<mo::SignalInfo*>& output, mo::_counter_<C> dummy) \
{ \
    static mo::SignalInfo info{mo::TypeInfo(typeid(RETURN(__VA_ARGS__))), std::string(#NAME)}; \
    list_signal_info_(output, --dummy); \
    output.push_back(&info); \
}


#ifdef BOOST_PP_VARIADICS_MSVC
  #define SIGNAL_CALL(N, name, ...) BOOST_PP_CAT(BOOST_PP_OVERLOAD(SIGNAL_CALL_, __VA_ARGS__)(N, name, __VA_ARGS__), BOOST_PP_EMPTY())
#else
  #define SIGNAL_CALL(N, name, ...) BOOST_PP_OVERLOAD(SIGNAL_CALL_, __VA_ARGS__)(N, name, __VA_ARGS__)
#endif

#define MO_SIGNAL_(N, RETURN, NAME, ...) \
mo::TypedSignal<RETURN(__VA_ARGS__)> COMBINE(_sig_##NAME##_,N); \
SIGNAL_CALL(N, NAME, RETURN, ##__VA_ARGS__) \
INIT_SIGNALS_(N, __COUNTER__, RETURN, NAME, __VA_ARGS__)



#define DESCRIBE_SIGNAL_(NAME, DESCRIPTION, N) \
std::vector<slot_info> list_signals_(mo::_counter_<N> dummy) \
{ \
    auto signal_info = list_signals_(mo::_counter_<N-1>()); \
    for(auto& info : signal_info) \
    { \
        if(info.name == #NAME) \
        { \
            info.description = DESCRIPTION; \
        } \
    } \
    return signal_info; \
}

#ifndef __CUDACC__
  #define DESCRIBE_SIGNAL(NAME, DESCRIPTION) DESCRIBE_SIGNAL_(NAME, DESCRIPTION, __COUNTER__);
  #define MO_SIGNAL(RETURN, NAME, ...) MO_SIGNAL_(__LINE__, RETURN, NAME, ##__VA_ARGS__)
#else
  #define DESCRIBE_SIGNAL(NAME, DESCRIPTION)
  #define MO_SIGNAL(RETURN, NAME, ...)
#endif
