#pragma once
#ifndef __CUDACC__
#include "MetaObject/MetaObjectInfo.hpp"
#include "MetaObject/MetaObjectPolicy.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include <boost/preprocessor.hpp>
#include <shared_ptr.hpp>
#include "ObjectInterfacePerModule.h"
#include <type_traits>
#include <vector>
#include <string>


struct ISimpleSerializer;
namespace mo
{
    struct SignalInfo;
    struct SlotInfo;
    struct ParameterInfo;
}

// ---------------- SIGNAL_INFO ------------
#define SIGNAL_INFO_START(N_) \
template<int N> static void list_signal_info_(std::vector<mo::SignalInfo*>& info, mo::_counter_<N> dummy) \
{ \
    return list_signal_info_(info, --dummy); \
} \
static void list_signal_info_(std::vector<mo::SignalInfo*>& info, mo::_counter_<N_> dummy) \
{ \
} 

#define SIGNAL_INFO_END(N) \
static void GetSignalInfoStatic(std::vector<mo::SignalInfo*> & info) \
{ \
    list_signal_info_(info, mo::_counter_<N-1>()); \
    _list_parent_signals(info); \
    std::sort(info.begin(), info.end()); \
    info.erase(std::unique(info.begin(), info.end()), info.end()); \
} \
static std::vector<mo::SignalInfo*> GetSignalInfoStatic() \
{ \
    std::vector<mo::SignalInfo*> info; \
    GetSignalInfoStatic(info); \
    return info; \
} \
void GetSignalInfo(std::vector<mo::SignalInfo*> & info) const \
{ \
    GetSignalInfoStatic(info); \
}

// ---------------- SIGNALS ------------
#define SIGNALS_START(N_) \
template<int N> int init_signals_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    return init_signals_(firstInit, --dummy); \
} \
int init_signals_(bool firstInit, mo::_counter_<N_> dummy) \
{ \
    return 0; \
}


#define SIGNALS_END(N_) \
virtual int InitSignals(bool firstInit) \
{ \
    int count = _init_parent_signals(firstInit); \
	return init_signals_(firstInit, mo::_counter_<N_-1>()) + count; \
}

// ---------------- SLOT INFO ------------
#define SLOT_INFO_START(N_) \
template<int N> static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N> dummy) \
{ \
    return list_slots_(info, --dummy); \
} \
static void list_slots_(std::vector<mo::SlotInfo*>& info, mo::_counter_<N_> dummy) \
{ \
}

#define SLOT_INFO_END(N) \
static void GetSlotInfoStatic(std::vector<mo::SlotInfo*> & info) \
{ \
    list_slots_(info, mo::_counter_<N-1>()); \
    _list_parent_slots(info); \
    std::sort(info.begin(), info.end()); \
    info.erase(std::unique(info.begin(), info.end()), info.end()); \
} \
static std::vector<mo::SlotInfo*> GetSlotInfoStatic() \
{ \
    std::vector<mo::SlotInfo*> info; \
    GetSlotInfoStatic(info); \
    return info; \
} \
void GetSlotInfo(std::vector<mo::SlotInfo*>& info) const \
{ \
    GetSlotInfoStatic(info); \
}


// ---------------- PARAMETERS INFO ------------

#define PARAMETER_INFO_START(N_) \
template<int N> static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_parameter_info_(info, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N_> dummy) \
{ \
} 


#define PARAMETER_INFO_END(N) \
static void GetParameterInfoStatic(std::vector<mo::ParameterInfo*>& info) \
{ \
    list_parameter_info_(info, mo::_counter_<N-1>()); \
    _list_parent_parameter_info(info); \
    std::sort(info.begin(), info.end()); \
    info.erase(std::unique(info.begin(), info.end()), info.end()); \
} \
static std::vector<mo::ParameterInfo*> GetParameterInfoStatic() \
{ \
    std::vector<mo::ParameterInfo*> info; \
    GetParameterInfoStatic(info); \
    return info; \
} \
void GetParameterInfo(std::vector<mo::ParameterInfo*>& info) const \
{ \
    GetParameterInfoStatic(info); \
}

// ---------------- PARAMETERS ------------
#define PARAMETER_START(N_) \
template<int N> void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    init_parameters_(firstInit, --dummy); \
} \
void init_parameters_(bool firstInit, mo::_counter_<N_> dummy) \
{ \
} \
template<int N> void _serialize_parameters(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy) \
{ \
    _serialize_parameters(pSerializer, --dummy); \
} \
void _serialize_parameters(ISimpleSerializer* pSerializer, mo::_counter_<N_> dummy) \
{ \
} \
template<class T, int N> void _load_parameters(T& ar, mo::_counter_<N> dummy) \
{ \
    _load_parameters<T>(ar, --dummy); \
} \
template<class T, int N> void _save_parameters(T& ar, mo::_counter_<N> dummy) const \
{ \
    _save_parameters<T>(ar, --dummy); \
} \
template<class T> void _load_parameters(T& ar, mo::_counter_<N_> dummy) \
{ \
} \
template<class T> void _save_parameters(T& ar, mo::_counter_<N_> dummy) const \
{ \
}

#define PARAMETER_END(N_) \
void InitParameters(bool firstInit) \
{ \
    init_parameters_(firstInit, mo::_counter_<N_ - 1>()); \
    _init_parent_params(firstInit); \
} \
void SerializeParameters(ISimpleSerializer* pSerializer) \
{ \
    _serialize_parameters(pSerializer, mo::_counter_<N_ - 1>()); \
    _serialize_parent_params(pSerializer); \
} \
template<class T> void load(T& ar) \
{ \
    _load_parameters<T>(ar, mo::_counter_<N_ -1>()); \
    _load_parent<T>(ar); \
} \
template<class T> void save(T& ar) const \
{ \
    _save_parameters<T>(ar, mo::_counter_<N_ -1>()); \
    _save_parent<T>(ar); \
}


// -------------- SLOTS -------------
#define SLOT_START(N_) \
template<int N> void bind_slots_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    bind_slots_(firstInit, --dummy); \
} \
void bind_slots_(bool firstInit, mo::_counter_<N_> dummy)  \
{  \
}

#define SLOT_END(N_) \
void BindSlots(bool firstInit) \
{ \
    _bind_parent_slots(firstInit); \
    bind_slots_(firstInit, mo::_counter_<N_-1>()); \
}

#define _HANDLE_PARENT_1(PARENT1) \
void _init_parent_params(bool firstInit) \
{ \
    PARENT1::InitParameters(firstInit); \
} \
void _serialize_parent_params(ISimpleSerializer* pSerializer) \
{ \
    PARENT1::SerializeParameters(pSerializer); \
} \
template<class T> void _load_parent(T& ar) \
{ \
    PARENT1::load(ar); \
} \
template<class T> void _save_parent(T& ar) const \
{ \
   PARENT1::save(ar); \
} \
void _bind_parent_slots(bool firstInit) \
{ \
    PARENT1::BindSlots(firstInit); \
} \
static void _list_parent_parameter_info(std::vector<mo::ParameterInfo*>& info) \
{ \
    PARENT1::GetParameterInfoStatic(info); \
} \
static void _list_parent_signals(std::vector<mo::SignalInfo*>& info) \
{ \
    PARENT1::GetSignalInfoStatic(info); \
} \
static void _list_parent_slots(std::vector<mo::SlotInfo*>& info) \
{ \
    PARENT1::GetSlotInfoStatic(info); \
} \
int _init_parent_signals(bool firstInit) \
{ \
    return PARENT1::InitSignals(firstInit); \
}


#define _HANDLE_PARENT_2(PARENT1, PARENT2) \
void _init_parent_params(bool firstInit) \
{ \
    PARENT1::InitParameters(firstInit); \
    PARENT2::InitParameters(firstInit); \
} \
void _serialize_parent_params(ISimpleSerializer* pSerializer) \
{ \
    PARENT1::SerializeParameters(pSerializer); \
    PARENT2::SerializeParameters(pSerializer); \
} \
template<class T> void _load_parent(T& ar) \
{ \
    PARENT1::load(ar); \
    PARENT2::load(ar); \
} \
template<class T> void _save_parent(T& ar) const \
{ \
   PARENT1::save(ar); \
   PARENT2::save(ar); \
} \
void _bind_parent_slots(bool firstInit) \
{ \
    PARENT1::BindSlots(firstInit); \
    PARENT2::BindSlots(firstInit); \
} \
static void _list_parent_parameter_info(std::vector<mo::ParameterInfo*>& info) \
{ \
    PARENT1::GetParameterInfoStatic(info); \
    PARENT2::GetParameterInfoStatic(info); \
} \
static void _list_parent_signals(std::vector<mo::SignalInfo*>& info) \
{ \
    PARENT1::GetSignalInfoStatic(info); \
    PARENT2::GetSignalInfoStatic(info); \
} \
static void _list_parent_slots(std::vector<mo::SlotInfo*>& info) \
{ \
    PARENT1::GetSlotInfoStatic(info); \
    PARENT2::GetSlotInfoStatic(info); \
} \
int _init_parent_signals(bool firstInit) \
{ \
    return PARENT1::InitSignals(firstInit) + PARENT2::InitSignals(firstInit);; \
} \


#ifdef _MSC_VER
#define _HANDLE_PARENT(...)  BOOST_PP_CAT(BOOST_PP_OVERLOAD(_HANDLE_PARENT_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY())
#else
#define _HANDLE_PARENT(...)  BOOST_PP_OVERLOAD(_HANDLE_PARENT_, __VA_ARGS__)(__VA_ARGS__)
#endif


#define _HANDLE_NO_PARENT \
void _init_parent_params(bool firstInit){ } \
void _serialize_parent_params(ISimpleSerializer* pSerializer) { } \
template<class T> void _load_parent(T& ar) { } \
template<class T> void _save_parent(T& ar) const { } \
void _bind_parent_slots(bool firstInit) { } \
static void _list_parent_parameter_info(std::vector<mo::ParameterInfo*>& info) { } \
static void _list_parent_signals(std::vector<mo::SignalInfo*>& info) { } \
static void _list_parent_slots(std::vector<mo::SlotInfo*>& info) { } \
int _init_parent_signals(bool firstInit) {     return 0; }


#define MO_BEGIN_1(CLASS_NAME, N_) \
typedef CLASS_NAME THIS_CLASS;      \
_HANDLE_NO_PARENT; \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_) \
PARAMETER_START(N_) \
static rcc::shared_ptr<CLASS_NAME> Create();

#define MO_DERIVE_(N_, CLASS_NAME, ...) \
typedef CLASS_NAME THIS_CLASS; \
_HANDLE_PARENT(__VA_ARGS__); \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_)\
PARAMETER_START(N_) \
static rcc::shared_ptr<CLASS_NAME> Create();

#define MO_END_(N) \
SIGNAL_INFO_END(N) \
SLOT_INFO_END(N) \
PARAMETER_INFO_END(N) \
SIGNALS_END(N) \
SLOT_END(N) \
PARAMETER_END(N)

#define MO_ABSTRACT_(N_, CLASS_NAME, ...) \
typedef CLASS_NAME THIS_CLASS; \
_HANDLE_PARENT(__VA_ARGS__); \
SIGNAL_INFO_START(N_) \
SIGNALS_START(N_) \
SLOT_INFO_START(N_) \
PARAMETER_INFO_START(N_) \
SLOT_START(N_)\
PARAMETER_START(N_)

#define MO_REGISTER_OBJECT(TYPE) \
    static ::mo::MetaObjectInfo<TActual<TYPE>> TYPE##_info; \
    static ::mo::MetaObjectPolicy<TActual<TYPE>, __COUNTER__, void> TYPE##_policy; \
    ::rcc::shared_ptr<TYPE> TYPE::Create() \
    { \
        auto obj = ::mo::MetaObjectFactory::Instance()->Create(#TYPE); \
        return ::rcc::shared_ptr<TYPE>(obj); \
    } \
    REGISTERCLASS(TYPE, &TYPE##_info);

#define MO_REGISTER_CLASS(TYPE) MO_REGISTER_OBJECT(TYPE)

#else
#define MO_REGISTER_OBJECT(TYPE)
#define MO_REGISTER_CLASS(TYPE)
#define MO_BEGIN_1(CLASS, N)
#define MO_BEGIN_2(CLASS, PARENT, N)
#define MO_END_(N)
#endif
