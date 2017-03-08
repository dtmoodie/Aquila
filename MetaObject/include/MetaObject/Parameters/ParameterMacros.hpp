#pragma once
#ifndef __CUDACC__
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "detail/ParameterMacrosImpl.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"

#define PARAM(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
PARAM_(type, name, init, __COUNTER__)


#define ENUM_PARAM(name, ...) \
mo::TypedParameterPtr<mo::EnumParameter> name##_param; \
mo::EnumParameter name; \
ENUM_PARAM_(__COUNTER__, name, __VA_ARGS__)


#define RANGED_PARAM(type, name, init, min, max)

#define INPUT(type, name, init) \
const type* name = init; \
mo::TypedInputParameterPtr<type> name##_param; \
void init_parameters_(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    name##_param.SetMtx(_mtx); \
    name##_param.SetUserDataPtr(&name); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<__COUNTER__> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), \
                              #name, "", "", mo::Input_e, #init); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
}

#define OPTIONAL_INPUT(type, name, init) \
INPUT(type, name, init); \
APPEND_FLAGS(name, mo::Optional_e);

#define APPEND_FLAGS(name, flags) \
void init_parameters_(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    name##_param.AppendFlags(flags); \
    init_parameters_(firstInit, --dummy); \
}


#define PROPERTY(type, name, init) \
type name; \
void init_parameters_(bool firstInit, mo::_counter_<__COUNTER__> dummy) \
{ \
    if(firstInit) \
        name = init; \
    init_parameters_(firstInit, --dummy); \
} \
mo::TypedParameterPtr<type> name##_param; \
SERIALIZE_(name, __COUNTER__)

#define PERSISTENT_(type, name, N) \
mo::TypedParameterPtr<type> name##_param; \
INIT_(name, N) \
LOAD_SAVE_(name, N)

#define PERSISTENT(type, name) \
type name; \
PERSISTENT_(type, name, __COUNTER__)

#define INIT(name, init) INIT_(name, init, __COUNTER__)

#define STATUS(type, name, init)\
mo::TypedParameterPtr<type> name##_param; \
type name = init; \
STATUS_(type, name, init, __COUNTER__)

#define TOOLTIP(name, TOOLTIP) TOOLTIP_(name, TOOLTIP, __COUNTER__)

#define DESCRIPTION(name, DESCRIPTION)

#define OUTPUT(type, name, init) \
mo::TypedParameterPtr<type> name##_param; \
OUTPUT_(type, name, init, __COUNTER__); \
type name = init;

#else
#define PARAM(type, name, init)
#define PROPERTY(type, name, init)
#define INPUT(type, name, init)
#define OUTPUT(type, name, init)
#endif
