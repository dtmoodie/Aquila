#pragma once
#include <MetaObject/Detail/HelperMacros.hpp>
#include "MetaObject/Parameters/ParameterInfo.hpp"

#include "ISimpleSerializer.h"
#include "cereal/cereal.hpp"

#define PARAM_(type, name, init, N) \
LOAD_SAVE_(name, N) \
INIT_SET_(name, init, N) \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), \
                              #name, "", "", mo::Control_e, #init); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
} \
SERIALIZE_(name, N)

#define INIT_SET_(name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.SetMtx(_mtx); \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}

#define SET_(name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    init_parameters_(firstInit, --dummy); \
}

#define INIT_(name,  N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    name##_param.SetMtx(_mtx); \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
}

#define LOAD_SAVE_(name, N) \
template<class T> void _load_parameters(T& ar, mo::_counter_<N> dummy) \
{ \
  _load_parameters(ar, --dummy); \
  ar(CEREAL_NVP(name)); \
} \
template<class T> void _save_parameters(T& ar, mo::_counter_<N> dummy) const \
{ \
  _save_parameters(ar, --dummy); \
  ar(CEREAL_NVP(name)); \
}

#define ENUM_PARAM_(N, name, ...) \
template<class T> void _serialize_parameters(T& ar, mo::_counter_<N> dummy) \
{ \
    _serialize_parameters(ar, --dummy); \
    ar(CEREAL_NVP(name)); \
} \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
    { \
        name.SetValue(ENUM_EXPAND(__VA_ARGS__)); \
    }\
    name##_param.SetMtx(_mtx); \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(mo::EnumParameter)), #name); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
} \
SERIALIZE_(name, N)



#define OUTPUT_(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.SetMtx(_mtx); \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    name##_param.SetFlags(mo::ParameterType::Output_e); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), #name, "", "", mo::ParameterType::Output_e); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
} \
SERIALIZE_(name, N)


#define TOOLTIP_(NAME, TOOLTIP, N) \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    list_parameter_info_(info, --dummy); \
    for(auto it : info) \
    { \
        if(it->name == #NAME) \
        { \
            if(it->tooltip.empty()) \
            { \
                it->tooltip = TOOLTIP; \
            } \
        } \
    } \
}

#define STATUS_(type, name, init, N)\
template<class T> void _serialize_parameters(T& ar, mo::_counter_<N> dummy) \
{ \
    _serialize_parameters(ar, --dummy); \
    ar(CEREAL_NVP(name)); \
} \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    if(firstInit) \
        name = init; \
    name##_param.SetMtx(_mtx); \
    name##_param.UpdatePtr(&name); \
    name##_param.SetContext(_ctx); \
    name##_param.SetName(#name); \
    name##_param.SetFlags(mo::ParameterType::State_e); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), #name, "", "", mo::ParameterType::State_e); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
} \
SERIALIZE_(name, N)

#define SERIALIZE_(name, N) \
void _serialize_parameters(ISimpleSerializer* pSerializer, mo::_counter_<N> dummy) \
{ \
    SERIALIZE(name); \
    _serialize_parameters(pSerializer, --dummy); \
} 

#define INPUT_PARAM_(type, name, init, N) \
void init_parameters_(bool firstInit, mo::_counter_<N> dummy) \
{ \
    name##_param.SetMtx(_mtx); \
    name##_param.SetUserDataPtr(&name); \
    name##_param.SetName(#name); \
    AddParameter(&name##_param); \
    init_parameters_(firstInit, --dummy); \
} \
static void list_parameter_info_(std::vector<mo::ParameterInfo*>& info, mo::_counter_<N> dummy) \
{ \
    static mo::ParameterInfo s_info(mo::TypeInfo(typeid(type)), #name, "", "", mo::ParameterType::Input_e); \
    info.push_back(&s_info); \
    list_parameter_info_(info, --dummy); \
}
