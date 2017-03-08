#pragma once
#include "MetaObject/Detail/HelperMacros.hpp"
#include "MetaObject/Parameters/Demangle.hpp"
namespace mo
{
    template<class T, int N, typename Enable = void> struct MetaParameter: public MetaParameter<T, N-1, void>
    {
        MetaParameter(const char* name = nullptr):
            MetaParameter<T, N-1>(name){}
    };
    template<class T> struct MetaParameter<T, 0, void>
    {
        MetaParameter(const char* name = nullptr)
        {
            if(name)
                mo::Demangle::RegisterName(mo::TypeInfo(typeid(T)), name);
        }
    };
}

#define INSTANTIATE_META_PARAMETER(TYPE) \
static mo::MetaParameter<TYPE, __COUNTER__> COMBINE(g_meta_parameter, __LINE__)(#TYPE);
