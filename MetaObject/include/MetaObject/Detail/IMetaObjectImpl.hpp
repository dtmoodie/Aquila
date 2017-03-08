#pragma once
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Parameters/TypedParameter.hpp"
namespace mo
{
    class IMetaObject;

    template<class T> 
    ITypedParameter<T>* IMetaObject::GetParameter(const std::string& name) const
    {
        IParameter* param = GetParameter(name);
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if(typed)
        {
            return typed;
        }
        THROW(debug) << "Parameter \"" << name << "\" not convertable to type " << TypeInfo(typeid(T)).name();
        return nullptr;
    }

    template<class T> 
    T IMetaObject::GetParameterValue(const std::string& name, long long ts, Context* ctx) const
    {
        return GetParameter<T>(name)->GetData(ts, ctx);
    }
    template<class T> 
    ITypedParameter<T>* IMetaObject::GetParameterOptional(const std::string& name) const
    {
        auto param = GetParameterOptional(name);
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        return typed;
    }

    template<class T> 
    ITypedParameter<T>* IMetaObject::UpdateParameter(const std::string& name, T& value, long long ts, Context* ctx)
    {
        if(ctx == nullptr)
            ctx = _ctx;
        auto param = GetParameterOptional<T>(name);
        if(param)
        {
            param->UpdateData(value, ts, ctx);
            return param;
        }else
        {
            std::shared_ptr<ITypedParameter<T>> new_param(new TypedParameter<T>(name, value));
            AddParameter(new_param);
            return new_param.get();
        }
    }
    template<class T>
    ITypedParameter<T>* IMetaObject::UpdateParameter(const std::string& name, const T& value, long long ts, Context* ctx)
    {
        if (ctx == nullptr)
            ctx = _ctx;
        auto param = GetParameterOptional<T>(name);
        if (param)
        {
            param->UpdateData(value, ts, ctx);
            return param;
        }
        else
        {
            std::shared_ptr<ITypedParameter<T>> new_param(new TypedParameter<T>(name, value));
            AddParameter(new_param);
            return new_param.get();
        }
    }
    template<class T> 
    ITypedParameter<T>* IMetaObject::UpdateParameterPtr(const std::string& name, T& ptr)
    {
        return nullptr;
    }

    /*template<class Sig>
    bool IMetaObject::ConnectCallback(const std::string& callback_name, const std::string& slot_name, IMetaObject* slot_owner, bool force_queue)
    {
        ConnectCallback(TypedInfo(typeid(Sig)), callback_name, slot_name, slot_owner, force_queue);
    }*/
    template<class T> 
    TypedSlot<T>* IMetaObject::GetSlot(const std::string& name) const
    {
        return dynamic_cast<TypedSlot<T>*>(this->GetSlot(name, TypeInfo(typeid(T))));
    }
    template<class T> 
    std::vector<IParameter*> IMetaObject::GetOutputs(const std::string& name_filter) const
    {
        return GetOutputs(TypeInfo(typeid(T)), name_filter);
    }

    template<class T> 
    std::vector<InputParameter*> IMetaObject::GetInputs(const std::string& name_filter) const
    {
        return GetInputs(TypeInfo(typeid(T)), name_filter);
    }

    template<class T> 
    ITypedInputParameter<T>* IMetaObject::GetInput(const std::string& name)
    {
        auto ptr = GetInput(name);
        if(ptr)
        {
            return dynamic_cast<ITypedInputParameter<T>*>(ptr);
        }
        return nullptr;
    }
    template<class T> 
    ITypedParameter<T>* IMetaObject::GetOutput(const std::string& name) const
    {
        auto ptr = GetOutput(name);
        if(ptr)
        {
            if(ptr->GetTypeInfo() == TypeInfo(typeid(T)))
            {
                return dynamic_cast<ITypedParameter<T>*>(ptr);
            }
        }
        return nullptr;
    }
    template<class T> 
    bool IMetaObject::Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name)
    {
        return Connect(sender, signal_name, receiver, slot_name, TypeInfo(typeid(T)));
    }
}
