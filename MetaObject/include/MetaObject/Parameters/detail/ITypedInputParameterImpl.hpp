#pragma once
#ifndef __CUDACC__
#include "MetaObject/Logging/Log.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include <functional>

namespace mo
{
    template<class T> 
    class ITypedInputParameter;

    template<class T> 
    ITypedInputParameter<T>::ITypedInputParameter(const std::string& name, Context* ctx):
            ITypedParameter<T>(name, Input_e, -1, ctx),
            input(nullptr),
            IParameter(name, Input_e)
    {
		update_slot = std::bind(&ITypedInputParameter<T>::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2);
		delete_slot = std::bind(&ITypedInputParameter<T>::onInputDelete, this, std::placeholders::_1);
    }
    
    template<class T> 
    ITypedInputParameter<T>::~ITypedInputParameter()
    {
        if(input)
            input->Unsubscribe();
        if(shared_input)
            shared_input->Unsubscribe();
    }
    
    template<class T> 
    bool ITypedInputParameter<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        boost::recursive_mutex::scoped_lock lock(this->mtx());
        if(param == nullptr)
        {
            if(shared_input)
            {
                shared_input->Unsubscribe();
            }else if(input)
            {
                input->Unsubscribe();
            }
            input = nullptr;
            shared_input.reset();
            this->OnUpdate(nullptr);
            return true;
        }
        auto casted_param = std::dynamic_pointer_cast<ITypedParameter<T>>(param);
        if(casted_param)
        {
            if(input) input->Unsubscribe();
            if(shared_input) shared_input->Unsubscribe();
            update_slot.Clear();
            delete_slot.Clear();
            if(casted_param->GetTimestamp() != -1)
            {
                UpdateData(casted_param->GetDataPtr(), casted_param->GetTimestamp(), casted_param->GetContext());
            }
            shared_input = casted_param;
            casted_param->RegisterUpdateNotifier(&update_slot);
			casted_param->RegisterDeleteNotifier(&delete_slot);

            this->OnUpdate(casted_param->GetContext());
            return true;
        }
        return false;
    }

    template<class T> 
    bool ITypedInputParameter<T>::SetInput(IParameter* param)
    {
        boost::recursive_mutex::scoped_lock lock(this->mtx());
        if(param == nullptr)
        {
            if(shared_input)
            {
                shared_input->Unsubscribe();
            }else if(input)
            {
                input->Unsubscribe();
            }
            update_slot.Clear();
            delete_slot.Clear();
            input = nullptr;
            shared_input.reset();
            this->OnUpdate(nullptr);
            return true;
        }
        auto casted_param = dynamic_cast<ITypedParameter<T>*>(param);
        if(casted_param)
        {
            if(input) input->Unsubscribe();
            if(shared_input) shared_input->Unsubscribe();
            if(casted_param->GetTimestamp() != -1)
            {
                UpdateData(casted_param->GetDataPtr(), casted_param->GetTimestamp(), casted_param->GetContext());
            }
            input = casted_param;
            input->Subscribe();
			casted_param->RegisterUpdateNotifier(&update_slot);
			casted_param->RegisterDeleteNotifier(&delete_slot);
            this->OnUpdate(casted_param->GetContext());
            return true;
        }
        return false;
    }

    template<class T> 
    bool ITypedInputParameter<T>::AcceptsInput(std::weak_ptr<IParameter> param) const
    {
        if(auto ptr = param.lock())
            return ptr->GetTypeInfo() == GetTypeInfo();
        return false;
    }

    template<class T> 
    bool ITypedInputParameter<T>::AcceptsInput(IParameter* param) const
    {
        return param->GetTypeInfo() == GetTypeInfo();
    }

    template<class T> 
    bool ITypedInputParameter<T>::AcceptsType(TypeInfo type) const
    {
        return type == GetTypeInfo();
    }

    template<class T> 
    IParameter* ITypedInputParameter<T>::GetInputParam()
    {
        if(shared_input)
            return shared_input.get();
        return input;
    }
    
    template<class T> 
    T* ITypedInputParameter<T>::GetDataPtr(long long ts, Context* ctx)
    {
        if(input)
            return input->GetDataPtr(ts, ctx);
        if(shared_input)
            return shared_input->GetDataPtr(ts, ctx);
        return nullptr;
    }

    template<class T> 
    bool ITypedInputParameter<T>::GetData(T& value, long long ts, Context* ctx)
    {
        if(input)
            return input->GetData(value, ts, ctx);
        if(shared_input)
            return shared_input->GetData(value, ts, ctx);
        return false;
    }
    
    template<class T> 
    T ITypedInputParameter<T>::GetData(long long ts, Context* ctx)
    {
        if(input)
            return input->GetData(ts, ctx);
        if(shared_input)
            return shared_input->GetData(ts, ctx);
        THROW(debug) << "Input not set for " << GetTreeName();
        return T();
    }

    template<class T>
    bool ITypedInputParameter<T>::GetInput(long long ts)
    {
        return true;
    }
    
    // ---- protected functions
    template<class T> 
    void ITypedInputParameter<T>::onInputDelete(IParameter const* param)
    {
        boost::recursive_mutex::scoped_lock lock(this->mtx());
        this->shared_input.reset();
        this->input = nullptr;
        this->OnUpdate(GetContext());
    }
    
    
    template<class T> 
    void ITypedInputParameter<T>::onInputUpdate(Context* ctx, IParameter* param)
    {
        this->OnUpdate(ctx);
    }

    template<class T>
    ITypedParameter<T>* ITypedInputParameter<T>::UpdateData(T& data_, long long ts, Context* ctx)
    {
        if (ts != -1)
            _timestamp = ts;
        return this;
    }

    template<class T>
    ITypedParameter<T>* ITypedInputParameter<T>::UpdateData(const T& data_, long long ts, Context* ctx)
    {
        if (ts != -1)
            _timestamp = ts;
        return this;
    }

    template<class T>
    ITypedParameter<T>* ITypedInputParameter<T>::UpdateData(T* data_, long long ts, Context* ctx)
    {
        if (ts != -1)
            _timestamp = ts;
        return this;
    }
}
#endif
