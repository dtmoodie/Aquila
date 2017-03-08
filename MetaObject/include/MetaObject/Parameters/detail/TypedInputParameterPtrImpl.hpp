#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    template<typename T> class TypedInputParameterPtr;

    template<typename T> TypedInputParameterPtr<T>::TypedInputParameterPtr(const std::string& name, const T** userVar_, Context* ctx) :
            userVar(userVar_),
            ITypedInputParameter<T>(name, ctx),
            IParameter(name, Input_e, -1, ctx),
            ITypedParameter<T>(name, Input_e, -1, ctx)
    {
    }
        
    template<typename T>
    bool TypedInputParameterPtr<T>::SetInput(std::shared_ptr<IParameter> param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ITypedInputParameter<T>::SetInput(param))
        {
            if(userVar)
            {
                if(this->input)
                    *userVar = this->input->GetDataPtr();
                if(this->shared_input)
                    *userVar = this->shared_input->GetDataPtr();
            }
            return true;
        }
        return false;
    }

    template<typename T>
    bool TypedInputParameterPtr<T>::SetInput(IParameter* param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ITypedInputParameter<T>::SetInput(param))
        {
            if(userVar)
            {
                if(this->input)
                    *userVar = this->input->GetDataPtr();
                if(this->shared_input)
                    *userVar = this->shared_input->GetDataPtr();
            }
            return true;
        }
        return false;
    }

    template<typename T>
    void TypedInputParameterPtr<T>::SetUserDataPtr(const T** user_var_)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        userVar = user_var_;
    }

    template<typename T>
    void TypedInputParameterPtr<T>::onInputUpdate(Context* ctx, IParameter* param)
    {
        if(this->input)
        {
            this->Commit(this->input->GetTimestamp(), ctx);
            if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
            {
                if(userVar)
                    *userVar = this->input->GetDataPtr();
            }
        }else if(this->shared_input)
        {
            this->Commit(this->shared_input->GetTimestamp(), ctx);
            if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
            {
                if(userVar)
                    *userVar = this->shared_input->GetDataPtr();
            }
        }
    }

    template<typename T>
    bool TypedInputParameterPtr<T>::GetInput(long long ts)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(userVar)
        {
            if(this->shared_input)
            {
                *userVar = this->shared_input->GetDataPtr(ts, this->_ctx);
                return *userVar != nullptr;
            }
            if(this->input)
            {
                *userVar = this->input->GetDataPtr(ts, this->_ctx);
                return *userVar != nullptr;
            }
        }
        return false;
    }

    template<typename T>
    void TypedInputParameterPtr<T>::onInputDelete(IParameter const* param)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        this->shared_input.reset();
        this->input = nullptr;
    }
}
#endif
