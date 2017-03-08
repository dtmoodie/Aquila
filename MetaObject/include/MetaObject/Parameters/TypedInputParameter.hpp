/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#pragma once
#include "ITypedInputParameter.hpp"


namespace mo
{
    template<typename T> class TypedInputParameterCopy : public ITypedInputParameter<T>
    {
    public:
        TypedInputParameterCopy(const std::string& name, T* userVar_,
            ParameterType type = Control_e)
        {
            this->input = nullptr;
        }
        
        T* GetDataPtr(long long ts = -1, Context* ctx = nullptr)
        {
            return userVar;
        }
        bool GetData(T& value, long long time_step = -1, Context* ctx = nullptr)
        {
            if (userVar)
            {
                value = *userVar;
                return true;
            }
            return false;                
        }
        T GetData(long long ts = -1, Context* ctx = nullptr)
        {
            if(this->input)
                return this->input->GetData(ts, ctx);
            if(this->shared_input)
                return this->shared_input->GetData(ts, ctx);
            return false;                
        }
        void UpdateData(T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {
            if(userVar)
                *userVar = data_;
        }
        void UpdateData(const T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {
            if(userVar)
                *userVar = data_;
        }
        void UpdateData(T* data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
        {
            if(userVar )
                *userVar = *data_;
        }
    protected:
        T* userVar; // Pointer to the user space variable of type T

        void onInputUpdate(Context* ctx, IParameter* param)
        {
            if(this->input && userVar)
                this->input->GetData(*userVar, -1, this->GetContext());
            IParameter::OnUpdate(ctx);
        }
        void onInputDelete(IParameter* param)
        {
            this->input = nullptr;
            IParameter::OnUpdate(nullptr);
        }
    };

    // Meant to reference a pointer variable in user space, and to update that variable whenever 
    // IE int* myVar; 
    // auto typedParam = TypedInputParameterPtr(&myVar); // TypedInputParameter now updates myvar to point to whatever the
    // input variable is for typedParam.
    template<typename T> class TypedInputParameterPtr : public ITypedInputParameter<T>
    {
    public:
        TypedInputParameterPtr(const std::string& name = "", const T** userVar_ = nullptr, Context* ctx = nullptr);
        bool SetInput(std::shared_ptr<IParameter> input);
        bool SetInput(IParameter* input);
        void SetUserDataPtr(const T** user_var_);
        bool GetInput(long long ts = -1);
    protected:
        const T** userVar; // Pointer to the user space pointer variable of type T
        void updateUserVar();
        virtual void onInputUpdate(Context* ctx, IParameter* param);
        virtual void onInputDelete(IParameter const* param);
    };
}
#include "MetaObject/Parameters/detail/TypedInputParameterImpl.hpp"
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
