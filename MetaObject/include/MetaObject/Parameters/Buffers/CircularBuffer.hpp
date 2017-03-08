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

#include "MetaObject/Parameters/ITypedParameter.hpp"
#include "MetaObject/Parameters/ParameterConstructor.hpp"
#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "IBuffer.hpp"
#include "BufferConstructor.hpp"
#include <boost/circular_buffer.hpp>

namespace mo
{
    namespace Buffer
    {
        template<typename T> class CircularBuffer: public IBuffer, public ITypedInputParameter<T>
        {
            static ParameterConstructor<CircularBuffer<T>> _circular_buffer_parameter_constructor;
            static BufferConstructor<CircularBuffer<T>> _circular_buffer_constructor;
            boost::circular_buffer<std::pair<long long, T>> _data_buffer;
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = cbuffer_e;

            CircularBuffer(const std::string& name = "",
                const T& init = T(), long long ts = -1,
                ParameterType type = Buffer_e);

            T*   GetDataPtr(long long ts = -1, Context* ctx = nullptr);
            bool GetData(T& value, long long ts = -1, Context* ctx = nullptr);
            T    GetData(long long ts = -1, Context* ctx = nullptr);

            ITypedParameter<T>* UpdateData(T& data_, long long ts = -1, Context* ctx = nullptr);
            ITypedParameter<T>* UpdateData(const T& data_, long long ts = -1, Context* ctx= nullptr);
            ITypedParameter<T>* UpdateData(T* data_, long long ts = -1, Context* ctx = nullptr);
    
            bool Update(IParameter* other, Context* ctx = nullptr);
            std::shared_ptr<IParameter> DeepCopy() const;

            virtual void SetSize(long long size);
            virtual long long GetSize();
            virtual void GetTimestampRange(long long& start, long long& end);
            
            void onInputUpdate(Context* ctx, IParameter* param);
            virtual ParameterTypeFlags GetBufferType() const{ return cbuffer_e;}
        };
    }
    
    #define MO_METAPARAMETER_INSTANCE_CBUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
    { \
        static ParameterConstructor<Buffer::CircularBuffer<T>> _circular_buffer_parameter_constructor; \
        static BufferConstructor<Buffer::CircularBuffer<T>> _circular_buffer_constructor;  \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_circular_buffer_constructor; \
            (void)&_circular_buffer_parameter_constructor; \
        } \
    }; \
    template<class T> ParameterConstructor<Buffer::CircularBuffer<T>> MetaParameter<T, N>::_circular_buffer_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::CircularBuffer<T>> MetaParameter<T, N>::_circular_buffer_constructor;
    
    MO_METAPARAMETER_INSTANCE_CBUFFER_(__COUNTER__)
}
#include "detail/CircularBufferImpl.hpp"
