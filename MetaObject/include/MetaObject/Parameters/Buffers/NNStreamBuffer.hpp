#pragma once

#include "StreamBuffer.hpp"

namespace mo
{
    namespace Buffer
    {
        template<class T> class MO_EXPORTS NNStreamBuffer: public StreamBuffer<T>
        {
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = NNStreamBuffer_e;

            NNStreamBuffer(const std::string& name = "");

            T*   GetDataPtr(long long ts = -1, Context* ctx = nullptr);
            bool GetData(T& value, long long ts = -1, Context* ctx = nullptr);
            T    GetData(long long ts = -1, Context* ctx = nullptr);
            virtual ParameterTypeFlags GetBufferType() const{ return NNStreamBuffer_e;}
        protected:
        };
    }

#define MO_METAPARAMETER_INSTANCE_NNBUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
    { \
        static BufferConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_constructor;  \
        static ParameterConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_parameter_constructor; \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_nn_stream_buffer_constructor; \
            (void)&_nn_stream_buffer_parameter_constructor; \
        } \
    }; \
    template<class T> BufferConstructor<Buffer::NNStreamBuffer<T>> MetaParameter<T, N>::_nn_stream_buffer_constructor; \
    template<class T> ParameterConstructor<Buffer::NNStreamBuffer<T>> MetaParameter<T, N>::_nn_stream_buffer_parameter_constructor;

    MO_METAPARAMETER_INSTANCE_NNBUFFER_(__COUNTER__)
}
#include "detail/NNStreamBufferImpl.hpp"
