#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "map.hpp"
#include "IBuffer.hpp"
#include <boost/thread/condition_variable.hpp>
namespace mo
{
    namespace Buffer
    {
        template<class T> class MO_EXPORTS StreamBuffer: public Map<T>
        {
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = StreamBuffer_e;

            StreamBuffer(const std::string& name = "");

            T*   GetDataPtr(long long ts = -1, Context* ctx = nullptr);
            bool GetData(T& value, long long ts = -1, Context* ctx = nullptr);
            T    GetData(long long ts = -1, Context* ctx = nullptr);
            void SetSize(long long size);
            std::shared_ptr<IParameter> DeepCopy() const;
            virtual ParameterTypeFlags GetBufferType() const{ return StreamBuffer_e;}
        protected:
            virtual void prune();
            long long _current_timestamp;
            long long _padding;
        };
        template<class T> class MO_EXPORTS BlockingStreamBuffer : public StreamBuffer<T>
        {
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = BlockingStreamBuffer_e;

            BlockingStreamBuffer(const std::string& name = "");

            void SetSize(long long size);

            ITypedParameter<T>* UpdateData(T& data_, long long ts = -1, Context* ctx = nullptr);
            ITypedParameter<T>* UpdateData(const T& data_, long long ts = -1, Context* ctx = nullptr);
            ITypedParameter<T>* UpdateData(T* data_, long long ts = -1, Context* ctx = nullptr);
            virtual ParameterTypeFlags GetBufferType() const{ return BlockingStreamBuffer_e;}
        protected:
            virtual void prune();
            long long _size;
            boost::condition_variable_any _cv;
        };
    }

#define MO_METAPARAMETER_INSTANCE_SBUFFER_(N) \
    template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
    { \
        static BufferConstructor<Buffer::StreamBuffer<T>> _stream_buffer_constructor;  \
        static BufferConstructor<Buffer::BlockingStreamBuffer<T>> _blocking_stream_buffer_constructor;  \
        static ParameterConstructor<Buffer::StreamBuffer<T>> _stream_buffer_parameter_constructor; \
        static ParameterConstructor<Buffer::BlockingStreamBuffer<T>> _blocking_stream_buffer_parameter_constructor; \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_stream_buffer_constructor; \
            (void)&_stream_buffer_parameter_constructor; \
            (void)&_blocking_stream_buffer_parameter_constructor; \
            (void)&_blocking_stream_buffer_constructor; \
        } \
    }; \
    template<class T> BufferConstructor<Buffer::StreamBuffer<T>> MetaParameter<T, N>::_stream_buffer_constructor; \
    template<class T> ParameterConstructor<Buffer::StreamBuffer<T>> MetaParameter<T, N>::_stream_buffer_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::BlockingStreamBuffer<T>> MetaParameter<T, N>::_blocking_stream_buffer_constructor; \
    template<class T> ParameterConstructor<Buffer::BlockingStreamBuffer<T>> MetaParameter<T, N>::_blocking_stream_buffer_parameter_constructor;
    MO_METAPARAMETER_INSTANCE_SBUFFER_(__COUNTER__)
}
#include "detail/StreamBufferImpl.hpp"
