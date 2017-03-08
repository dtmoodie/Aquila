#pragma once
#include "MetaObject/Parameters/ITypedParameter.hpp"
namespace mo
{
    template<typename T> class ITypedParameter;
    namespace Buffer
    {
        class IBuffer;
        template<typename T> class Proxy : public ITypedParameter<T>, public IBuffer
        {
            ITypedParameter<T>* _input_parameter;
            std::shared_ptr<ITypedParameter<T>> _buffer;

            std::shared_ptr<Signals::connection> _input_update_connection;
            std::shared_ptr<Signals::connection> _input_delete_connection;
        public:
            Proxy(ITypedParameter<T>* input, Parameters::Parameter::Ptr buffer) :
                _input_parameter(input),
                _buffer(std::dynamic_pointer_cast<ITypedParameter>(buffer)),
                ITypedParameter<T>("proxy for " + (input ? input->GetName() : std::string("null")))
            {
                if (input)
                {
                    _input_update_connection = _input_parameter->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_connection = _input_parameter->RegisterDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                    _input_parameter->subscribers++;
                }
            }
            Proxy(ITypedParameter<T>* input, ITypedParameter<T>* buffer) :
                _input_parameter(input),
                _buffer(buffer),
                ITypedParameter<T>("proxy")

            {
                if (input)
                {
                    _input_update_connection = _input_parameter->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_connection = _input_parameter->RegisterDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                    _input_parameter->subscribers++;
                }
            }
            ~Proxy()
            {
                if (_input_parameter)
                    _input_parameter->subscribers--;
                _input_parameter = nullptr;
                _input_update_connection.reset();
                _input_delete_connection.reset();
            }
            void setInput(Parameter* param)
            {
                if (_input_parameter = dynamic_cast<ITypedParameter<T>*>(param))
                {
                    _input_update_connection = _input_parameter->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_connection = _input_parameter->RegisterDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                }
            }
            void onInputDelete()
            {
                _input_update_connection.reset();
                _input_delete_connection.reset();
                _input_parameter = nullptr;
            }
            void onInputUpdate(cv::cuda::Stream* stream)
            {
                if (_input_parameter)
                {
                    auto time = _input_parameter->GetTimeIndex();
                    if (auto data = _input_parameter->Data(time))
                    {
                        _buffer->UpdateData(data, time, stream);
                    }
                }
            }
            virtual T* Data(long long timestamp)
            {
                return _buffer->Data(timestamp);
            }
            virtual bool GetData(T& value, long long time_index = -1)
            {
                return _buffer->GetData(value, time_index);
            }
            virtual void UpdateData(T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
            {
            }
            virtual void UpdateData(const T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
            {    
            }
            virtual void UpdateData(T* data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
            {
            }

            virtual Parameter::Ptr DeepCopy() const
            {
                return Parameter::Ptr(new Proxy(_input_parameter, _buffer->DeepCopy()));
            }
            virtual void SetSize(long long size)
            {
                dynamic_cast<IBuffer*>(_buffer.get())->SetSize(size);
            }
            virtual long long GetSize()
            {
                return dynamic_cast<IBuffer*>(_buffer.get())->GetSize();
            }
            virtual void GetTimestampRange(long long& start, long long& end)
            {
                dynamic_cast<IBuffer*>(_buffer.get())->GetTimestampRange(start, end);
            }
            virtual std::recursive_mutex& mtx()
            {
                return _buffer->mtx();
            }
        };
    }
}