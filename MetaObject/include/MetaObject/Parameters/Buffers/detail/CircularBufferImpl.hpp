#pragma once

namespace mo
{
    namespace Buffer
    {
        template<class T> CircularBuffer<T>::CircularBuffer(const std::string& name, const T& init, long long ts, ParameterType type ):
            ITypedInputParameter<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e)
        {
            (void)&_circular_buffer_constructor;
            (void)&_circular_buffer_parameter_constructor;
            _data_buffer.set_capacity(10);
            _data_buffer.push_back(std::make_pair(ts, init));
        }


        template<class T> T* CircularBuffer<T>::GetDataPtr(long long ts, Context* ctx)
        {
            if (ts == -1 && _data_buffer.size())
                return &_data_buffer.back().second;

            for (auto& itr : _data_buffer)
            {
                if (itr.first == ts)
                {
                    return &itr.second;
                }
            }
            return nullptr;
        }
         
        template<class T> bool CircularBuffer<T>::GetData(T& value, long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if (ts == -1 && _data_buffer.size())
            {
                value = _data_buffer.back().second;
                return true;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.first == ts)
                {
                    value = itr.second;
                    return true;
                }
            }
            return false;
        }

        template<class T> T CircularBuffer<T>::GetData(long long ts, Context* ctx)
        {
            if (ts == -1 && _data_buffer.size())
            {
                return _data_buffer.back().second;
            }
            for (auto& itr : _data_buffer)
            {
                if (itr.first == ts)
                {
                    return itr.second;
                }
            }
            THROW(debug) << "Could not find timestamp " << ts << " in range (" << _data_buffer.back().first << "," << _data_buffer.front().first <<")";
            return T();
        }
            

        template<class T> ITypedParameter<T>* CircularBuffer<T>::UpdateData(T& data_, long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            _data_buffer.push_back(std::pair<long long, T>(ts, data_));
            IParameter::modified = true;
            IParameter::OnUpdate(ctx);
            return this;
        }
            
        template<class T> ITypedParameter<T>* CircularBuffer<T>::UpdateData(const T& data_, long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            _data_buffer.push_back(std::pair<long long, T>(ts, data_));
            IParameter::modified = true;
            IParameter::OnUpdate(ctx);
            return this;
        }

        template<class T> ITypedParameter<T>* CircularBuffer<T>::UpdateData(T* data_, long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            _data_buffer.push_back(std::pair<long long, T>(ts, *data_));
            IParameter::modified = true;
            IParameter::OnUpdate(ctx);
            return this;
        }
        
        template<class T> bool CircularBuffer<T>::Update(IParameter* other, Context* ctx)
        {
            auto typedParameter = dynamic_cast<ITypedParameter<T>*>(other);
            if (typedParameter)
            {
                T data;
                if (typedParameter->GetData(data))
                {
                    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                    _data_buffer.push_back(std::pair<long long, T>(typedParameter->GetTimestamp(), data));
                    IParameter::modified = true;
                    IParameter::OnUpdate(ctx);
                }
            }
            return false;
        }

        template<class T> void  CircularBuffer<T>::SetSize(long long size)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            _data_buffer.set_capacity(size);
        }
        template<class T> long long  CircularBuffer<T>::GetSize() 
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            return _data_buffer.capacity();
        }
        template<class T> void  CircularBuffer<T>::GetTimestampRange(long long& start, long long& end) 
        {
            if (_data_buffer.size())
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                start = _data_buffer.back().first;
                end = _data_buffer.front().first;
            }
        }
        template<class T> std::shared_ptr<IParameter>  CircularBuffer<T>::DeepCopy() const
        {
            auto buffer = new CircularBuffer<T>(IParameter::_name);
            buffer->SetInput(this->input);
            return std::shared_ptr<IParameter>(buffer);
        }

        template<class T> void  CircularBuffer<T>::onInputUpdate(Context* ctx, IParameter* param)
        {
            if(this->input)
            {
                UpdateData(this->input->GetDataPtr(), this->input->GetTimestamp(), ctx);
            }
        }
        template<typename T> ParameterConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_parameter_constructor;
        template<typename T> BufferConstructor<CircularBuffer<T>> CircularBuffer<T>::_circular_buffer_constructor;
    }
}
