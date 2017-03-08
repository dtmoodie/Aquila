#pragma once

namespace mo
{
    namespace Buffer
    {
        template<class T> Map<T>::Map(const std::string& name) :
            ITypedInputParameter<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e)
        {
            this->SetFlags(Buffer_e);
        }

        template<class T> T* Map<T>::GetDataPtr(long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if (ts == -1 && _data_buffer.size())
            {
                return &(_data_buffer.rbegin()->second);
            }
            else
            {
                auto itr = _data_buffer.find(ts);
                if (itr != _data_buffer.end())
                {
                    return &itr->second;
                }
            }
            return nullptr;
        }
        template<class T> bool Map<T>::GetData(T& value, long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if (ts == -1 && _data_buffer.size())
            {
                value = _data_buffer.rbegin()->second;
                return true;
            }
            auto itr = _data_buffer.find(ts);
            if (itr != _data_buffer.end())
            {
                value = itr->second;
                return true;
            }
            return false;
        }
        template<class T> T Map<T>::GetData(long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if (ts == -1 && _data_buffer.size())
            {
                return _data_buffer.rbegin()->second;

            }
            auto itr = _data_buffer.find(ts);
            if (itr != _data_buffer.end())
            {
                return  itr->second;
            }
            THROW(debug) << "Desired time (" << ts << ") not found " << _data_buffer.begin()->first << ", " << _data_buffer.rbegin()->first;
            return T();
        }
        template<class T> ITypedParameter<T>* Map<T>::UpdateData(T& data_, long long ts, Context* ctx)
        {
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                _data_buffer[ts] = data_;
                IParameter::modified = true;
            }
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T> ITypedParameter<T>* Map<T>::UpdateData(const T& data_, long long ts, Context* ctx)
        {
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                _data_buffer[ts] = data_;
                IParameter::modified = true;
            }
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T> ITypedParameter<T>* Map<T>::UpdateData(T* data_, long long ts, Context* ctx)
        {
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                _data_buffer[ts] = *data_;
                IParameter::modified = true;
                this->_timestamp = ts;
            }
            IParameter::OnUpdate(ctx);
            return this;
        }

        template<class T> bool Map<T>::Update(IParameter* other, Context* ctx)
        {
            auto typedParameter = std::dynamic_pointer_cast<ITypedParameter<T>*>(other);
            if (typedParameter)
            {
                auto ptr = typedParameter->Data();
                if (ptr)
                {
                    {
                        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                        _data_buffer[typedParameter->GetTimeIndex()] = *ptr;
                        IParameter::modified = true;
                    }
                    IParameter::OnUpdate(ctx);
                }
            }
            return false;
        }
        template<class T> void Map<T>::SetSize(long long size)
        {

        }
        template<class T> long long Map<T>::GetSize() 
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            return _data_buffer.size();
        }
        template<class T> void Map<T>::GetTimestampRange(long long& start, long long& end) 
        {
            if (_data_buffer.size())
            {
                boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
                start = _data_buffer.begin()->first;
                end = _data_buffer.rbegin()->first;
            }
        }
        template<class T> std::shared_ptr<IParameter> Map<T>::DeepCopy() const
        {
            auto ptr = new Map<T>(IParameter::_name);
            ptr->_data_buffer = this->_data_buffer;
            return std::shared_ptr<IParameter>(ptr);
        }
        template<class T> void Map<T>::onInputUpdate(Context* ctx, IParameter* param)
        {
            UpdateData(this->input->GetDataPtr(), this->input->GetTimestamp(), ctx);
        }
    }
}
