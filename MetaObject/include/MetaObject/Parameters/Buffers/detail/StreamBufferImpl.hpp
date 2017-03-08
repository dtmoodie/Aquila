#pragma once
#include <MetaObject/Logging/Log.hpp>
#include <boost/chrono.hpp>
namespace mo
{
    namespace Buffer
    {
        template<class T> StreamBuffer<T>::StreamBuffer(const std::string& name):
            ITypedParameter<T>(name, Buffer_e),
            _current_timestamp(-1), _padding(5)
        {
        
        }

        template<class T> T*   StreamBuffer<T>::GetDataPtr(long long ts, Context* ctx)
        {
            T* result = Map<T>::GetDataPtr(ts, ctx);
            if(result)
            {
                if(ts == -1)
                {
                    boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
                    _current_timestamp = this->_data_buffer.rbegin()->first;
                }else
                {
                    _current_timestamp = ts;
                }
                prune();
            }
            return result;
        }
        template<class T> bool StreamBuffer<T>::GetData(T& value, long long ts, Context* ctx)
        {
            if(Map<T>::GetData(value, ts, ctx))
            {
                _current_timestamp = ts;
                prune();
                return true;
            }
            return false;
        }
        template<class T> T StreamBuffer<T>::GetData(long long ts, Context* ctx)
        {
            T result = Map<T>::GetData(ts, ctx);
            _current_timestamp = ts;
            prune();
            return result;
        }
        template<class T> void StreamBuffer<T>::SetSize(long long size)
        {
            _padding = size;
        }
        template<class T> void StreamBuffer<T>::prune()
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if(_current_timestamp != -1)
            {
                auto itr = this->_data_buffer.begin();
                while(itr != this->_data_buffer.end())
                {
                    if(itr->first < _current_timestamp - _padding)
                    {
                        itr = this->_data_buffer.erase(itr);
                    }else
                    {
                        break;
                    }
                }
            }
        }
        template<class T> std::shared_ptr<IParameter> StreamBuffer<T>::DeepCopy() const
        {
            return std::shared_ptr<IParameter>(new StreamBuffer<T>());
        }

        // ------------------------------------------------------------
        template<class T> BlockingStreamBuffer<T>::BlockingStreamBuffer(const std::string& name) :
            StreamBuffer<T>(name),
            ITypedParameter<T>(name, mo::Buffer_e),
            _size(100)
        {

        }
        template<class T> void BlockingStreamBuffer<T>::SetSize(long long size)
        {
            _size = size;
        }
        
        template<class T>
        ITypedParameter<T>* BlockingStreamBuffer<T>::UpdateData(T& data_, long long ts, Context* ctx)
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            while (this->_data_buffer.size() >= _size)
            {
                LOG(trace) << "Pushing to " << this->GetTreeName() << " waiting on read";
                _cv.wait(lock);
            }
            this->_data_buffer[ts] = data_;
            IParameter::modified = true;
            this->_timestamp = ts;
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T>
        ITypedParameter<T>* BlockingStreamBuffer<T>::UpdateData(const T& data_, long long ts, Context* ctx)
        {
            boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
            while (this->_data_buffer.size() >= _size)
            {
                LOG(trace) << "Pushing to " << this->GetTreeName() << " waiting on read";
                _cv.wait(lock);
            }
            this->_data_buffer[ts] = data_;
            IParameter::modified = true;
            this->_timestamp = ts;
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T>
        ITypedParameter<T>* BlockingStreamBuffer<T>::UpdateData(T* data_, long long ts, Context* ctx)
        {
            {
                boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
                while(this->_data_buffer.size() >= _size)
                {
                    LOG(trace) << "Pushing to " << this->GetTreeName() << " waiting on read";
                    _cv.wait_for(lock, boost::chrono::microseconds(100));
                    lock.unlock();
                    IParameter::OnUpdate(ctx);
                    lock.lock();
                }
                this->_data_buffer[ts] = *data_;
                IParameter::modified = true;
                this->_timestamp = ts;
            }
            IParameter::OnUpdate(ctx);
            return this;
        }
        template<class T>
        void BlockingStreamBuffer<T>::prune()
        {
            boost::unique_lock<boost::recursive_mutex> lock(IParameter::mtx());
            if (this->_current_timestamp != -1)
            {
                auto itr = this->_data_buffer.begin();
                while (itr != this->_data_buffer.end())
                {
                    if (itr->first < this->_current_timestamp - this->_padding)
                    {
                        itr = this->_data_buffer.erase(itr);
                    }
                    else
                    {
                        break;
                    }
                }
            }else
            {
                // start dropping data from the beginning
                int size = this->_data_buffer.size();
                auto itr = this->_data_buffer.begin();
                while(size >= this->_size && itr != this->_data_buffer.end())
                {
                    itr = this->_data_buffer.erase(itr);
                    --size;
                }
            }
            lock.unlock();
            _cv.notify_all();
        }
    }
}
