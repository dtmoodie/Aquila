#pragma once
#ifndef __CUDACC__
#include "MetaObject/Logging/Log.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
	template<typename T> class TypedParameterPtr;
    template<typename T, int N, typename Enable> struct MetaParameter;
    

	template<typename T> TypedParameterPtr<T>::TypedParameterPtr(const std::string& name, T* ptr_, ParameterType type, bool ownsData_) :
		ptr(ptr_), ownsData(ownsData_), ITypedParameter<T>(name, type)
	{
        (void)&_meta_parameter;
	}
	
    template<typename T> 
    TypedParameterPtr<T>::~TypedParameterPtr()
	{
		if (ownsData && ptr)
			delete ptr;
	}

    template<typename T> 
    T* TypedParameterPtr<T>::GetDataPtr(long long ts, Context* ctx)
	{
        if(ts != -1)
        {
            if(ts != this->_timestamp && this->_timestamp != -1)
            {
                LOG(trace) << "Requested timestamp != current [" << ts << " != " << this->_timestamp << "] for parameter " << this->GetTreeName();
                return nullptr;
            }else if(this->_timestamp == -1)
            {
                return ptr;
            }
        }
		return ptr;
	}
	
    template<typename T> 
    T TypedParameterPtr<T>::GetData(long long ts, Context* ctx)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ts != -1)
        {
            if(ts != this->_timestamp)
            {
                THROW(debug) << "Requested timestamp != current [" << ts << " != " << this->_timestamp << "] for parameter " << this->GetTreeName();
            }
        }
        if(ptr == nullptr)
            THROW(debug) << "Data pointer not set";
		return *ptr;
	}
	
    template<typename T> 
    bool TypedParameterPtr<T>::GetData(T& value, long long ts, Context* ctx)
	{
		boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ts != -1)
        {
            if(ts != this->_timestamp)
            {
                LOG(trace) << "Requested timestamp != current [" << ts << " != " << this->_timestamp << "] for parameter " << this->GetTreeName();
                return false;
            }
        }
		
		if (ptr)
		{
			value = *ptr;
			return true;
		}
		return false;
	}
	
    template<typename T> 
    ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(T& data_, long long time_index, Context* ctx)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ptr)
        {
            *ptr = data_;
            IParameter::_timestamp = time_index;
            IParameter::modified = true;
            IParameter::OnUpdate(ctx);
        }
        return this;
	}
	
    template<typename T> 
    ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(const T& data_, long long time_index, Context* ctx)
	{
        bool updated = false;
        {
            boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
            if (ptr)
            {
                *ptr = data_;
                IParameter::_timestamp = time_index;
                IParameter::modified = true;
                updated = true;
            }
        }
        if(updated)
        {
            IParameter::OnUpdate(ctx);
            return this;
        }
        return nullptr;
	}
	
    template<typename T> 
    ITypedParameter<T>* TypedParameterPtr<T>::UpdateData(T* data_, long long time_index, Context* ctx)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ptr)
        {
            *ptr = *data_;
            IParameter::_timestamp = time_index;
            IParameter::modified = true;
            IParameter::OnUpdate(ctx);
        }
        return this;
	}
	
    template<typename T> 
    bool TypedParameterPtr<T>::Update(IParameter* other)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
		auto typed = dynamic_cast<ITypedParameter<T>*>(other);
		if (typed)
		{
			*ptr = typed->GetData();
			IParameter::_timestamp = other->GetTimestamp();
			IParameter::modified = true;
			IParameter::OnUpdate(nullptr);
			return true;
		}
		return false;
	}
	
    template<typename T> 
    std::shared_ptr<IParameter> TypedParameterPtr<T>::DeepCopy() const
	{
		return std::shared_ptr<IParameter>(new TypedParameterPtr<T>(IParameter::GetName(), ptr));
	}

    template<typename T>
    ITypedParameter<T>* TypedParameterPtr<T>::UpdatePtr(T* ptr)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        this->ptr = ptr;
        return this;
    }
    
    template<typename T> MetaParameter<T, 100, void> TypedParameterPtr<T>::_meta_parameter;
}
#endif
