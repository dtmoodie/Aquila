#pragma once
#include "MetaObject/Logging/Log.hpp"
namespace mo
{
	template<typename T> 
    TypedParameter<T>::TypedParameter(const std::string& name, const T& init, ParameterType type, long long ts, Context* ctx) :
		ITypedParameter<T>(name, type, ts, ctx), data(init),
        IParameter(name, mo::Control_e, ts, ctx)
	{
		(void)&_typed_parameter_constructor;
        (void)&_meta_parameter;
	}

	template<typename T> 
    T* TypedParameter<T>::GetDataPtr(long long ts, Context* ctx)
	{
		if (ts != -1)
        {
            if(ts != this->_timestamp)
            {
                LOG(debug) << "Requested timestamp " << ts << " != " << this->_timestamp;
                return nullptr;
            }
        }
		return &data;
	}

	template<typename T> 
    bool TypedParameter<T>::GetData(T& value, long long ts, Context* ctx)
	{
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
		if (ts == -1)
        {
            value = data;
            return true;
        }else
        {
            if(ts == this->_timestamp)
            {
                value = data;
                return true;
            }
        }
        return false;
	}

    template<typename T>
    T TypedParameter<T>::GetData(long long ts, Context* ctx)
    {
        boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
        if(ts == -1)
        {
            return data;
        }else
        {
            ASSERT_EQ(ts, this->_timestamp) << " Timestamps do not match";
            return data;
        }
    }
	
    template<typename T> 
    ITypedParameter<T>* TypedParameter<T>::UpdateData(T& data_, long long ts, Context* ctx)
	{
		data = data_;
        IParameter::Commit(ts, ctx);
		return this;
	}

	template<typename T> 
    ITypedParameter<T>* TypedParameter<T>::UpdateData(const T& data_, long long ts, Context* ctx)
	{
		data = data_;
        IParameter::Commit(ts, ctx);
		return this;
	}
	
    template<typename T> 
    ITypedParameter<T>* TypedParameter<T>::UpdateData(T* data_, long long ts, Context* ctx)
	{
		data = *data_;
        IParameter::Commit(ts, ctx);
		return this;
	}
	
    template<typename T> 
    std::shared_ptr<IParameter> TypedParameter<T>::DeepCopy() const
	{
        return std::shared_ptr<IParameter>(new TypedParameter<T>(IParameter::GetName(), data));
	}

	template<typename T>  
    bool TypedParameter<T>::Update(IParameter* other, Context* ctx)
	{
		auto typed = dynamic_cast<ITypedParameter<T>*>(other);
		if (typed)
		{
			if (typed->GetData(data, -1, ctx))
			{
                IParameter::Commit(other->GetTimestamp(), ctx);
				return true;
			}
		}
		return false;
	}

	template<typename T> ParameterConstructor<TypedParameter<T>> TypedParameter<T>::_typed_parameter_constructor;
    template<typename T> MetaParameter<T, 100>  TypedParameter<T>::_meta_parameter;
}
