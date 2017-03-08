#pragma once
#include <MetaObject/Context.hpp>
#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Signals/Connection.hpp"
namespace mo
{
	template<class Sig> class TypedSignalRelay;

	template<class...T> 
	void TypedSignalRelay<void(T...)>::operator()(TypedSignal<void(T...)>* sig, T&... args)
	{
        std::lock_guard<std::mutex> lock(mtx);
		for (auto slot : _slots)
		{
			auto slot_ctx = slot->GetContext();
			auto sig_ctx = sig->GetContext();
			if (slot_ctx && sig_ctx)
			{
				if (slot_ctx->process_id == sig_ctx->process_id)
				{
					if (slot_ctx->thread_id != sig_ctx->thread_id)
					{
                        ThreadSpecificQueue::Push(
                            std::bind([slot](T... args)
                        {
                            (*slot)(args...);
                        }, args...), slot_ctx->thread_id, slot);
                        continue;
					}
				}
			}
            if(slot)
			    (*slot)(args...);
		}
	}
    template<class...T>
    void TypedSignalRelay<void(T...)>::operator()(T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto slot : _slots)
        {
            (*slot)(args...);
        }
    }
    template<class...T> 
    void TypedSignalRelay<void(T...)>::operator()(Context* ctx, T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto slot : _slots)
        {
            auto slot_ctx = slot->GetContext();
            if(slot_ctx)
            {
                if(slot_ctx->process_id == ctx->process_id && slot_ctx->thread_id != ctx->thread_id)
                {
                    ThreadSpecificQueue::Push(
                        std::bind([slot](T... args)
                        {
                            (*slot)(args...);
                        }, args...), slot_ctx->thread_id, slot);
                    continue;
                }
            }
            (*slot)(args...);
        }
    }
	
	template<class...T> 
	bool TypedSignalRelay<void(T...)>::Connect(ISignal* signal)
	{
		auto typed = dynamic_cast<TypedSignal<void(T...)>*>(signal);
		if (typed)
		{
			return Connect(typed);
		}
		return false;
	}

	template<class...T>
	bool TypedSignalRelay<void(T...)>::Connect(TypedSignal<void(T...)>* signal)
	{		
		return true;
	}

	template<class...T>
	bool TypedSignalRelay<void(T...)>::Connect(ISlot* slot)
	{
		auto typed = dynamic_cast<TypedSlot<void(T...)>*>(slot);
		if (typed)
		{
			return Connect(typed);
		}
		return false;
	}

	template<class...T>
	bool TypedSignalRelay<void(T...)>::Connect(TypedSlot<void(T...)>* slot)
	{
        std::lock_guard<std::mutex> lock(mtx);
		_slots.insert(slot);
		return true;
	}

	template<class...T> 
	bool TypedSignalRelay<void(T...)>::Disconnect(ISlot* slot)
	{
        std::lock_guard<std::mutex> lock(mtx);
		return _slots.erase(static_cast<TypedSlot<void(T...)>*>(slot)) > 0;
	}

	template<class...T> 
	bool TypedSignalRelay<void(T...)>::Disconnect(ISignal* signal)
	{
		return false; // Currently not storing signal information to cache the connection types
	}

	template<class...T> 
	TypeInfo TypedSignalRelay<void(T...)>::GetSignature() const
	{
		return TypeInfo(typeid(void(T...)));
	}
	template<class...T> bool TypedSignalRelay<void(T...)>::HasSlots() const
	{
		return _slots.size() !=  0;
	}
	
	// ------------------------------------------------------------------
	// Return value specialization
	template<class R, class...T> 
	TypedSignalRelay<R(T...)>::TypedSignalRelay(): 
		_slot(nullptr)
	{
	}

	template<class R, class...T> 
	R TypedSignalRelay<R(T...)>::operator()(TypedSignal<R(T...)>* sig, T&... args)
	{
        std::lock_guard<std::mutex> lock(mtx);
		if (_slot)
		{
			return (*_slot)(args...);
		}
		THROW(debug) << "Slot not connected";
		return R();
	}

    template<class R, class...T>
    R TypedSignalRelay<R(T...)>::operator()(Context* ctx, T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if(_slot)
            return (*_slot)(args...);
        THROW(debug) << "Slot not connected";
        return R();
    }
    template<class R, class... T>
    R TypedSignalRelay<R(T...)>::operator()(T&... args)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (_slot && *_slot)
            return (*_slot)(args...);
        THROW(debug) << "Slot not connected";
        return R();
    }
	template<class R, class...T> 
	bool TypedSignalRelay<R(T...)>::Connect(ISlot* slot)
	{
		auto typed = dynamic_cast<TypedSlot<R(T...)>*>(slot);
		if (typed)
		{
			return Connect(typed);
		}
		return true;
	}
	
	template<class R, class...T>
	bool TypedSignalRelay<R(T...)>::Connect(TypedSlot<R(T...)>* slot)
	{
        std::lock_guard<std::mutex> lock(mtx);
		if (_slot == slot)
			return false;
		_slot = slot;
		return true;
	}

	template<class R, class...T> 
	bool TypedSignalRelay<R(T...)>::Connect(ISignal* signal)
	{
		return true;
	}

	template<class R, class...T> 
	bool TypedSignalRelay<R(T...)>::Connect(TypedSignal<R(T...)>* sig)
	{
		return true;
	}
	
	template<class R, class...T> 
	bool TypedSignalRelay<R(T...)>::Disconnect(ISlot* slot)
	{
        std::lock_guard<std::mutex> lock(mtx);
		if (_slot == slot)
		{
			_slot = nullptr;
			return true;
		}
		return false;
	}
	
	template<class R, class...T> 
	bool TypedSignalRelay<R(T...)>::Disconnect(ISignal* signal)
	{
		return false;
	}
	
	template<class R, class...T> 
	TypeInfo TypedSignalRelay<R(T...)>::GetSignature() const
	{
		return TypeInfo(typeid(R(T...)));
	}

	template<class R, class...T> 
	bool TypedSignalRelay<R(T...)>::HasSlots() const
	{
        return _slot != nullptr && *_slot;
	}
	
}
