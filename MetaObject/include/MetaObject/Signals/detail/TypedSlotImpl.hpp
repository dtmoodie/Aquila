#pragma once
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Signals/TypedSignalRelay.hpp"
#include "MetaObject/Signals/Connection.hpp"

namespace mo
{
	template<class Sig> class TypedSlot;

	template<class R, class...T>
	TypedSlot<R(T...)>::TypedSlot()
	{
		
	}

	template<class R, class...T>
	TypedSlot<R(T...)>::TypedSlot(const std::function<R(T...)>& other) :
		std::function<R(T...)>(other)
	{
		
	}
    
    template<class R, class...T>
    TypedSlot<R(T...)>::TypedSlot(std::function<R(T...)>&& other):
        std::function<R(T...)>(other)
    {
    }

	template<class R, class...T> 
	TypedSlot<R(T...)>::~TypedSlot()
	{
        Clear();
	}

	template<class R, class...T>
	TypedSlot<R(T...)>& TypedSlot<R(T...)>::operator=(const std::function<R(T...)>& other)
	{
		std::function<R(T...)>::operator=(other);
		return *this;
	}

	template<class R, class...T>
	TypedSlot<R(T...)>& TypedSlot<R(T...)>::operator=(const TypedSlot<R(T...)>& other)
	{
		this->_relays = other._relays;
		return *this;
	}


	template<class R, class...T> 
	std::shared_ptr<Connection> TypedSlot<R(T...)>::Connect(ISignal* sig)
	{
		auto typed = dynamic_cast<TypedSignal<R(T...)>*>(sig);
		if (typed)
		{
			return Connect(typed);
		}
		return std::shared_ptr<Connection>();
	}

	template<class R, class...T>
	std::shared_ptr<Connection> TypedSlot<R(T...)>::Connect(TypedSignal<R(T...)>* typed)
	{
		std::shared_ptr<TypedSignalRelay<R(T...)>> relay(new TypedSignalRelay<R(T...)>());
		relay->Connect(this);
		typed->Connect(relay);
		_relays.push_back(relay);
		return std::shared_ptr<Connection>(new SlotConnection(this, relay));
	}
    template<class R, class...T>
    std::shared_ptr<Connection> TypedSlot<R(T...)>::Connect(std::shared_ptr<TypedSignalRelay<R(T...)>>& relay)
    {
        relay->Connect(this);
        _relays.push_back(relay);
        return std::shared_ptr<Connection>(new SlotConnection(this, std::dynamic_pointer_cast<ISignalRelay>(relay)));
    }
	template<class R, class...T> 
	std::shared_ptr<Connection> TypedSlot<R(T...)>::Connect(std::shared_ptr<ISignalRelay>& relay)
	{
		if (relay == nullptr)
		{
			relay.reset(new TypedSignalRelay<R(T...)>());
		}
		auto typed = std::dynamic_pointer_cast<TypedSignalRelay<R(T...)>>(relay);
		if (typed)
		{
			_relays.push_back(typed);
			if (relay->Connect(this))
			{
				return std::shared_ptr<Connection>(new SlotConnection(this, relay));
			}
		}
		return std::shared_ptr<Connection>();
	}

	template<class R, class...T>
	bool TypedSlot<R(T...)>::Disconnect(std::weak_ptr<ISignalRelay> relay_)
	{
		auto relay = relay_.lock();
		for (auto itr = _relays.begin(); itr != _relays.end(); ++itr)
		{
			if ((*itr) == relay)
			{
				(*itr)->Disconnect(this);
				_relays.erase(itr);
				return true;
			}
		}
		return false;
	}
    template<class R, class... T>
    void TypedSlot<R(T...)>::Clear()
    {
        for (auto& relay : _relays)
        {
            relay->Disconnect(this);
        }
    }

	template<class R, class...T> 
	TypeInfo TypedSlot<R(T...)>::GetSignature() const
	{
		return TypeInfo(typeid(R(T...)));
	}
}
