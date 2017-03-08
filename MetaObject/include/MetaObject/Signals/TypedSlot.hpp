#pragma once
#include "ISlot.hpp"
#include "TypedSignal.hpp"
#include "MetaObject/Context.hpp"
#include "TypedSignalRelay.hpp"
#include <functional>
#include <future>
namespace mo
{
    template<typename Sig> class TypedSlot{};
	template<typename Sig> class TypedSignalRelay;

    template<typename R, typename... T> class TypedSlot<R(T...)>: public std::function<R(T...)>, public ISlot
    {
    public:
		TypedSlot();
		TypedSlot(const std::function<R(T...)>& other);
        TypedSlot(std::function<R(T...)>&& other);
		~TypedSlot();

		TypedSlot& operator=(const std::function<R(T...)>& other);
		TypedSlot& operator=(const TypedSlot& other);

		std::shared_ptr<Connection> Connect(ISignal* sig);
		std::shared_ptr<Connection> Connect(TypedSignal<R(T...)>* signal);
		std::shared_ptr<Connection> Connect(std::shared_ptr<ISignalRelay>& relay);
        std::shared_ptr<Connection> Connect(std::shared_ptr<TypedSignalRelay<R(T...)>>& relay);
		virtual bool Disconnect(std::weak_ptr<ISignalRelay> relay);
        void Clear();
		TypeInfo GetSignature() const;
	protected:
		std::vector< std::shared_ptr< TypedSignalRelay<R(T...)> > > _relays;
		
    };
}
#include "detail/TypedSlotImpl.hpp"
