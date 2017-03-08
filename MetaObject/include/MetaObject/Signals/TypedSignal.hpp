#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Signals/ISignal.hpp"
#include <mutex>
#include <memory>
#include <vector>
namespace mo
{
    class IMetaObject;
	class Context;
    class Connection;
    template<class Sig> class TypedSignalRelay;
    template<class Sig> class TypedSignal{};
	template<class...T> class MO_EXPORTS TypedSignal<void(T...)> : public ISignal
	{
	public:
		TypedSignal();
		void operator()(T... args);
        void operator()(Context* ctx, T... args);
		TypeInfo GetSignature() const;

		std::shared_ptr<Connection> Connect(ISlot* slot);
		std::shared_ptr<Connection> Connect(std::shared_ptr<ISignalRelay>& relay);
		std::shared_ptr<Connection> Connect(std::shared_ptr<TypedSignalRelay<void(T...)>>& relay);

		bool Disconnect();
		bool Disconnect(ISlot* slot);
		bool Disconnect(std::weak_ptr<ISignalRelay> relay);
	protected:
        std::recursive_mutex mtx;
		std::vector<std::shared_ptr<TypedSignalRelay<void(T...)>>> _typed_relays;
	};

	template<class R, class...T> class MO_EXPORTS TypedSignal<R(T...)> : public ISignal
    {
    public:
		TypedSignal();
		R operator()(T... args);
        R operator()(Context* ctx, T... args);
		TypeInfo GetSignature() const;

		std::shared_ptr<Connection> Connect(ISlot* slot);
		std::shared_ptr<Connection> Connect(std::shared_ptr<ISignalRelay>& relay);
		std::shared_ptr<Connection> Connect(std::shared_ptr<TypedSignalRelay<R(T...)>>& relay);

		bool Disconnect();
		bool Disconnect(ISlot* slot);
		bool Disconnect(std::weak_ptr<ISignalRelay> relay);
	protected:
        std::recursive_mutex mtx;
		std::shared_ptr<TypedSignalRelay<R(T...)>> _typed_relay;
    };
}
#include "detail/TypedSignalImpl.hpp"
