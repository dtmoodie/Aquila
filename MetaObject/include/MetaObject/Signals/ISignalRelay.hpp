#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
namespace mo
{
	class ISlot;
	class ISignal;
	class Connection;
	template<class Sig> class TypedSlot;
	template<class Sig> class TypedSignal;
	class MO_EXPORTS ISignalRelay
	{
	public:
		virtual ~ISignalRelay() {}
		virtual TypeInfo GetSignature() const = 0;
        virtual bool HasSlots() const = 0;
	protected:
		friend class ISlot;
		friend class ISignal;
		template<class T> friend class TypedSignal;
		template<class T> friend class TypedSlot;
		virtual bool Connect(ISlot* slot) = 0;
		virtual bool Disconnect(ISlot* slot) = 0;

		virtual bool Connect(ISignal* signal) = 0;
		virtual bool Disconnect(ISignal* signal) = 0;
	};
}
