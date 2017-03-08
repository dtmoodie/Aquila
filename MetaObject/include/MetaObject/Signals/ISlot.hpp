#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <memory>
namespace mo
{
    class ISignal;
    class Context;
    class Connection;
    class IMetaObject;
	class ISignalRelay;
    class MO_EXPORTS ISlot
    {
    public:
        virtual ~ISlot();
        virtual std::shared_ptr<Connection> Connect(ISignal* sig) = 0;
        virtual std::shared_ptr<Connection> Connect(std::shared_ptr<ISignalRelay>& relay) = 0;
		virtual bool Disconnect(std::weak_ptr<ISignalRelay> relay) = 0;
        virtual void Clear() = 0;
        virtual TypeInfo GetSignature() const = 0;
		IMetaObject* GetParent() const;
        const Context* GetContext() const;
        void SetContext(Context* ctx);

	protected:
		friend class IMetaObject;
		void SetParent(IMetaObject* parent);
		IMetaObject* _parent = nullptr;
        Context* _ctx = nullptr;
    };
}
