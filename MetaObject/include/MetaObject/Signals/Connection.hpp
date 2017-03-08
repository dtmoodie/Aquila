#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <memory>
namespace mo
{
	class ISlot;
	class ISignal;
	class ISignalRelay;
	class IMetaObject;

	// A connection is only valid for as long as the underlying slot is valid
    class MO_EXPORTS Connection
    {
    public:
        virtual ~Connection();
		virtual bool Disconnect() = 0;
    };

	class MO_EXPORTS SlotConnection : public Connection
	{
	public:
		SlotConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay);
        virtual ~SlotConnection();
		virtual bool Disconnect();
	protected:
		ISlot* _slot;
		std::weak_ptr<ISignalRelay> _relay;
	};

    class MO_EXPORTS ClassConnection: public SlotConnection
    {
    public:
		ClassConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay, IMetaObject* obj);
		~ClassConnection();
		virtual bool Disconnect();
	protected:
		IMetaObject* _obj;
    };

	class MO_EXPORTS SignalConnection : public Connection
	{
	public:
		SignalConnection(ISignal* signal, std::shared_ptr<ISignalRelay> relay);
		bool Disconnect();
	protected:
		ISignal* _signal;
		std::weak_ptr<ISignalRelay> _relay;
	};

    
} // namespace Signals
