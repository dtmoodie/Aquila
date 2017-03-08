#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/ISignalRelay.hpp"
#include "MetaObject/Signals/ISignal.hpp"

using namespace mo;


Connection::~Connection()
{

}

SlotConnection::SlotConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay):
	_slot(slot), _relay(relay)
{

}
SlotConnection::~SlotConnection()
{
    //Disconnect();
}
bool SlotConnection::Disconnect()
{
	if (_slot)
	{
		return _slot->Disconnect(_relay);
	}
	return false;
}

ClassConnection::ClassConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay, IMetaObject* obj):
	SlotConnection(slot, relay), _obj(obj)
{

}

ClassConnection::~ClassConnection()
{
    Disconnect();	
}

bool ClassConnection::Disconnect()
{
	if (SlotConnection::Disconnect())
	{
		ThreadSpecificQueue::RemoveFromQueue(_obj);
		return true;
	}
	return false;
}

SignalConnection::SignalConnection(ISignal* signal, std::shared_ptr<ISignalRelay> relay):
	_signal(signal), _relay(relay)
{

}
bool SignalConnection::Disconnect()
{
	if (_signal)
	{
		return _signal->Disconnect(_relay);
	}
    return false;
}