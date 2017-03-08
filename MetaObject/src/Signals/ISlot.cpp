#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/Signals/ISignalRelay.hpp"
#include "MetaObject/IMetaObject.hpp"
using namespace mo;

ISlot::~ISlot()
{
	ThreadSpecificQueue::RemoveFromQueue(this);
	if (_parent)
	{
		ThreadSpecificQueue::RemoveFromQueue(_parent);
	}
}

void ISlot::SetParent(IMetaObject* parent)
{
	_parent = parent;
}

IMetaObject* ISlot::GetParent() const
{
	return _parent;
}
const Context* ISlot::GetContext() const
{
    if(_parent)
    {
        return _parent->GetContext();
    }
    return _ctx;
}
void ISlot::SetContext(Context* ctx)
{
    _ctx = ctx;
}