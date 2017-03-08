#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/IMetaObject.hpp"
using namespace mo;

const Context* ISignal::GetContext() const
{
    if(_parent)
    {
        return _parent->GetContext();
    }
    return nullptr;
}
IMetaObject* ISignal::GetParent() const
{
	return _parent;
}

void ISignal::SetParent(IMetaObject* parent)
{
	_parent = parent;
}