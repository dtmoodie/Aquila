#include "IObjectState.hpp"
#include "IObject.h"
#include "shared_ptr.hpp"

IObjectSharedState* IObjectSharedState::Get(IObject* obj)
{
    return obj->GetConstructor()->GetState(obj->GetPerTypeId());
}

IObjectSharedState::IObjectSharedState(IObject* obj, IObjectConstructor* constructor)
{
    object_ref_count = 0;
    state_ref_count = 0;
    object = obj;
    this->constructor = constructor;
    id = obj->GetPerTypeId();
}

IObjectSharedState::~IObjectSharedState()
{
    if(object)
    {
        delete object;
        object = 0;
    }
    constructor->DeRegister(id);
}

IObject* IObjectSharedState::GetIObject()
{
    return object;
}

rcc::shared_ptr<IObject> IObjectSharedState::GetSharedPtr()
{
    return rcc::shared_ptr<IObject>(this);
}

rcc::weak_ptr<IObject> IObjectSharedState::GetWeakPtr()
{
    return rcc::weak_ptr<IObject>(this);
}

void IObjectSharedState::IncrementObject()
{
    ++object_ref_count;
}

void IObjectSharedState::IncrementState()
{
    ++state_ref_count;
}

void IObjectSharedState::DecrementObject()
{
    --object_ref_count;
    if(object_ref_count == 0)
    {
        delete object;
        object = 0;
    }
}

void IObjectSharedState::DecrementState()
{
    --state_ref_count;
    if(state_ref_count == 0)
    {
        delete this;
    }
}
void IObjectSharedState::SetObject(IObject* obj)
{
    object = obj;
}
void IObjectSharedState::SetConstructor(IObjectConstructor* constructor)
{
    this->constructor = constructor;
}
int IObjectSharedState::ObjectCount() const
{
    return object_ref_count;
}
int IObjectSharedState::StateCount() const
{
    return state_ref_count;
}
