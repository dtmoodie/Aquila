#include <MetaObject/Signals/Serialization.hpp>
#include <MetaObject/Signals/ISlot.hpp>
#include <MetaObject/Detail/TypeInfo.h>
#include <map>

using namespace mo;

struct RegisteredFunctions
{
    SignalSerializationFactory::call_function_f call;
    SignalSerializationFactory::signal_caller_constructor_f caller_constructor;
    SignalSerializationFactory::signal_sink_constructor_f sink_constructor;
};
struct SignalSerializationFactory::impl
{
    std::map<mo::TypeInfo, RegisteredFunctions> _registry;
};

SignalSerializationFactory::SignalSerializationFactory()
{
    _pimpl = new impl();
}

SignalSerializationFactory* SignalSerializationFactory::Instance()
{
    static SignalSerializationFactory inst;
    return &inst;
}

SignalSerializationFactory::call_function_f SignalSerializationFactory::GetTextFunction(ISlot* slot)
{
    auto itr = _pimpl->_registry.find(slot->GetSignature());
    if (itr != _pimpl->_registry.end())
    {
        return itr->second.call;
    }
    return call_function_f();
}

ISignalCaller* SignalSerializationFactory::GetTextFunctor(ISlot* slot)
{
    auto itr = _pimpl->_registry.find(slot->GetSignature());
    if(itr != _pimpl->_registry.end())
    {
        return itr->second.caller_constructor(slot);
    }
    return nullptr;
}

void SignalSerializationFactory::SetTextFunctions(ISlot* slot,
    call_function_f function,
    signal_caller_constructor_f caller_constructor,
    signal_sink_constructor_f sink_constructor)
{
    _pimpl->_registry[slot->GetSignature()] = {function, caller_constructor, sink_constructor};
}