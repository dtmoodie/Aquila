#pragma once
#include "MetaObject/Signals/RelayFactory.hpp"
#include "MetaObject/Detail/TypeInfo.h"
namespace mo
{
    template<class Sig> class SignalRelayConstructor
    {
    public:
        SignalRelayConstructor()
        {
            RelayFactory::Instance()->RegisterCreator(
                []()->ISignalRelay*
            {
                return new TypedSignalRelay<Sig>();
            }, TypeInfo(typeid(Sig)));
        }
    };
}