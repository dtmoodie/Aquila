#pragma once
#include <MetaObject/Detail/Export.hpp>
#include <functional>
#include <sstream>
#include <memory>
namespace mo
{
    class ISlot;
    class ISignalCaller;
    class ISignalRelay;
    class ISignalSink;

    class MO_EXPORTS SignalSerializationFactory
    {
    public:
        typedef std::function<void(ISlot*, std::istream&)> call_function_f;
        typedef std::function<ISignalCaller*(ISlot*)> signal_caller_constructor_f;
        typedef std::function<ISignalSink*(std::shared_ptr<ISignalRelay>, std::ostream&)> signal_sink_constructor_f;


        SignalSerializationFactory* Instance();
        call_function_f GetTextFunction(ISlot* slot);
        ISignalCaller* GetTextFunctor(ISlot* slot);

        void SetTextFunctions(ISlot* slot, 
            call_function_f function, 
            signal_caller_constructor_f caller_constructor,
            signal_sink_constructor_f sink_constructor);

    private:
        SignalSerializationFactory();
        struct impl;
        impl* _pimpl;
    };

    class MO_EXPORTS ISignalCaller
    {
    public:
        virtual ~ISignalCaller(){}
        virtual void Call(const std::istream& ss) = 0;
    };

    class MO_EXPORTS ISignalSink
    {
    public:
        virtual ~ISignalSink() {}
    };
}