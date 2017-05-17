#pragma once

namespace mo
{
    class IVariableManager;
}
namespace aq
{
    class IVariableSink
    {
    public:
        virtual void SerializeVariables(unsigned long long frame_number, mo::IVariableManager* manager) = 0;
    };
}
