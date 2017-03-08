#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"

#include <vector>
#include <memory>

namespace mo
{
    class IParameter;
    template<typename T> class ITypedParameter;
    template<typename T> class TypedInputParameter;
    class MO_EXPORTS IVariableManager
    {
    public:
        virtual void AddParameter(IParameter* param) = 0;

        virtual void RemoveParameter(IParameter* param) = 0;

        template<typename T> std::vector<IParameter*> GetOutputParameters();
        virtual std::vector<IParameter*> GetOutputParameters(TypeInfo type) = 0;

        virtual std::vector<IParameter*> GetAllParmaeters() = 0;
        virtual std::vector<IParameter*> GetAllOutputParameters() = 0;


        virtual IParameter* GetOutputParameter(std::string name) = 0;
        virtual IParameter* GetParameter(std::string name) = 0;

        // Links an output parameter to an input parameter with either a direct link or a buffered link.
        // Buffered links allow parameters to desync frame number between the producer and the consumer
        virtual void LinkParameters(IParameter* output, IParameter* input) = 0;
    };

    template<typename T> std::vector<IParameter*> IVariableManager::GetOutputParameters()
    {
        return GetOutputParameters(TypeInfo(typeid(T)));
    }
}
