#pragma once

#include "MetaObject/Parameters/IVariableManager.h"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Detail/Export.hpp"

namespace mo
{
    class IParameter;
    class MO_EXPORTS VariableManager: public IVariableManager
    {
    public:
        VariableManager();
        ~VariableManager();
        virtual void AddParameter(IParameter* param);
        virtual void RemoveParameter(IParameter* param);

        virtual std::vector<IParameter*> GetOutputParameters(TypeInfo type);
        virtual std::vector<IParameter*> GetAllParmaeters();
        virtual std::vector<IParameter*> GetAllOutputParameters();

        virtual IParameter* GetOutputParameter(std::string name);
        virtual IParameter* GetParameter(std::string name);
        virtual void LinkParameters(IParameter* output, IParameter* input);

    private:
        struct impl;
        impl* pimpl;
    };

    // By default uses a buffered connection when linking two parameters together
    class MO_EXPORTS BufferedVariableManager : public VariableManager
    {
    };
}
