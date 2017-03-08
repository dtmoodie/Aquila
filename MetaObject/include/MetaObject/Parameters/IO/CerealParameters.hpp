#pragma once
#include "cereal/archives/json.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "SerializationFunctionRegistry.hpp"

namespace cereal
{
    template<class AR>
    void load(AR& ar, std::vector<mo::IParameter*>& parameters)
    {
        for (auto& param : parameters)
        {
            if (param->CheckFlags(mo::Output_e) || param->CheckFlags(mo::Input_e))
                continue;
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetLoadFunction(param->GetTypeInfo(), ar);
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to deserialize " << param->GetName() << " of type " << param->GetTypeInfo().name();
                }
            }
            else
            {
                LOG(debug) << "No deserialization function exists for  " << param->GetName() << " of type " << param->GetTypeInfo().name();
            }
        }
    }
    template<class AR>
    void save(AR& ar, std::vector<mo::IParameter*> const& parameters)
    {
        for (auto& param : parameters)
        {
            if (param->CheckFlags(mo::Output_e) || param->CheckFlags(mo::Input_e))
                continue;
            auto func1 = mo::SerializationFunctionRegistry::Instance()->GetSaveFunction(param->GetTypeInfo(), ar);
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to deserialize " << param->GetName() << " of type " << param->GetTypeInfo().name();
                }
            }
            else
            {
                LOG(debug) << "No serialization function exists for  " << param->GetName() << " of type " << param->GetTypeInfo().name();
            }
        }
    }

}
