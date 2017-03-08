#include "MetaObject/Parameters/VariableManager.h"
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"
#include "MetaObject/Signals/TypedSignalRelay.hpp"
#include <map>
using namespace mo;
struct VariableManager::impl
{
    std::map<std::string, IParameter*> _parameters;
    //std::map<std::string, std::shared_ptr<Connection>> _delete_connections;
	TypedSlot<void(IParameter*)> delete_slot;
};
VariableManager::VariableManager()
{
    pimpl = new impl();
	pimpl->delete_slot = std::bind(&VariableManager::RemoveParameter, this, std::placeholders::_1);
}
VariableManager::~VariableManager()
{
    delete pimpl;
}
void VariableManager::AddParameter(IParameter* param)
{
    pimpl->_parameters[param->GetTreeName()] = param;
    param->RegisterDeleteNotifier(&pimpl->delete_slot);
}
void VariableManager::RemoveParameter(IParameter* param)
{
    pimpl->_parameters.erase(param->GetTreeName());
}
std::vector<IParameter*> VariableManager::GetOutputParameters(TypeInfo type)
{
    std::vector<IParameter*> valid_outputs;
    for(auto itr = pimpl->_parameters.begin(); itr != pimpl->_parameters.end(); ++itr)
    {
        if(itr->second->GetTypeInfo() == type && itr->second->CheckFlags(Output_e))
        {
            valid_outputs.push_back(itr->second);
        }   
    }
    return valid_outputs;
}
std::vector<IParameter*> VariableManager::GetAllParmaeters()
{
    std::vector<IParameter*> output;
    for(auto& itr : pimpl->_parameters)
    {
        output.push_back(itr.second);
    }
    return output;
}
std::vector<IParameter*> VariableManager::GetAllOutputParameters()
{
    std::vector<IParameter*> output;
    for(auto& itr : pimpl->_parameters)
    {
        if(itr.second->CheckFlags(Output_e))
        {
            output.push_back(itr.second);
        }
    }
    return output;    
}
IParameter* VariableManager::GetParameter(std::string name)
{
    auto itr = pimpl->_parameters.find(name);
    if(itr != pimpl->_parameters.end())
    {
        return itr->second;
    }
    return nullptr;
}

IParameter* VariableManager::GetOutputParameter(std::string name)
{
    auto itr = pimpl->_parameters.find(name);
    if(itr != pimpl->_parameters.end())
    {
        return itr->second;
    }
    // Check if the passed in value is the item specific name
    std::vector<IParameter*> potentials;
    for(auto& itr : pimpl->_parameters)
    {
        if(itr.first.find(name) != std::string::npos)
        {
            potentials.push_back(itr.second);
        }
    }
    if(potentials.size())
    {
        if(potentials.size() > 1)
        {
            std::stringstream ss;
            for(auto potential : potentials)
                ss << potential->GetTreeName() << "\n";
            LOG(debug) << "Warning ambiguous name \"" << name << "\" passed in, multiple potential matches\n " << ss.str();
        }
        return potentials[0];
    }
    LOG(debug) << "Unable to find parameter named " << name;
    return nullptr;
}
void VariableManager::LinkParameters(IParameter* output, IParameter* input)
{
    if(auto input_param = dynamic_cast<InputParameter*>(input))
        input_param->SetInput(output);
}
