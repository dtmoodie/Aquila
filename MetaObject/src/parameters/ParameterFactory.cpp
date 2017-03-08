#include "MetaObject/Parameters/ParameterFactory.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
#include <map>

using namespace mo;
struct ParameterFactory::impl
{
    std::map<TypeInfo, std::map<int, create_f>> _registered_constructors;
    std::map<TypeInfo, create_f> _registered_constructors_exact;
};
ParameterFactory* ParameterFactory::instance()
{
    static ParameterFactory* inst = nullptr;
    if(inst == nullptr)
        inst = new ParameterFactory();
    if(inst->pimpl == nullptr)
        inst->pimpl.reset(new ParameterFactory::impl());
    return inst;
}

void ParameterFactory::RegisterConstructor(TypeInfo data_type, create_f function, ParameterTypeFlags parameter_type)
{
    pimpl->_registered_constructors[data_type][parameter_type] = function;
}
void ParameterFactory::RegisterConstructor(TypeInfo parameter_type, create_f function)
{
    pimpl->_registered_constructors_exact[parameter_type] = function;
}

std::shared_ptr<IParameter> ParameterFactory::create(TypeInfo data_type, ParameterTypeFlags parameter_type)
{
    auto itr = pimpl->_registered_constructors.find(data_type);
    if(itr != pimpl->_registered_constructors.end())
    {
        auto itr2 = itr->second.find(parameter_type);
        if(itr2 != itr->second.end())
        {
            return std::shared_ptr<IParameter>(itr2->second());
        }
        LOG(debug) << "Requested datatype (" << data_type.name() << ") exists but the specified parameter type : " << parameter_type << " does not.";
    }
    LOG(debug) << "Requested datatype (" << data_type.name() << ") does not exist";
    return nullptr;
}

std::shared_ptr<IParameter> ParameterFactory::create(TypeInfo parameter_type)
{
    auto itr = pimpl->_registered_constructors_exact.find(parameter_type);
    if(itr != pimpl->_registered_constructors_exact.end())
    {
        return std::shared_ptr<IParameter>(itr->second());
    }
    return nullptr;
}