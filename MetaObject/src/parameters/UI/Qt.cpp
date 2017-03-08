/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/

#include "MetaObject/Parameters/UI/WidgetFactory.hpp"
#include "MetaObject/Parameters/UI/Qt/DefaultProxy.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Parameters/IParameter.hpp"

#include <map>

using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

struct WidgetFactory::impl
{
    std::map<TypeInfo, WidgetFactory::HandlerConstructor_f> registry;
};

WidgetFactory::WidgetFactory()
{
    _pimpl = new impl();
}
WidgetFactory* WidgetFactory::Instance()
{
    static WidgetFactory inst;
    return &inst;
}

void WidgetFactory::RegisterConstructor(const TypeInfo& type, WidgetFactory::HandlerConstructor_f f)
{
    LOG(trace) << "Registering type " << type.name();
    auto itr = _pimpl->registry.find(type);
    if(itr == _pimpl->registry.end())
    {
        _pimpl->registry[type] = f;
    }   
}
std::string print_types(std::map<TypeInfo, WidgetFactory::HandlerConstructor_f>& registry)
{
    std::stringstream ss;
    for(auto& item : registry)
    {
        ss << item.first.name() << ", ";
    }
    return ss.str();
}

std::shared_ptr<IParameterProxy> WidgetFactory::CreateProxy(IParameter* param)
{
    std::string typeName = param->GetTypeInfo().name();
    std::string treeName = param->GetTreeName();
    auto itr = _pimpl->registry.find(param->GetTypeInfo());
    if (itr == _pimpl->registry.end())
    {
        LOG(debug) << "No Widget Factory registered for type " << typeName 
            << " unable to make widget for parameter: " << treeName 
            << ".  Known types: " << print_types(_pimpl->registry);
            
        return std::shared_ptr<IParameterProxy>(new DefaultProxy(param));
    }
    LOG(trace) << "Creating handler for " << typeName << " " << treeName;
    return itr->second(param);
}

