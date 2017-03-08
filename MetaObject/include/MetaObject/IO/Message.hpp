#pragma once
#include <string>
#include <map>
#include "ObjectInterface.h"
namespace mo
{
    class IParameter;
    class IMetaObject;

    struct Message
    {
        std::string topic;
        std::map<ObjectId, IMetaObject*> objects;
        std::map<std::string, IParameter*> parameters;
        template<class AR> void serialize(AR& ar)
        {
            ar(topic);
            ar(objects);
            ar(parameters);
        }
    };
    struct ParameterUpdate
    {
        
    };
}