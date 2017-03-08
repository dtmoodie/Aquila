#pragma once
#include "CerealParameters.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "shared_ptr.hpp"


#include <type_traits>

namespace cereal
{
    template<class AR, class T>
    typename std::enable_if<std::is_base_of<mo::IMetaObject, T>::value>::type
    load(AR& ar, rcc::shared_ptr<T>& obj)
    {
        std::string type;
        ar(CEREAL_NVP(type));
        obj = mo::MetaObjectFactory::Instance()->Create(type.c_str());
        if(!obj)
        {
            LOG(warning) << "Unable to create object with type: " << type;
            return;
        }
        auto parameters = obj->GetParameters();
        ar(CEREAL_NVP(parameters));
    }

    template<class AR, class T>
    typename std::enable_if<std::is_base_of<mo::IMetaObject, T>::value>::type
    save(AR& ar, rcc::shared_ptr<T> const& obj)
    {
        if(obj)
        {
            auto parameters = obj->GetParameters();
            std::string type = obj->GetTypeName();
            ar(CEREAL_NVP(type));
            ar(CEREAL_NVP(parameters));
        }
    }
}
