#pragma once
#include "Aquila/nodes/Node.hpp"
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include "serialize.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/serialization/Serializer.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>

namespace cereal {

/*template<class AR, class T> void save(AR& ar, rcc::shared_ptr<T> const & m)
    {
        if(m)
        {
            if (mo::CheckHasBeenSerialized(m->GetObjectId()))
            {
                std::string type = m->GetTypeName();
                ObjectId id = m->GetObjectId();
                ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
                ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
                ar(make_nvp("TypeName", type));
                return;
            }
            if (std::is_base_of<aq::nodes::Node, T>::value)
            {
                aq::Serialize(ar, m.get());
            }
            else
            {
                mo::Serialize(ar, m.get());
            }
            mo::SetHasBeenSerialized(m->GetObjectId());
        }
    }

    template<class AR, class T> void load(AR& ar, rcc::shared_ptr<T> & m)
    {
        std::string type;
        ObjectId id;
        if (m.empty())
        {
            ar(make_nvp("TypeName", type));
            ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
            ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
            if (auto obj = mo::MetaObjectFactory::instance()->get(id, type.c_str()))
            {
                m = obj;
            }
            else
            {
                m = mo::MetaObjectFactory::instance()->create(type.c_str());
            }
        }
        if (mo::CheckHasBeenSerialized(m->GetObjectId()))
            return;
        if (std::is_base_of<aq::nodes::Node, T>::value)
        {
            aq::DeSerialize(ar, m.get());
        }else
        {
            mo::DeSerialize(ar, m.get());
        }
        mo::SetHasBeenSerialized(m->GetObjectId());
    }
    template<class AR, class T> void save(AR& ar, rcc::weak_ptr<T> const & m)
    {
        if(m)
        {
            std::string type = m->GetTypeName();
            ObjectId id = m->GetObjectId();
            ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
            ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
            ar(make_nvp("TypeName", type));
        }
    }

    template<class AR, class T> void load(AR& ar, rcc::weak_ptr<T> & m)
    {
        std::string type;
        ObjectId id;
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(make_nvp("TypeName", type));
        m.reset(mo::MetaObjectFactory::instance()->get(id, type.c_str()));
    }*/
}
