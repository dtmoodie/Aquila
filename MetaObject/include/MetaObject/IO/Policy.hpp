#pragma once
#include "Serializer.hpp"

#include "cereal/archives/binary.hpp"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/json.hpp"

namespace mo
{
    template<class T, int N, typename Enable> struct MetaObjectPolicy;

    template<class T> struct SerializationPolicyImpl
    {
        SerializationPolicyImpl()
        {
            SerializerFactory::RegisterSerializationFunctionBinary(T::GetTypeNameStatic(), std::bind(&SerializationPolicyImpl<T>::SerializeBinary, std::placeholders::_1, std::placeholders::_2));
            SerializerFactory::RegisterDeSerializationFunctionBinary(T::GetTypeNameStatic(), std::bind(&SerializationPolicyImpl<T>::DeSerializeBinary, std::placeholders::_1, std::placeholders::_2));
            SerializerFactory::RegisterSerializationFunctionXML(T::GetTypeNameStatic(), std::bind(&SerializationPolicyImpl<T>::SerializeXml, std::placeholders::_1, std::placeholders::_2));
            SerializerFactory::RegisterDeSerializationFunctionXML(T::GetTypeNameStatic(), std::bind(&SerializationPolicyImpl<T>::DeSerializeXml, std::placeholders::_1, std::placeholders::_2));
        }
        static void SerializeBinary(const IMetaObject* obj, cereal::BinaryOutputArchive& ar)
        {
            const T* typed = static_cast<const T*>(obj);
            if(typed)
            {
                ar(*typed);
            }
        }
        static void DeSerializeBinary(IMetaObject* obj, cereal::BinaryInputArchive& ar)
        {
            T* typed = static_cast<T*>(obj);
            if (typed)
            {
                ar(*typed);
            }
        }
        static void SerializeXml(const IMetaObject* obj, cereal::XMLOutputArchive& ar)
        {
            const T* typed = static_cast<const T*>(obj);
            if (typed)
            {
                ar(*typed);
            }
        }
        static void DeSerializeXml(IMetaObject* obj, cereal::XMLInputArchive& ar)
        {
            T* typed = static_cast<T*>(obj);
            if (typed)
            {
                ar(*typed);
            }
        }
    };

#define MO_INSTANTIATE_SERIALIZATION_POLICY_(N)  \
    template<class T> struct MetaObjectPolicy<T, N, void>: public MetaObjectPolicy<T, N - 1, void>, public SerializationPolicyImpl<T> \
    { \
        MetaObjectPolicy(): MetaObjectPolicy<T, N-1, void>() {} \
    };

MO_INSTANTIATE_SERIALIZATION_POLICY_(__COUNTER__);
}
