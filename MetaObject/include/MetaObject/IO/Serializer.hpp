#pragma once
#include <MetaObject/IMetaObject.hpp>
#include "MetaObject/Detail/Export.hpp"
#include "ISerializer.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "shared_ptr.hpp"

#include <functional>

namespace cereal
{
    class BinaryInputArchive;
    class BinaryOutputArchive;
    class XMLOutputArchive;
    class XMLInputArchive;
	class JSONOutputArchive;
	class JSONInputArchive;
}

namespace mo
{
    class MO_EXPORTS SerializerFactory
    {
    public:
        enum SerializationType
        {
            Binary_e = 0,
            xml_e,
            json_e
        };
        static void Serialize(const rcc::shared_ptr<IMetaObject>& obj, std::ostream& os, SerializationType type);
        static void DeSerialize(IMetaObject* obj, std::istream& os, SerializationType type);
        static rcc::shared_ptr<IMetaObject> DeSerialize(std::istream& os, SerializationType type);

        typedef std::function<void(const IMetaObject*, cereal::BinaryOutputArchive&)> BinarySerialize_f;
        typedef std::function<void(IMetaObject*, cereal::BinaryInputArchive&)> BinaryDeSerialize_f;

        typedef std::function<void(const IMetaObject*, cereal::XMLOutputArchive&)> XMLSerialize_f;
        typedef std::function<void(IMetaObject*, cereal::XMLInputArchive&)> XMLDeSerialize_f;

		typedef std::function<void(const IMetaObject*, cereal::JSONOutputArchive&)> JSONSerialize_f;
		typedef std::function<void(IMetaObject*, cereal::JSONInputArchive&)> JSONDeSerialize_f;

        static void RegisterSerializationFunctionBinary(const char* obj_type, BinarySerialize_f f);
        static void RegisterDeSerializationFunctionBinary(const char* obj_type, BinaryDeSerialize_f f);
        static void RegisterSerializationFunctionXML(const char* obj_type, XMLSerialize_f f);
        static void RegisterDeSerializationFunctionXML(const char* obj_type, XMLDeSerialize_f f);
		static void RegisterSerializationFunctionJSON(const char* obj_type, JSONSerialize_f f);
		static void RegisterDeSerializationFunctionJSON(const char* obj_type, JSONDeSerialize_f f);

		static BinarySerialize_f    GetSerializationFunctionBinary(const char* obj_type);
		static BinaryDeSerialize_f  GetDeSerializationFunctionBinary(const char* obj_type);
		static XMLSerialize_f       GetSerializationFunctionXML(const char* obj_type);
		static XMLDeSerialize_f     GetDeSerializationFunctionXML(const char* obj_type);
		static JSONSerialize_f       GetSerializationFunctionJSON(const char* obj_type);
		static JSONDeSerialize_f     GetDeSerializationFunctionJSON(const char* obj_type);
    };
    MO_EXPORTS void StartSerialization();
    MO_EXPORTS bool Serialize(cereal::BinaryOutputArchive& ar, const IMetaObject* obj);
    MO_EXPORTS bool DeSerialize(cereal::BinaryInputArchive& ar, IMetaObject* obj);
    MO_EXPORTS bool Serialize(cereal::XMLOutputArchive& ar, const IMetaObject* obj);
    MO_EXPORTS bool DeSerialize(cereal::XMLInputArchive& ar, IMetaObject* obj);
    MO_EXPORTS bool Serialize(cereal::JSONOutputArchive& ar, const IMetaObject* obj);
    MO_EXPORTS bool DeSerialize(cereal::JSONInputArchive& ar, IMetaObject* obj);
    MO_EXPORTS void SetHasBeenSerialized(ObjectId id);
    MO_EXPORTS bool CheckHasBeenSerialized(ObjectId id);
    MO_EXPORTS void EndSerialization();
}
