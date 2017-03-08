#pragma once
#include "MetaObject/Detail/Export.hpp"
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
    class IParameter;
    class TypeInfo;
    class Context;
    
    class MO_EXPORTS IBinarySerializer
    {
    public:
        virtual ~IBinarySerializer() {}
        virtual void SetInput(IParameter* input) = 0;
    };
    
    class MO_EXPORTS IXmlSerializer
    {
    
    };
    
    class MO_EXPORTS IJsonSerializer
    {
    
    };
    
    class MO_EXPORTS ITextSerializer
    {
    
    };
    
    class MO_EXPORTS SerializationFunctionRegistry
    {
    public:
        static SerializationFunctionRegistry* Instance();

        typedef std::function<bool(IParameter*, cereal::BinaryOutputArchive&)> SerializeBinary_f;
        typedef std::function<bool(IParameter*, cereal::BinaryInputArchive&)> DeSerializeBinary_f;
        typedef std::function<bool(IParameter*, cereal::XMLOutputArchive&)> SerializeXml_f;
        typedef std::function<bool(IParameter*, cereal::XMLInputArchive&)> DeSerializeXml_f;
        typedef std::function<bool(IParameter*, cereal::JSONOutputArchive&)> SerializeJson_f;
        typedef std::function<bool(IParameter*, cereal::JSONInputArchive&)> DeSerializeJson_f;
        typedef std::function<bool(IParameter*, std::stringstream&)> SerializeText_f;
        typedef std::function<bool(IParameter*, std::stringstream&)> DeSerializeText_f;

        SerializeBinary_f   GetBinarySerializationFunction(const TypeInfo& type);
        DeSerializeBinary_f GetBinaryDeSerializationFunction(const TypeInfo& type);

        SerializeXml_f      GetXmlSerializationFunction(const TypeInfo& type);
        DeSerializeXml_f    GetXmlDeSerializationFunction(const TypeInfo& type);

        SerializeJson_f     GetJsonSerializationFunction(const TypeInfo& type);
        DeSerializeJson_f   GetJsonDeSerializationFunction(const TypeInfo& type);

        SerializeText_f     GetTextSerializationFunction(const TypeInfo& type);
        DeSerializeText_f   GetTextDeSerializationFunction(const TypeInfo& type);

        SerializeBinary_f GetSaveFunction(const TypeInfo& type, cereal::BinaryOutputArchive& ar);
        DeSerializeBinary_f GetLoadFunction(const TypeInfo& type, cereal::BinaryInputArchive& ar);

        SerializeXml_f GetSaveFunction(const TypeInfo& type, cereal::XMLOutputArchive& ar);
        DeSerializeXml_f GetLoadFunction(const TypeInfo& type, cereal::XMLInputArchive& ar);

        SerializeJson_f GetSaveFunction(const TypeInfo& type, cereal::JSONOutputArchive& ar);
        DeSerializeJson_f GetLoadFunction(const TypeInfo& type, cereal::JSONInputArchive& ar);

        void SetBinarySerializationFunctions(const TypeInfo& type, SerializeBinary_f serialize, DeSerializeBinary_f deserialize);
        
        void SetXmlSerializationFunctions(const TypeInfo& type, SerializeXml_f serialize, DeSerializeXml_f deserialize);
        
        void SetJsonSerializationFunctions(const TypeInfo& type, SerializeJson_f serialize, DeSerializeJson_f deserialize);
        
        void SetTextSerializationFunctions(const TypeInfo& type, SerializeText_f serialize, DeSerializeText_f deserialize);
    private:
        SerializationFunctionRegistry();
        ~SerializationFunctionRegistry();
        struct impl;
        impl* _pimpl;
    };
}
