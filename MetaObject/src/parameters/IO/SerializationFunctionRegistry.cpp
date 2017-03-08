#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/Detail/TypeInfo.h"

#include <map>

using namespace mo;
struct SerializationFunctionRegistry::impl
{
    std::map<TypeInfo, std::pair<SerializeBinary_f, DeSerializeBinary_f >> _binary_map;
    std::map<TypeInfo, std::pair<SerializeXml_f, DeSerializeXml_f >> _xml_map;
    std::map<TypeInfo, std::pair<SerializeJson_f, DeSerializeJson_f >> _json_map;
    std::map<TypeInfo, std::pair<SerializeText_f, DeSerializeText_f >> _text_map;
};

SerializationFunctionRegistry::SerializationFunctionRegistry()
{
    _pimpl = new impl();    
}

SerializationFunctionRegistry::~SerializationFunctionRegistry()
{
    delete _pimpl;
}

SerializationFunctionRegistry* SerializationFunctionRegistry::Instance()
{
    static SerializationFunctionRegistry* g_inst = nullptr;
    if(g_inst == nullptr)
        g_inst = new SerializationFunctionRegistry();
    return g_inst;
}

SerializationFunctionRegistry::SerializeBinary_f SerializationFunctionRegistry::GetBinarySerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_binary_map.find(type);
    if(itr != _pimpl->_binary_map.end())
    {
        return itr->second.first;
    }
    return SerializeBinary_f();
}

SerializationFunctionRegistry::DeSerializeBinary_f SerializationFunctionRegistry::GetBinaryDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_binary_map.find(type);
    if (itr != _pimpl->_binary_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeBinary_f ();
}

SerializationFunctionRegistry::SerializeXml_f SerializationFunctionRegistry::GetXmlSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_xml_map.find(type);
    if (itr != _pimpl->_xml_map.end())
    {
        return itr->second.first;
    }
    return SerializeXml_f();
}

SerializationFunctionRegistry::DeSerializeXml_f SerializationFunctionRegistry::GetXmlDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_xml_map.find(type);
    if (itr != _pimpl->_xml_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeXml_f();
}

SerializationFunctionRegistry::SerializeJson_f SerializationFunctionRegistry::GetJsonSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_json_map.find(type);
    if (itr != _pimpl->_json_map.end())
    {
        return itr->second.first;
    }
    return SerializeJson_f();
}
SerializationFunctionRegistry::DeSerializeJson_f SerializationFunctionRegistry::GetJsonDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_json_map.find(type);
    if (itr != _pimpl->_json_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeJson_f();
}

SerializationFunctionRegistry::SerializeText_f SerializationFunctionRegistry::GetTextSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_text_map.find(type);
    if (itr != _pimpl->_text_map.end())
    {
        return itr->second.first;
    }
    return SerializeText_f();
}

SerializationFunctionRegistry::DeSerializeText_f SerializationFunctionRegistry::GetTextDeSerializationFunction(const TypeInfo& type)
{
    auto itr = _pimpl->_text_map.find(type);
    if (itr != _pimpl->_text_map.end())
    {
        return itr->second.second;
    }
    return DeSerializeText_f();
}


void SerializationFunctionRegistry::SetBinarySerializationFunctions(const TypeInfo& type, SerializeBinary_f s, DeSerializeBinary_f l)
{
    _pimpl->_binary_map[type] = std::make_pair(s, l);
}

void SerializationFunctionRegistry::SetXmlSerializationFunctions(const TypeInfo& type, SerializeXml_f s, DeSerializeXml_f l)
{
    _pimpl->_xml_map[type] = std::make_pair(s, l);
}

void SerializationFunctionRegistry::SetJsonSerializationFunctions(const TypeInfo& type, SerializeJson_f serialize, DeSerializeJson_f deserialize)
{
    _pimpl->_json_map[type] = std::make_pair(serialize, deserialize);
}

void SerializationFunctionRegistry::SetTextSerializationFunctions(const TypeInfo& type, SerializeText_f serialize, DeSerializeText_f deserialize)
{
    _pimpl->_text_map[type] = std::make_pair(serialize, deserialize);
}

SerializationFunctionRegistry::SerializeBinary_f SerializationFunctionRegistry::GetSaveFunction(const TypeInfo& type, cereal::BinaryOutputArchive& ar)
{
    (void)ar;
    return GetBinarySerializationFunction(type);
}

SerializationFunctionRegistry::DeSerializeBinary_f SerializationFunctionRegistry::GetLoadFunction(const TypeInfo& type, cereal::BinaryInputArchive& ar)
{
    (void)ar;
    return GetBinaryDeSerializationFunction(type);
}

SerializationFunctionRegistry::SerializeXml_f SerializationFunctionRegistry::GetSaveFunction(const TypeInfo& type, cereal::XMLOutputArchive& ar)
{
    (void)ar;
    return GetXmlSerializationFunction(type);
}

SerializationFunctionRegistry::DeSerializeXml_f SerializationFunctionRegistry::GetLoadFunction(const TypeInfo& type, cereal::XMLInputArchive& ar)
{
    (void)ar;
    return GetXmlDeSerializationFunction(type);
}

SerializationFunctionRegistry::SerializeJson_f SerializationFunctionRegistry::GetSaveFunction(const TypeInfo& type, cereal::JSONOutputArchive &ar)
{
    (void)ar;
    return GetJsonSerializationFunction(type);
}

SerializationFunctionRegistry::DeSerializeJson_f SerializationFunctionRegistry::GetLoadFunction(const TypeInfo& type, cereal::JSONInputArchive &ar)
{
    (void)ar;
    return GetJsonDeSerializationFunction(type);
}
