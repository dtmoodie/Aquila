#include "MetaObject/serialization/CerealPolicy.hpp"
#include <Aquila/serialization/cereal//serialize.hpp>
#include <Aquila/serialization/cereal/memory.hpp>
#include <Aquila/nodes/Node.hpp>

#include "MetaObject/serialization/Serializer.hpp"
#include "MetaObject/serialization/SerializationFactory.hpp"
#include "MetaObject/serialization/Policy.hpp"
#include "MetaObject/params/traits/MemoryTraits.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

using namespace aq;
using namespace aq::Nodes;

bool aq::Serialize(cereal::BinaryOutputArchive& ar, const Node* obj)
{
    if (auto func = mo::SerializerFactory::GetSerializationFunctionBinary(obj->GetTypeName()))
    {
        func(obj, ar);
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params)
        {
            auto func1 = mo::SerializationFactory::instance()->getBinarySerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
    }
    return true;
}

bool aq::DeSerialize(cereal::BinaryInputArchive& ar, Node* obj)
{
    return false;
}

bool aq::Serialize(cereal::XMLOutputArchive& ar, const Node* obj)
{
    if (auto func = mo::SerializerFactory::getSerializationFunctionXML(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params)
        {
            auto func1 = mo::SerializationFactory::instance()->getXmlSerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
        return true;
    }
}

bool aq::DeSerialize(cereal::XMLInputArchive& ar, Node* obj)
{
    return false;
}

bool aq::Serialize(cereal::JSONOutputArchive& ar, const Node* obj)
{
    /*if (auto func = mo::SerializerFactory::GetSerializationFunctionJSON(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        std::string type = obj->GetTypeName();
        ObjectId id = obj->GetObjectId();
        ar(cereal::make_nvp("TypeId", id.m_ConstructorId));
        ar(cereal::make_nvp("InstanceId", id.m_PerTypeId));
        ar(cereal::make_nvp("TypeName", type));
        for (auto& param : params)
        {
            if (param->checkFlags(mo::Input_e))
            {
                mo::InputParam* input = dynamic_cast<mo::InputParam*>(param);
                if (input)
                {
                    auto input_source_param = input->getInputParam();
                    if (input_source_param)
                    {
                        std::string input_source = input_source_param->getTreeName();
                        std::string param_name = param->getName();
                        ar(cereal::make_nvp(param_name, input_source));
                        continue;
                    }else
                    {
                        std::string blank;
                        std::string param_name = param->getName();
                        ar(cereal::make_nvp(param_name, blank));
                        continue;
                    }
                }
            }
            if (param->checkFlags(mo::Output_e))
                continue;
            auto func1 = mo::SerializationFactory::instance()->getJsonSerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
        }
        return true;
    }*/
    return true;
}

bool aq::DeSerialize(cereal::JSONInputArchive& ar, Node* obj)
{
    // TODO reimplment
    /*if (obj == nullptr)
        return false;
    if (auto func = mo::SerializerFactory::GetDeSerializationFunctionJSON(obj->GetTypeName()))
    {
        func(obj, ar);
        return true;
    }
    else
    {
        LOG(debug) << "No object specific serialization function found for " << obj->GetTypeName();
        auto params = obj->getParams();
        for (auto& param : params)
        {
            if (param->checkFlags(mo::Input_e))
                continue;
            if (param->checkFlags(mo::Output_e))
                continue;
            auto func1 = mo::SerializationFactory::instance()->getJsonDeSerializationFunction(param->getTypeInfo());
            if (func1)
            {
                if (!func1(param, ar))
                {
                    LOG(debug) << "Unable to serialize " << param->getTreeName();
                }
            }
            else
            {
                LOG(debug) << "No serialization function found for " << param->getTypeInfo().name();
            }
            if (param->getName() == "_dataStream")
            {
                auto typed = dynamic_cast<mo::ITParam<rcc::weak_ptr<IDataStream>>*>(param);
                if (typed)
                {
                    mo::ParamTraits<rcc::weak_ptr<IDataStream>>::InputStorage_t data;
                    if(typed->getData(data)){
                        obj->setDataStream(data.get());
                    }
                }
            }
        }
        obj->setParamRoot(obj->getTreeName());
        for (auto& param : params)
        {
            if (param->checkFlags(mo::Input_e))
            {
                mo::InputParam* input = dynamic_cast<mo::InputParam*>(param);
                if (input)
                {
                    std::string input_source;
                    std::string param_name = param->getName();
                    try
                    {
                        ar(cereal::make_nvp(param_name, input_source));
                    }
                    catch (cereal::Exception& e)
                    {
                        continue;
                    }
                    if (input_source.size())
                    {
                        auto token_index = input_source.find(':');
                        if (token_index != std::string::npos)
                        {
                            auto stream = obj->getDataStream();
                            if(stream)
                            {
                                auto output_node = stream->getNode(input_source.substr(0, token_index));
                                if (output_node)
                                {
                                    auto output_param = output_node->getOutput(input_source.substr(token_index + 1));
                                    if (output_param)
                                    {
                                        //obj->connectInput(output_node, output_param, input, mo::BlockingStreamBuffer_e);
                                        obj->IMetaObject::connectInput(input, output_node, output_param, mo::BlockingStreamBuffer_e);
                                        obj->setDataStream(output_node->getDataStream());
                                        obj->setContext(output_node->getContext());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }*/
    return true;
}
