#define BOOST_TEST_MAIN
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <MetaObject/Parameters/IO/TextPolicy.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/Parameters/Buffers/BufferFactory.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
#include "MetaObject/IO/Policy.hpp"
#include "MetaObject/IO/memory.hpp"
#include "shared_ptr.hpp"
#include "RuntimeObjectSystem.h"
#include "shared_ptr.hpp"
#include "IObjectFactorySystem.h"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/portable_binary.hpp"
#include <fstream>
#include <istream>
#include "instantiate.hpp"
#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#endif
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

struct serializable_object: public IMetaObject
{
    MO_BEGIN(serializable_object);
    PARAM(int, test, 5);
    PARAM(int, test2, 6);
    MO_END;
};



BuildCallback* cb = nullptr;
MO_REGISTER_OBJECT(serializable_object);



BOOST_AUTO_TEST_CASE(test_serialization)
{
    BOOST_REQUIRE(mo::IO::Text::imp::stream_serializable<size_t>::value);
    BOOST_REQUIRE(!mo::IO::Text::imp::stream_serializable<rcc::shared_ptr<serializable_object>>::value);
}


BOOST_AUTO_TEST_CASE(serialize_manual_xml)
{
    cb = new BuildCallback();
    mo::instantiations::initialize();
    MetaObjectFactory::Instance()->GetObjectSystem()->SetupObjectConstructors(PerModuleInterface::GetInstance());
    rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
    {
        std::ofstream ofs("test.xml");
        cereal::XMLOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.Get()));
    }
    {
        std::ifstream ifs("test.xml");
        cereal::XMLInputArchive inar(ifs);
        rcc::shared_ptr<serializable_object> obj2 = serializable_object::Create();
        inar(*(obj2.Get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_manual_binary)
{
    rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
    {
        std::ofstream ofs("test.bin");
        cereal::BinaryOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.Get()));
    }
    {
        std::ifstream ifs("test.bin");
        cereal::BinaryInputArchive inar(ifs);
        rcc::shared_ptr<serializable_object> obj2 = serializable_object::Create();
        inar(*(obj2.Get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_by_policy_xml)
{
    rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
	obj->test = 14;
	obj->test2 = 13;
    {
        std::ofstream ofs("test2.xml");
        SerializerFactory::Serialize(obj, ofs, SerializerFactory::xml_e);
    }
    {
        std::ifstream ifs("test2.xml");
        SerializerFactory::DeSerialize(obj.Get(), ifs, SerializerFactory::xml_e);
    }
}

BOOST_AUTO_TEST_CASE(serialize_by_policy_binary)
{
	rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
	obj->test = 14;
	obj->test2 = 13;
	{
		std::ofstream ofs("test2.bin", std::ios::binary);
        SerializerFactory::Serialize(obj, ofs, SerializerFactory::Binary_e);
	}
	{
		std::ifstream ifs("test2.bin", std::ios::binary);
        SerializerFactory::DeSerialize(obj.Get(), ifs, SerializerFactory::Binary_e);
		BOOST_REQUIRE_EQUAL(obj->test, 14);
		BOOST_REQUIRE_EQUAL(obj->test2, 13);
	}
}

BOOST_AUTO_TEST_CASE(deserialize_to_new_object_xml)
{
    std::ifstream ifs("test2.xml");
    auto obj = SerializerFactory::DeSerialize(ifs, SerializerFactory::xml_e);
    BOOST_REQUIRE(obj);
	auto typed = obj.DynamicCast<serializable_object>();
	BOOST_REQUIRE(typed);
	BOOST_REQUIRE_EQUAL(typed->test, 14);
	BOOST_REQUIRE_EQUAL(typed->test2, 13);
}

BOOST_AUTO_TEST_CASE(deserialize_to_new_object_binary)
{
	std::ifstream ifs("test2.bin", std::ios::binary);
    auto obj = SerializerFactory::DeSerialize(ifs, SerializerFactory::Binary_e);
	BOOST_REQUIRE(obj);
	auto typed = obj.DynamicCast<serializable_object>();
	BOOST_REQUIRE(typed);
    BOOST_REQUIRE_EQUAL(typed->test, 14);
	BOOST_REQUIRE_EQUAL(typed->test2, 13);
}

BOOST_AUTO_TEST_CASE(serialize_multi_by_policy_binary)
{
	rcc::shared_ptr<serializable_object> obj1 = serializable_object::Create();
	rcc::shared_ptr<serializable_object> obj2 = serializable_object::Create();
	obj1->test = 14;
	obj1->test2 = 13;
	obj2->test = 15;
	obj2->test2 = 16;

	{
		std::ofstream ofs("test2.bin", std::ios::binary);
        SerializerFactory::Serialize(obj1.Get(), ofs, SerializerFactory::Binary_e);
        SerializerFactory::Serialize(obj2.Get(), ofs, SerializerFactory::Binary_e);
	}
	{
		std::ifstream ifs("test2.bin", std::ios::binary);
        auto new_obj1 = rcc::shared_ptr<serializable_object>(SerializerFactory::DeSerialize(ifs, SerializerFactory::Binary_e));
        auto new_obj2 = rcc::shared_ptr<serializable_object>(SerializerFactory::DeSerialize(ifs, SerializerFactory::Binary_e));
		BOOST_REQUIRE_EQUAL(new_obj1->test, 14);
		BOOST_REQUIRE_EQUAL(new_obj1->test2, 13);
		BOOST_REQUIRE_EQUAL(new_obj2->test, 15);
		BOOST_REQUIRE_EQUAL(new_obj2->test2, 16);
	}
}


INSTANTIATE_META_PARAMETER(mo::ReadFile);
INSTANTIATE_META_PARAMETER(std::vector<int>);
BOOST_AUTO_TEST_CASE(deserialize_text_path)
{
    mo::ReadFile data;
    mo::TypedParameterPtr<mo::ReadFile> param;
    param.UpdatePtr(&data);
    auto deserialization_function = mo::SerializationFunctionRegistry::Instance()->GetTextDeSerializationFunction(param.GetTypeInfo());
    BOOST_REQUIRE(deserialization_function);
    std::stringstream ss;
    ss << "/asdf/asdf/asdf/test.txt";
    deserialization_function(&param, ss);
    BOOST_REQUIRE(data == "/asdf/asdf/asdf/test.txt");
}

BOOST_AUTO_TEST_CASE(serialize_text_vector)
{
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7};
    mo::TypedParameterPtr<std::vector<int>> param;
    param.UpdatePtr(&data);
    auto serialization_function = mo::SerializationFunctionRegistry::Instance()->GetTextSerializationFunction(param.GetTypeInfo());
    BOOST_REQUIRE(serialization_function);
    std::stringstream ss;
    serialization_function(&param,ss);
    std::string str = ss.str();
    BOOST_REQUIRE(str == "8[0, 1, 2, 3, 4, 5, 6, 7]");
}


BOOST_AUTO_TEST_CASE(deserialize_text_vector)
{
    std::vector<int> data;
    mo::TypedParameterPtr<std::vector<int>> param;
    param.UpdatePtr(&data);
    auto deserialization_function = mo::SerializationFunctionRegistry::Instance()->GetTextDeSerializationFunction(param.GetTypeInfo());
    BOOST_REQUIRE(deserialization_function);
    std::stringstream ss;
    ss << "[0, 1, 2, 3, 4, 5, 6, 7]";
    deserialization_function(&param, ss);
    BOOST_REQUIRE(data.size() == 8);
    for(int expected_value = 0; expected_value < 8; ++expected_value)
    {
        BOOST_REQUIRE_EQUAL(data[expected_value], expected_value);
    }
}

BOOST_AUTO_TEST_CASE(serialize_unique_instance)
{
    {
        auto inst1 = serializable_object::Create();
        auto inst2 = inst1;
        mo::StartSerialization();
        std::ofstream ofs("test_unique_instance.json");
        cereal::JSONOutputArchive ar(ofs);
        ar(CEREAL_NVP(inst1));
        ar(CEREAL_NVP(inst2));
        mo::EndSerialization();
    }
    {
        mo::StartSerialization();
        std::ifstream ifs("test_unique_instance.json");
        cereal::JSONInputArchive ar(ifs);
        rcc::shared_ptr<serializable_object> inst1;
        rcc::shared_ptr<serializable_object> inst2;
        ar(CEREAL_NVP(inst1));
        ar(CEREAL_NVP(inst2));
        mo::EndSerialization();
        BOOST_REQUIRE_EQUAL(inst1, inst2);
    }
}


BOOST_AUTO_TEST_CASE(cleanup)
{
    delete cb;
}
