
#define BOOST_TEST_MAIN
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/RelayManager.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/POD.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"

#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObject"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

struct test_meta_obj_empty: public IMetaObject
{
    MO_BEGIN(test_meta_obj_empty);

    MO_END;
};

struct test_meta_obj_params: public IMetaObject
{

};

struct test_meta_object_signals: public IMetaObject
{
    MO_BEGIN(test_meta_object_signals);
	MO_SIGNAL(void, test_void);
	MO_SIGNAL(void, test_int, int);
    MO_END;
};

struct test_meta_object_slots: public IMetaObject
{
    MO_BEGIN(test_meta_object_slots);
        MO_SLOT(void, test_void);
        MO_SLOT(void, test_void, int);
    MO_END;
	int slot_called = 0;
};
void test_meta_object_slots::test_void()
{
    std::cout << "test_void called\n";
	++slot_called;
}
void test_meta_object_slots::test_void(int)
{
    
}

struct test_meta_object_callback: public IMetaObject
{
    MO_BEGIN(test_meta_object_callback);
	    MO_SLOT(int, test_int);
	    MO_SLOT(void, test_void);
    MO_END;
    
};
int test_meta_object_callback::test_int()
{
	return 5;
}
void test_meta_object_callback::test_void()
{

}


struct test_meta_object_parameter: public IMetaObject
{
    MO_BEGIN(test_meta_object_parameter);
        PARAM(int, test_int, 5);
        TOOLTIP(test_int, "test tooltip")
    MO_END;
};

struct test_meta_object_input: public IMetaObject
{
    MO_BEGIN(test_meta_object_input);
        INPUT(int, test_int, nullptr);
    MO_END;
};



MO_REGISTER_OBJECT(test_meta_object_signals)
MO_REGISTER_OBJECT(test_meta_object_slots)
MO_REGISTER_OBJECT(test_meta_object_callback)
MO_REGISTER_OBJECT(test_meta_object_parameter)
MO_REGISTER_OBJECT(test_meta_object_input)

//RuntimeObjectSystem obj_sys;

BOOST_AUTO_TEST_CASE(test_meta_object_static_introspection_global)
{
    MetaObjectFactory::Instance()->RegisterTranslationUnit();
	auto info = MetaObjectInfoDatabase::Instance()->GetMetaObjectInfo();
	BOOST_REQUIRE(info.size());
	for (auto& item : info)
	{
		std::cout << item->Print() << std::endl;
	}
}

BOOST_AUTO_TEST_CASE(test_meta_object_static_introspection_specific)
{
	auto info = MetaObjectInfoDatabase::Instance()->GetMetaObjectInfo("test_meta_object_signals");
	BOOST_REQUIRE(info);
	std::cout << info->Print() << std::endl;
}


BOOST_AUTO_TEST_CASE(test_meta_object_dynamic_introspection)
{
	RelayManager mgr;
	auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_signals");
	auto obj = constructor->Construct();
    auto state = constructor->GetState(obj->GetPerTypeId());
    auto weak_ptr = state->GetWeakPtr();
    {
        auto ptr = state->GetSharedPtr();
        rcc::shared_ptr<IMetaObject> meta_obj(ptr);
        meta_obj->SetupSignals(&mgr);
        meta_obj->Init(true);
        auto signal_info = meta_obj->GetSignalInfo();
        BOOST_REQUIRE(!weak_ptr.empty());
        BOOST_REQUIRE_EQUAL(signal_info.size(), 2);

        auto signals_= meta_obj->GetSignals();
        BOOST_REQUIRE_EQUAL(signals_.size(), 4);
    }
    BOOST_REQUIRE(weak_ptr.empty());
    BOOST_REQUIRE_EQUAL(constructor->GetNumberConstructedObjects(), 0);
}

BOOST_AUTO_TEST_CASE(test_meta_object_dynamic_access)
{
	RelayManager mgr;
	auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_signals");
	auto obj = constructor->Construct();
	auto meta_obj = static_cast<IMetaObject*>(obj);
    meta_obj->Init(true);
	meta_obj->SetupSignals(&mgr);
	
	auto signals_ = meta_obj->GetSignals();
	BOOST_REQUIRE_EQUAL(signals_.size(), 4);
    int input_parameter = 0;
    int call_value = 5;
    test_meta_object_signals* typed = dynamic_cast<test_meta_object_signals*>(meta_obj);
    std::shared_ptr<mo::ISlot> slot(new mo::TypedSlot<void(int)>(
        std::bind([&input_parameter](int value)
    {
        input_parameter += value;
    }, std::placeholders::_1)));
    auto connection = mgr.Connect(slot.get(), "test_int");
    typed->sig_test_int(call_value);
    BOOST_REQUIRE_EQUAL(input_parameter, 5);
    typed->sig_test_int(call_value);
    BOOST_REQUIRE_EQUAL(input_parameter, 10);
	delete obj;
}

BOOST_AUTO_TEST_CASE(test_meta_object_external_slot)
{
	RelayManager mgr;
	auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_signals");
	auto obj = constructor->Construct();
	auto meta_obj = static_cast<test_meta_object_signals*>(obj);
	meta_obj->SetupSignals(&mgr);
	meta_obj->Init(true);
	bool slot_called = false;
	TypedSlot<void(int)> int_slot([&slot_called](int value)
	{
		slot_called = value == 5;
	});
	BOOST_REQUIRE(meta_obj->ConnectByName("test_int", &int_slot));
	int desired_value = 5;
	meta_obj->sig_test_int(desired_value);
	BOOST_REQUIRE(slot_called);
	delete obj;
}


BOOST_AUTO_TEST_CASE(test_meta_object_internal_slot)
{
	RelayManager mgr;
	auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_slots");
	auto obj = constructor->Construct();
	auto meta_obj = static_cast<test_meta_object_slots*>(obj);
    auto slot = meta_obj->GetSlot_test_void<void()>();
    auto overload = meta_obj->GetSlot_test_void<void(int)>();
	meta_obj->Init(true);
	meta_obj->SetupSignals(&mgr);
	TypedSignal<void(void)> signal;
	BOOST_REQUIRE(meta_obj->ConnectByName("test_void", &signal));
	signal();
	BOOST_REQUIRE_EQUAL(meta_obj->slot_called, 1);
	signal();
	BOOST_REQUIRE_EQUAL(meta_obj->slot_called, 2);
	delete obj;
}
BOOST_AUTO_TEST_CASE(inter_object_typed)
{
	RelayManager mgr;
	auto constructor =MetaObjectFactory::Instance()->GetConstructor("test_meta_object_signals");
	auto obj = constructor->Construct();
	auto signal_object = static_cast<test_meta_object_signals*>(obj);
	signal_object->SetupSignals(&mgr);
	signal_object->Init(true);
	constructor =MetaObjectFactory::Instance()->GetConstructor("test_meta_object_slots");
	obj = constructor->Construct();
	auto slot_object = static_cast<test_meta_object_slots*>(obj);
	slot_object->SetupSignals(&mgr);
	slot_object->Init(true);

	BOOST_REQUIRE(IMetaObject::Connect(signal_object, "test_void", slot_object, "test_void", TypeInfo(typeid(void(void)))));
	signal_object->sig_test_void();
	BOOST_REQUIRE_EQUAL(slot_object->slot_called, 1);
    delete obj;
    delete signal_object;
}

BOOST_AUTO_TEST_CASE(inter_object_named)
{
	RelayManager mgr;
	auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_signals");
	auto obj = constructor->Construct();
	auto signal_object = static_cast<test_meta_object_signals*>(obj);
	signal_object->SetupSignals(&mgr);
	signal_object->Init(true);
	constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_slots");
	obj = constructor->Construct();
	auto slot_object = static_cast<test_meta_object_slots*>(obj);
	slot_object->SetupSignals(&mgr);
	slot_object->Init(true);

	BOOST_REQUIRE_EQUAL(IMetaObject::Connect(signal_object, "test_void", slot_object, "test_void"), 1);
	signal_object->sig_test_void();
	BOOST_REQUIRE_EQUAL(slot_object->slot_called, 1);
    delete obj;
    delete signal_object;
}

BOOST_AUTO_TEST_CASE(rest)
{
	RelayManager mgr;
    {
        auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_callback");
        auto obj = constructor->Construct();
        test_meta_object_callback* meta_obj = static_cast<test_meta_object_callback*>(obj);
        meta_obj->Init(true);
        meta_obj->SetupSignals(&mgr);
		TypedSignal<int(void)> signal;
		auto slot = meta_obj->GetSlot("test_int", TypeInfo(typeid(int(void))));
		BOOST_REQUIRE(slot);
		auto connection = slot->Connect(&signal);
		BOOST_REQUIRE(connection);
		BOOST_REQUIRE_EQUAL(signal(), 5);
        delete obj;
    }
    {
        auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_callback");
        auto obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_callback* cb = static_cast<test_meta_object_callback*>(obj);
        constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_slots");
        obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_slots* slot = static_cast<test_meta_object_slots*>(obj);
        cb->test_void();
        delete cb;
        delete slot;
    }
    {
        auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_parameter");
        auto obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_parameter* ptr = static_cast<test_meta_object_parameter*>(obj);
    }
}


BOOST_AUTO_TEST_CASE(test_parameters)
{
	RelayManager mgr;
    auto constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_parameter");
    auto obj = constructor->Construct();
    obj->Init(true);
    test_meta_object_parameter* ptr = static_cast<test_meta_object_parameter*>(obj);
	ptr->SetupSignals(&mgr);
    constructor = MetaObjectFactory::Instance()->GetConstructor("test_meta_object_input");
    obj = constructor->Construct();
    obj->Init(true);
	test_meta_object_input* input = static_cast<test_meta_object_input*>(obj);
    input->SetupSignals(&mgr);
    delete obj;
}
