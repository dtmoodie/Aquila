#define BOOST_TEST_MAIN
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"

#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/ParameterInfo.hpp"

#include "MetaObject/MetaObjectFactory.hpp"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObjectInheritance"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

struct base: public IMetaObject
{
    
    MO_BEGIN(base)
        PARAM(int, base_param, 5);
        MO_SIGNAL(void, base_signal, int);
        MO_SLOT(void, base_slot, int);
        MO_SLOT(void, override_slot, int);
    MO_END;
    int base_count = 0;
};

struct derived_parameter: virtual public base
{
    MO_DERIVE(derived_parameter, base);
        PARAM(int, derived_param, 10);
    MO_END;
};

struct derived_signals: virtual public base
{
    static std::string GetDescriptionStatic()
    {
        return "test description";
    }
    static std::string GetTooltipStatic()
    {
        return "test tooltip";
    }

    MO_DERIVE(derived_signals, base);
        MO_SIGNAL(void, derived_signal, int);
        MO_SLOT(void, derived_slot, int);
    MO_END;

    void override_slot(int value);
    int derived_count = 0;
};
struct multi_derive: virtual public derived_parameter, virtual public derived_signals
{
    MO_DERIVE(multi_derive, derived_parameter, derived_signals)

    MO_END;
};

void base::base_slot(int value)
{
    base_count += value;
}
void base::override_slot(int value)
{
    base_count += value*2;
}
void derived_signals::derived_slot(int value)
{
    derived_count += value;
}
void derived_signals::override_slot(int value)
{
    derived_count += 3*value;
}

struct base1: public TInterface<0, IMetaObject>
{
    MO_BEGIN(base1);
    MO_END;
};

struct derived1: public TInterface<1, base1>
{
    MO_DERIVE(derived1, base1);
    MO_END;
};

MO_REGISTER_OBJECT(derived_signals);
MO_REGISTER_OBJECT(derived_parameter);
MO_REGISTER_OBJECT(derived1);
MO_REGISTER_OBJECT(multi_derive);

BOOST_AUTO_TEST_CASE(initialize)
{
    mo::MetaObjectFactory::Instance();
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
}

BOOST_AUTO_TEST_CASE(object_print)
{
    auto info = mo::MetaObjectFactory::Instance()->GetObjectInfo("derived_signals");
    info->Print();
}

BOOST_AUTO_TEST_CASE(parameter_static)
{
    auto param_info = derived_parameter::GetParameterInfoStatic();
    if(param_info.size() == 1)
    {
        if(param_info[0]->name == "derived_param")
        {
            std::cout << "missing base parameter \"base_param\"\n";
        }else
        {
            std::cout << "missing derived parameter \"derived_param\"\n";
        }
    }
    BOOST_REQUIRE_EQUAL(param_info.size(), 2);
}

BOOST_AUTO_TEST_CASE(signals_static)
{
    auto signal_info = derived_signals::GetSignalInfoStatic();
    BOOST_REQUIRE_EQUAL(signal_info.size(), 2);
}

BOOST_AUTO_TEST_CASE(slots_static)
{
    auto slot_info = derived_signals::GetSlotInfoStatic();
    BOOST_REQUIRE_EQUAL(slot_info.size(), 3);
}

BOOST_AUTO_TEST_CASE(parameter_dynamic)
{
    auto derived_obj = derived_parameter::Create();
    BOOST_REQUIRE_EQUAL(derived_obj->base_param, 5);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_param, 10);
    derived_obj->base_param = 10;
    derived_obj->derived_param = 100;
    derived_obj->InitParameters(true);
    BOOST_REQUIRE_EQUAL(derived_obj->base_param, 5);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_param, 10);
}


BOOST_AUTO_TEST_CASE(call_base_slot)
{
    auto derived_obj = derived_signals::Create();
    TypedSignal<void(int)> sig;
    derived_obj->ConnectByName("base_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->base_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->base_count, 100);
}

BOOST_AUTO_TEST_CASE(call_derived_slot)
{
    auto derived_obj = derived_signals::Create();
    TypedSignal<void(int)> sig;
    derived_obj->ConnectByName("derived_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 100);
}

BOOST_AUTO_TEST_CASE(call_overloaded_slot)
{
    auto derived_obj = derived_signals::Create();
    TypedSignal<void(int)> sig;
    derived_obj->ConnectByName("override_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 300);
}

BOOST_AUTO_TEST_CASE(interface_id_check)
{
    auto constructor = mo::MetaObjectFactory::Instance()->GetConstructor("derived1");
    BOOST_REQUIRE(constructor);
    BOOST_REQUIRE_EQUAL(constructor->GetInterfaceId(), 1);
}

BOOST_AUTO_TEST_CASE(diamond)
{
    //auto obj = rcc::shared_ptr<multi_derive>::Create();
    auto constructor = mo::MetaObjectFactory::Instance()->GetConstructor("multi_derive");
    BOOST_REQUIRE(constructor);
    auto info = constructor->GetObjectInfo();
    std::cout << info->Print();
    //auto meta_info = dynamic_cast<MetaObjectInfo*>(info);
    //BOOST_REQUIRE(meta_info);

}