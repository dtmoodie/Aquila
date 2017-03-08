#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"
#include "shared_ptr.hpp"
#include "obj.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "MetaObject"
#if WIN32
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif
#include <boost/filesystem.hpp>
#include <iostream>

using namespace mo;

class build_callback: public ITestBuildNotifier
{
    virtual bool TestBuildCallback(const char* file, TestBuildResult type)
    {
        std::cout << "[" << file << "] - ";
        switch(type)
        {
        case TESTBUILDRRESULT_SUCCESS:
            std::cout << "TESTBUILDRRESULT_SUCCESS\n"; break;
        case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
            std::cout << "TESTBUILDRRESULT_NO_FILES_TO_BUILD\n"; break;
        case TESTBUILDRRESULT_BUILD_FILE_GONE:
            std::cout << "TESTBUILDRRESULT_BUILD_FILE_GONE\n"; break;
        case TESTBUILDRRESULT_BUILD_NOT_STARTED:
            std::cout << "TESTBUILDRRESULT_BUILD_NOT_STARTED\n"; break;
        case TESTBUILDRRESULT_BUILD_FAILED:
            std::cout << "TESTBUILDRRESULT_BUILD_FAILED\n"; break;
        case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
            std::cout << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL\n"; break;
        }
        return true;
    }
    virtual bool TestBuildWaitAndUpdate()
    {
        return true;
    }
};


build_callback* cb = nullptr;
BOOST_AUTO_TEST_CASE(test_recompile)
{
    cb = new build_callback;
    LOG(info) << "Current working directory " << boost::filesystem::current_path().string();
	MetaObjectFactory::Instance()->RegisterTranslationUnit();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_obj_swap)
{
    auto constructor = MetaObjectFactory::Instance()->GetObjectSystem()->GetObjectFactorySystem()->GetConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto state = constructor->GetState(obj->GetPerTypeId());
    BOOST_REQUIRE(state);
    auto ptr = state->GetSharedPtr();
    rcc::shared_ptr<test_meta_object_signals> typed_ptr(ptr);
    BOOST_REQUIRE(!typed_ptr.empty());
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE(!typed_ptr.empty());
    std::vector<SignalInfo*> signals;
    typed_ptr->GetSignalInfo(signals);
    BOOST_REQUIRE_EQUAL(signals.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_pointer_mechanics)
{
    auto obj = test_meta_object_signals::Create();
    BOOST_REQUIRE(!obj.empty());
    BOOST_REQUIRE(obj.GetState());
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);
    // Test construction of weak pointers from shared pointers
    {
        rcc::weak_ptr<test_meta_object_signals> weak_ptr(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);
    
    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object;
        weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    // Test construction of weak pointers from raw object pointer
    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::weak_ptr<IMetaObject> weak_ptr_meta_object = rcc::weak_ptr<IMetaObject>(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    // Test shared pointer mechanics
    {
        rcc::shared_ptr<test_meta_object_signals> shared(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object;
        shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(obj);
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }
    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    // Test construction of weak pointers from raw object pointer
    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

    {
        rcc::shared_ptr<IMetaObject> shared_ptr_meta_object = rcc::shared_ptr<IMetaObject>(obj.Get());
        BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 2);
        BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 2);
    }

    BOOST_REQUIRE_EQUAL(obj.GetState()->ObjectCount(), 1);
    BOOST_REQUIRE_EQUAL(obj.GetState()->StateCount(), 1);

}

BOOST_AUTO_TEST_CASE(test_creation_function)
{
    auto obj = test_meta_object_signals::Create();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
}

BOOST_AUTO_TEST_CASE(test_reconnect_signals)
{
    auto signals = test_meta_object_signals::Create();
    auto slots = test_meta_object_slots::Create();
    auto state = signals->GetConstructor()->GetState(signals->GetPerTypeId());
    IMetaObject::Connect(signals.Get(), "test_int", slots.Get(), "test_int");
    int value  = 5;
    signals->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(slots->call_count, value);
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    
    signals->sig_test_int(value);
    BOOST_REQUIRE_EQUAL(slots->call_count, 10);
}

BOOST_AUTO_TEST_CASE(test_input_output_parameter)
{
	auto output = rcc::shared_ptr<test_meta_object_output>::Create();
	auto input = rcc::shared_ptr<test_meta_object_input>::Create();
	auto output_param = output->GetOutput("test_output");
	BOOST_REQUIRE(output_param);
	auto input_param = input->GetInput("test_input");
	BOOST_REQUIRE(input_param);
	
	BOOST_REQUIRE(IMetaObject::ConnectInput(output.Get(), output_param, input.Get(), input_param));
	output->test_output = 5;
	BOOST_REQUIRE(input->test_input);
	BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
	BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
	output->test_output = 10;
	BOOST_REQUIRE(input->test_input);
	BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
	BOOST_REQUIRE_EQUAL(*input->test_input, 10);
}

BOOST_AUTO_TEST_CASE(test_parameter_persistence_recompile)
{
    auto obj = test_meta_object_parameters::Create();
    BOOST_REQUIRE_EQUAL(obj->test, 5);
    obj->test = 10;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj->test, 10);
}



BOOST_AUTO_TEST_CASE(test_multiple_objects)
{
    auto obj1 = rcc::shared_ptr<test_meta_object_parameters>::Create();
    auto obj2 = rcc::shared_ptr<test_meta_object_parameters>::Create();
    obj1->test = 1;
    obj2->test = 2;
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    BOOST_REQUIRE_EQUAL(obj1->test, 1);
    BOOST_REQUIRE_EQUAL(obj2->test, 2);
}

#ifdef HAVE_CUDA
BOOST_AUTO_TEST_CASE(test_cuda_recompile)
{
    auto obj = rcc::shared_ptr<test_cuda_object>::Create();
    obj->run_kernel();
    BOOST_REQUIRE_EQUAL(MetaObjectFactory::Instance()->GetObjectSystem()->TestBuildAllRuntimeSourceFiles(cb, true), 0);
    obj->run_kernel();
}
#endif



BOOST_AUTO_TEST_CASE(test_object_cleanup)
{
    AUDynArray<IObjectConstructor*> constructors;
    MetaObjectFactory::Instance()->GetObjectSystem()->GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        BOOST_REQUIRE_EQUAL(constructors[i]->GetNumberConstructedObjects(), 0);
    }
    delete cb;
}
