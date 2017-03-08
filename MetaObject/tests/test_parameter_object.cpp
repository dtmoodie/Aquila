#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/Map.hpp"
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
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "parameter"
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

INSTANTIATE_META_PARAMETER(int);



struct output_parametered_object: public IMetaObject
{
    MO_BEGIN(output_parametered_object)
        OUTPUT(int, test_output, 0);
        OUTPUT(double, test_double, 0.0);
    MO_END;
    void increment()
    {
        test_output++;
    }
};

struct input_parametered_object: public IMetaObject
{
    MO_BEGIN(input_parametered_object)
        INPUT(int, test_input, nullptr)
    MO_END;
};

MO_REGISTER_OBJECT(input_parametered_object)
MO_REGISTER_OBJECT(output_parametered_object)

CompileLogger* logger = nullptr;
BuildCallback* cb = nullptr;
BOOST_AUTO_TEST_CASE(input_parameter_manual)
{
    logger = new CompileLogger;
    cb = new BuildCallback();
    IRuntimeObjectSystem::Instance()->Initialise(logger, nullptr);
    auto input = input_parametered_object::Create();
    auto output = output_parametered_object::Create();
    input->test_input_param.SetInput(&output->test_output_param);
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(input->test_input, &output->test_output);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);    
}

BOOST_AUTO_TEST_CASE(input_parameter_programatic)
{
    auto input = input_parametered_object::Create();
    auto input_ = input->GetParameterOptional("test_input");

    auto output = output_parametered_object::Create();
    auto output_ = output->GetParameterOptional("test_output");
    
    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParameter*>(input_);
    BOOST_REQUIRE(input_param);
    BOOST_REQUIRE(input_param->SetInput(output_));
    BOOST_REQUIRE(input->test_input);
    BOOST_REQUIRE_EQUAL(input->test_input, &output->test_output);
    BOOST_REQUIRE_EQUAL(*input->test_input, output->test_output);
}

BOOST_AUTO_TEST_CASE(buffered_input)
{
    rcc::shared_ptr<input_parametered_object> input = input_parametered_object::Create();
    auto input_ = input->GetParameterOptional("test_input");

    auto output = output_parametered_object::Create();
    auto output_ = output->GetParameterOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParameter*>(input_);
    
    auto cbuffer = Buffer::BufferFactory::CreateProxy(output_, Buffer::BufferFactory::cbuffer);
    BOOST_REQUIRE(cbuffer);
    BOOST_REQUIRE(input_param->SetInput(cbuffer));
    output->test_output_param.UpdateData(0, 0);
    for(int i = 1; i < 100000; ++i)
    {
        output->test_output_param.UpdateData(i*10, i);
        int data;
        BOOST_REQUIRE(input->test_input_param.GetData(data, i-1));
        BOOST_REQUIRE_EQUAL(data, (i-1)*10);
    }
}

BOOST_AUTO_TEST_CASE(threaded_buffered_input)
{
    auto input = input_parametered_object::Create();
    auto input_ = input->GetParameterOptional("test_input");

    auto output = output_parametered_object::Create();
    auto output_ = output->GetParameterOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParameter*>(input_);

    auto cbuffer = Buffer::BufferFactory::CreateProxy(output_, Buffer::BufferFactory::cbuffer);
    BOOST_REQUIRE(cbuffer);
    BOOST_REQUIRE(input_param->SetInput(cbuffer));
    output->test_output_param.UpdateData(0, 0);
    
    bool quit = false;
    std::thread background_thread(
    [&quit,&input]()
    {
        int ts = 0;
        int data;
        while(!quit)
        {
            if(input->test_input_param.GetData(data, ts))
            {
                BOOST_REQUIRE_EQUAL(data, ts * 10);
                ++ts;
            }
        }
    });

    for(int i = 1; i < 100000; ++i)
    {
        output->test_output_param.UpdateData(i*10, i);
    }
    quit = true;
    background_thread.join();
}

BOOST_AUTO_TEST_CASE(threaded_stream_buffer)
{
    auto input = input_parametered_object::Create();
    auto input_ = input->GetParameterOptional("test_input");

    auto output = output_parametered_object::Create();
    auto output_ = output->GetParameterOptional("test_output");

    BOOST_REQUIRE(input_);
    BOOST_REQUIRE(output_);
    auto input_param = dynamic_cast<InputParameter*>(input_);

    auto buffer = Buffer::BufferFactory::CreateProxy(output_, Buffer::BufferFactory::StreamBuffer);
    BOOST_REQUIRE(buffer);
    BOOST_REQUIRE(input_param->SetInput(buffer));
    output->test_output_param.UpdateData(0, 0);
    
    std::thread background_thread(
        [&input]()
    {
        int data;
        for(int i = 0; i < 1000; ++i)
        {
            bool good = input->test_input_param.GetData(data, i);
            while(!good)
            {
                good = input->test_input_param.GetData(data, i);
            }
            BOOST_REQUIRE_EQUAL(data, i * 10);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }
    });

    for(int i = 1; i < 1000; ++i)
    {
        output->test_output_param.UpdateData(i*10, i);
    }
    background_thread.join();
}


BOOST_AUTO_TEST_CASE(cleanup)
{
    delete IRuntimeObjectSystem::Instance();
    delete cb;
    delete logger;
}



