#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <Aquila/core/Algorithm.hpp>

#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/BufferPolicy.hpp"
#include "MetaObject/core/detail/Allocator.hpp"
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "AquilaAlgorithm"

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <iostream>
using namespace aq;


class int_output: public Algorithm
{
public:
    bool processImpl()
    {
        ++value;
        return true;
    }
    MO_BEGIN(int_output);
        OUTPUT(int, value, 0);
    MO_END;
};

class int_input: public Algorithm
{
public:
    bool processImpl()
    {
        if(input)
            value = *input;
        return true;
    }

    MO_BEGIN(int_input);
        INPUT(int, input, nullptr);
    MO_END;
    int value;
};

class synced_input: public Algorithm
{
public:
    bool processImpl()
    {
        // TODO fix
        //BOOST_REQUIRE_EQUAL(timestamp, std::chrono::duration_cast<std::chrono::seconds>(*input_param.getTimestamp()).count());
        return true;
    }
    MO_BEGIN(synced_input);
        INPUT(int, input, nullptr);
    MO_END;
    int timestamp;
};

class multi_input: public Algorithm
{
public:
    bool processImpl()
    {
        BOOST_REQUIRE_EQUAL(*input1, *input2);
        return true;
    }

    MO_BEGIN(multi_input)
        INPUT(int, input1, nullptr);
        INPUT(int, input2, nullptr);
    MO_END;
};


MO_REGISTER_OBJECT(int_output);
MO_REGISTER_OBJECT(int_input);
MO_REGISTER_OBJECT(multi_input);

BOOST_AUTO_TEST_CASE(initialize)
{
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
}


BOOST_AUTO_TEST_CASE(test_no_input)
{
    auto obj = rcc::shared_ptr<int_output>::create();
    for(int i = 0; i < 100; ++i)
    {
        obj->process();
    }
    
    BOOST_REQUIRE_EQUAL(obj->value, 100);
}

BOOST_AUTO_TEST_CASE(test_counting_input)
{
    auto output = rcc::shared_ptr<int_output>::create();
    auto input = rcc::shared_ptr<int_input>::create();
    auto output_param = output->getOutput<int>("value");
    auto input_param = input->getInput<int>("input");
    BOOST_REQUIRE(output_param);
    BOOST_REQUIRE(input_param);
    BOOST_REQUIRE(input_param->setInput(output_param));
    for(int i = 0; i < 100; ++i)
    {
        output->process();
        input->process();
        BOOST_REQUIRE_EQUAL(output->value, *input->input);
    }
}

BOOST_AUTO_TEST_CASE(test_synced_input)
{
    mo::TParam<int> output;
    output.updateData(10, 0);
    auto input = rcc::shared_ptr<int_input>::create();
    input->input_param.setInput(&output);
    input->setSyncInput("input");
    int data;
    for(int i = 0; i < 100; ++i)
    {
        output.updateData(i+ 1, i);
        BOOST_REQUIRE(input->process());
        BOOST_REQUIRE(output.getData(data, (long long)i));
        BOOST_REQUIRE_EQUAL(input->value, data);
    }
}

BOOST_AUTO_TEST_CASE(test_threaded_input)
{
    auto ctx = mo::Context::create();
    mo::TParam<int> output("test", 0);

    auto obj = rcc::shared_ptr<int_input>::create();
    boost::thread thread([&obj, &output]()->void
    {
        auto _ctx = mo::Context::create();
        obj->setContext(_ctx);
        BOOST_REQUIRE(obj->connectInput("input", nullptr, &output));
        int success_count = 0;
        while(success_count < 1000)
        {
            if(obj->process())
            {
                ++success_count;
            }
        }
        obj->setContext(nullptr, true);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    });

    for(int i = 0; i < 1000; ++i)
    {
        output.updateData(i, i, &ctx);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }

    thread.join();
}

BOOST_AUTO_TEST_CASE(test_desynced_nput)
{
    auto ctx = mo::Context::create();
    mo::TParam<int> fast_output("test", 0);
    mo::TParam<int> slow_output("test2", 0);
    int addr1, addr2;
    fast_output.getData(addr1);
    slow_output.getData(addr2);
    

    auto obj = rcc::shared_ptr<multi_input>::create();

    bool thread1_done = false;
    bool thread2_done = false;

    boost::thread thread([&obj, &fast_output, &slow_output, &thread1_done, addr1, addr2]()->void
    {
        //mo::Context _ctx.get();
        auto _ctx = mo::Context::create();
        obj->setContext(_ctx);
        BOOST_REQUIRE(obj->connectInput("input1", nullptr, &fast_output));
        BOOST_REQUIRE(obj->connectInput("input2", nullptr, &slow_output));

        int success_count = 0;
        while(success_count < 999)
        {
            if(obj->process())
            {
                ++success_count;
                if(boost::this_thread::interruption_requested())
                    break;
            }
        }
        thread1_done = true;
        obj->setContext(nullptr, true);
    });
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    boost::thread slow_thread(
        [&slow_output, &thread2_done]()->void
    {
        auto _ctx = mo::Context::create();
        slow_output.setContext(_ctx.get());
        for(int i = 1; i < 1000; ++i)
        {
            slow_output.updateData(i, i, _ctx.get());
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
            if(boost::this_thread::interruption_requested())
                break;
        }
        thread2_done = true;
        slow_output.setContext(nullptr);
    });
    

    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    for(int i = 1; i < 1000; ++i)
    {
        fast_output.updateData(i, i, &ctx);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    while(thread2_done == false)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    BOOST_REQUIRE(thread1_done);
    BOOST_REQUIRE(thread2_done);
    slow_thread.join();
    thread.join();
}

BOOST_AUTO_TEST_CASE(cleanup)
{
}
