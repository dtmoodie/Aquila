#include <Aquila/core/Algorithm.hpp>

#include "MetaObject/core/detail/Allocator.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"
#include <MetaObject/core/AsyncStreamFactory.hpp>

#include <boost/fiber/operations.hpp>
#include <boost/thread.hpp>

#include "gtest/gtest.h"

#include <iostream>
using namespace aq;

int main(int argc, char** argv)
{
    std::shared_ptr<SystemTable> table;
    bool make_table = true;
    for (int i = 0; i < argc; ++i)
    {
        if (std::string("--gtest_list_tests") == argv[i])
        {
            make_table = false;
        }
    }
    ::testing::InitGoogleTest(&argc, argv);
    if (make_table)
    {
        table = SystemTable::instance();
        mo::MetaObjectFactory::instance()->registerTranslationUnit();
    }

    return RUN_ALL_TESTS();
}

class int_output : public Algorithm
{
  public:
    bool processImpl() override
    {
        value.publish(++counter);
        return true;
    }
    MO_BEGIN(int_output)
        OUTPUT(int, value, 0)
    MO_END;
    int counter = 0;
};

class int_input : public Algorithm
{
  public:
    bool processImpl() override
    {
        if (input)
        {
            value = *input;
        }
        return true;
    }

    MO_BEGIN(int_input)
        INPUT(int, input)
    MO_END;
    int value;
};

class synced_input : public Algorithm
{
  public:
    bool processImpl() override
    {
        // TODO fix
        // EXPECT_EQ(timestamp,
        // std::chrono::duration_cast<std::chrono::seconds>(*input_param.getTimestamp()).count());
        return true;
    }
    MO_BEGIN(synced_input)
        INPUT(int, input);
    MO_END;
    int timestamp;
};

class multi_input : public Algorithm
{
  public:
    bool processImpl() override
    {
        EXPECT_EQ(*input1, *input2);
        EXPECT_NE(last, *input1);
        last = *input1;
        first.push_back(*input1);
        second.push_back(*input2);
        return true;
    }

    MO_BEGIN(multi_input)
        INPUT(int, input1)
        INPUT(int, input2)
    MO_END;
    int last = -1;
    std::vector<int> first;
    std::vector<int> second;
};

MO_REGISTER_OBJECT(int_output);
MO_REGISTER_OBJECT(int_input);
MO_REGISTER_OBJECT(multi_input);

TEST(algorithm, reflection)
{
    ct::Reflect<aq::Algorithm>::printHierarchy(std::cout);
    ct::printStructInfo<aq::Algorithm>(std::cout);
}

TEST(algorithm, no_input)
{
    auto stream = mo::AsyncStreamFactory::instance()->create();
    auto obj = rcc::shared_ptr<int_output>::create();
    obj->setStream(stream);
    size_t success_count = 0;
    for (size_t i = 0; i < 100; ++i)
    {
        if(obj->process())
        {
            ++success_count;
        }
    }
    EXPECT_EQ(success_count, 100);
    EXPECT_EQ(obj->counter, 100);
}

TEST(algorithm, counting_input)
{
    auto stream = mo::AsyncStreamFactory::instance()->create();
    auto output = rcc::shared_ptr<int_output>::create();
    auto input = rcc::shared_ptr<int_input>::create();
    output->setStream(stream);
    input->setStream(stream);
    auto output_param = output->getOutput("value");
    auto input_param = input->getInput("input");
    EXPECT_NE(output_param, nullptr);
    EXPECT_NE(input_param, nullptr);
    EXPECT_EQ(input_param->setInput(output_param), true);
    for (int i = 0; i < 100; ++i)
    {
        output->process();
        input->process();
        EXPECT_EQ(output->counter, *input->input);
    }
}

TEST(algorithm, synced_input)
{
    auto ctx = mo::AsyncStreamFactory::instance()->create();
    mo::TPublisher<int> output;
    output.publish(10, mo::Header(0));
    auto input = rcc::shared_ptr<int_input>::create();
    input->input_param.setInput(&output);
    input->setSyncInput("input");
    for (int i = 0; i < 100; ++i)
    {
        output.publish(i + 1, i);
        EXPECT_EQ(input->process(), true);

        auto header = mo::Header(mo::FrameNumber(i));
        auto data = output.getData(&header, ctx.get());

        EXPECT_TRUE(data) << "Unable to retrieve data at fn=" << i + 1 << " what does exist is "
                          << output.getAvailableHeaders();
        auto ptr = data->ptr<int>();
        ASSERT_TRUE(ptr);
        EXPECT_EQ(input->value, *ptr) << "retrieved value does not match what we published";
        EXPECT_EQ(*ptr, i + 1);
        EXPECT_EQ(*input->input, input->value);
    }
}

TEST(algorithm, threaded_input)
{
    auto ctx = mo::AsyncStreamFactory::instance()->create();
    mo::TPublisher<int> output;
    output.setName("test");
    output.setStream(*ctx);
    auto obj = rcc::shared_ptr<int_input>::create();
    std::atomic<int> success_count(0);
    boost::thread thread([&obj, &output, &success_count]() -> void {
        auto thread_stream = mo::AsyncStreamFactory::instance()->create();
        obj->setStream(thread_stream);
        ASSERT_TRUE(obj->connectInput(&obj->input_param, nullptr, &output));
        auto pub = obj->input_param.getPublisher();
        ASSERT_TRUE(pub);
        ASSERT_TRUE(pub->checkFlags(mo::ParamFlags::kBUFFER)) << "Failed to setup a buffered connection";

        while (success_count < 1000 && !boost::this_thread::interruption_requested())
        {
            if (obj->process())
            {
                ++success_count;
            }
        }
        obj->setStream(nullptr);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    });

    for (int i = 0; i < 1000; ++i)
    {
        output.publish(i, mo::Header(std::chrono::milliseconds(i)));
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }

    const bool finished = thread.try_join_for(boost::chrono::seconds(5));
    EXPECT_TRUE(finished) << "expected to perform 1000 iterations, but instead performed only " << success_count;
    if (!finished)
    {
        thread.interrupt();
        thread.join();
    }

    // mo::getDefaultLogger().set_level(spdlog::level::info);
}

TEST(algorithm, desynced_input)
{
    const int iterations = 50;
    auto stream0 = mo::AsyncStreamFactory::instance()->create();
    auto stream1 = mo::AsyncStreamFactory::instance()->create();
    auto stream2 = mo::AsyncStreamFactory::instance()->create();

    ASSERT_NE(stream0.get(), stream1.get());
    ASSERT_NE(stream1.get(), stream2.get());

    mo::TPublisher<int> fast_output;
    mo::TPublisher<int> slow_output;

    fast_output.setName("fast");
    slow_output.setName("slow");

    slow_output.setStream(*stream1);
    fast_output.setStream(*stream0);

    fast_output.publish(0);
    slow_output.publish(0);

    auto data1 = fast_output.getData();
    auto data2 = slow_output.getData();

    auto obj = rcc::shared_ptr<multi_input>::create();
    ASSERT_TRUE(obj);

    boost::this_fiber::sleep_for(std::chrono::milliseconds(10));
    bool finished = false;
    stream1->pushWork([&slow_output, iterations, &finished](mo::IAsyncStream*) -> void {
        for (int i = 0; i < iterations; ++i)
        {
            slow_output.publish(i, mo::Header(std::chrono::milliseconds(i)));
            boost::this_fiber::sleep_for(std::chrono::milliseconds(10));
        }
        // slow_output.setStream(nullptr);
        MO_LOG(info, "Publishing thread finished");
        finished = true;
    });

    obj->setStream(stream2);

    EXPECT_TRUE(obj->connectInput(&obj->input1_param, nullptr, &fast_output));
    EXPECT_TRUE(obj->connectInput(&obj->input2_param, nullptr, &slow_output));

    auto pub1 = obj->input1_param.getPublisher();
    auto pub2 = obj->input2_param.getPublisher();

    ASSERT_NE(pub1, nullptr) << "Did not correctly connect a publisher to this subscriber";
    ASSERT_NE(pub2, nullptr) << "Did not correctly connect a publisher to this subscriber";

    EXPECT_TRUE(pub1->checkFlags(mo::ParamFlags::kBUFFER)) << "Did not correctly negotiate a buffered connection";
    EXPECT_TRUE(pub2->checkFlags(mo::ParamFlags::kBUFFER)) << "Did not correctly negotiate a buffered connection";

    stream2->pushWork([&obj, &fast_output, &slow_output, iterations](mo::IAsyncStream*) -> void {
        int success_count = 0;

        // mo::getDefaultLogger().set_level(spdlog::level::trace);
        while (success_count < iterations - 1)
        {
            if (obj->process())
            {
                ++success_count;
            }
            else
            {
                boost::this_fiber::sleep_for(std::chrono::milliseconds(30));
            }
        }
        MO_LOG(info, "Subscribing thread finished");

    });

    for (int i = 0; i < iterations; ++i)
    {
        fast_output.publish(i, mo::Header(std::chrono::milliseconds(i)));
        boost::this_fiber::sleep_for(std::chrono::milliseconds(1));
    }
    while(!finished)
    {
        boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
    }
}
