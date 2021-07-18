#include <Aquila/core/ParameterSynchronizer.hpp>

#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriber.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <gtest/gtest.h>

TEST(parameter_synchronizer_timestamp, single_input_dedoup)
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    mo::TPublisher<uint32_t> pub0;
    mo::TSubscriber<uint32_t> sub0;
    sub0.setInput(&pub0);
    synchronizer.setInputs({&sub0});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &sub0, &header](const mo::Time* time, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::SubscriberVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub0) != vec.end());
        ASSERT_TRUE(header.timestamp);
        ASSERT_TRUE(time);
        ASSERT_EQ(header.timestamp, *time);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 1; i < 20; ++i)
    {
        pub0.publish(i, header, stream.get());
        ASSERT_TRUE(callback_invoked) << "i = " << i;
        callback_invoked = false;
        pub0.publish(i, header, stream.get());
        ASSERT_FALSE(callback_invoked) << "i = " << i;
        header = mo::Header(std::chrono::milliseconds(i));
    }
}

TEST(parameter_synchronizer_timestamp, two_synchronized_inputs)
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();


    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    mo::TSubscriber<uint32_t> sub0;
    mo::TSubscriber<uint32_t> sub1;

    sub0.setInput(&pub0);
    sub1.setInput(&pub1);

    synchronizer.setInputs({&sub0, &sub1});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &sub0, &sub1, &header](const mo::Time* time, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::SubscriberVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub1) != vec.end());
        ASSERT_TRUE(header.timestamp);
        ASSERT_TRUE(time);
        ASSERT_EQ(header.timestamp, *time);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 1; i < 20; ++i)
    {
        pub0.publish(i, header);
        pub1.publish(i + 1, header);
        ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
        callback_invoked = false;
        pub0.publish(i, header);
        ASSERT_FALSE(callback_invoked);
        header = mo::Header(std::chrono::milliseconds(i));
    }
}


TEST(parameter_synchronizer_timestamp, two_desynchronized_inputs)
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    mo::TSubscriber<uint32_t> sub0;
    mo::TSubscriber<uint32_t> sub1;

    sub0.setInput(&pub0);
    sub1.setInput(&pub1);

    synchronizer.setInputs({&sub0, &sub1});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &sub0, &sub1, &header](const mo::Time* time, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::SubscriberVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub1) != vec.end());
        ASSERT_TRUE(header.timestamp);
        ASSERT_TRUE(time);
        ASSERT_EQ(header.timestamp, *time);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 1; i < 40; ++i)
    {
        pub0.publish(i, header);
        if(i % 2 == 0)
        {
            pub1.publish(i + 1, header);
            ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
        }else
        {
            ASSERT_FALSE(callback_invoked);
        }

        callback_invoked = false;
        pub0.publish(i, header);
        ASSERT_FALSE(callback_invoked);
        header = mo::Header(std::chrono::milliseconds(i));
    }
}

int randi(int start, int end)
{
    return int((std::rand() / float(RAND_MAX)) * (end - start) + start);
}

TEST(parameter_synchronizer_timestamp, two_non_exact_inputs)
{
    aq::ParameterSynchronizer synchronizer(std::chrono::milliseconds(10));
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    mo::TSubscriber<uint32_t> sub0;
    mo::TSubscriber<uint32_t> sub1;

    sub0.setInput(&pub0);
    sub1.setInput(&pub1);

    synchronizer.setInputs({&sub0, &sub1});

    mo::Time time(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &sub0, &sub1](const mo::Time* time_, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::SubscriberVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub1) != vec.end());
        ASSERT_TRUE(time_);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 1; i < 40; ++i)
    {
        pub0.publish(i, mo::Header(time));

        pub1.publish(i + 1, mo::Header(time + std::chrono::milliseconds(randi(0, 6))));
        ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();

        callback_invoked = false;
        pub0.publish(i, mo::Header(time));
        ASSERT_FALSE(callback_invoked);
        time = mo::Time(std::chrono::milliseconds(i*33));
    }
}

TEST(parameter_synchronizer_timestamp, multiple_inputs)
{
    aq::ParameterSynchronizer synchronizer(std::chrono::milliseconds(10));
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    int N = 100;
    std::vector<mo::TFPublisher<uint32_t>> publishers(N);
    std::vector<mo::TSubscriber<uint32_t>> subscribers(N);
    std::vector<mo::ISubscriber*> pub_ptrs;
    pub_ptrs.reserve(N);
    for(uint32_t i = 0; i < N; ++i)
    {
        subscribers[i].setInput(&publishers[i]);
        pub_ptrs.push_back(&subscribers[i]);
    }
    synchronizer.setInputs(pub_ptrs);

    mo::Time time(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &subscribers](const mo::Time* time_, const mo::FrameNumber* fn,
                                                     const aq::ParameterSynchronizer::SubscriberVec_t& vec)
    {
        callback_invoked = true;
        for(const auto& sub : subscribers)
        {
            ASSERT_TRUE(std::find(vec.begin(), vec.end(), &sub) != vec.end());
        }

        ASSERT_TRUE(time_);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 1; i < 40; ++i)
    {
        publishers[0].publish(i, mo::Header(time));
        for(int j = 1; j < N; ++j)
        {
            publishers[j].publish(i + j, mo::Header(time + std::chrono::milliseconds(randi(0, 6))));
        }

        ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();

        callback_invoked = false;
        publishers[0].publish(i, mo::Header(time));
        ASSERT_FALSE(callback_invoked);
        time = mo::Time(std::chrono::milliseconds(i*33));
    }
}

// This test case is specifically to test
TEST(parameter_synchronizer_timestamp, non_buffered_input)
{

}
