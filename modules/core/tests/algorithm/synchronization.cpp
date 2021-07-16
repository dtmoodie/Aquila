#include <Aquila/core/ParameterSynchronizer.hpp>

#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <gtest/gtest.h>

TEST(parameter_synchronizer, single_input_dedoup)
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();


    mo::TPublisher<uint32_t> pub0;

    synchronizer.setInputs({&pub0});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &pub0, &header](const mo::Time* time, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::PublisherVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub0) != vec.end());
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



TEST(parameter_synchronizer, two_synchronized_inputs_timestamp)
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    synchronizer.setInputs({&pub0, &pub1});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &pub0, &pub1, &header](const mo::Time* time, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::PublisherVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub1) != vec.end());
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


TEST(parameter_synchronizer, two_desynchronized_inputs_timestamp)
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    synchronizer.setInputs({&pub0, &pub1});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &pub0, &pub1, &header](const mo::Time* time, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::PublisherVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub1) != vec.end());
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


TEST(parameter_synchronizer, two_non_exact_inputs_timestamp)
{
    aq::ParameterSynchronizer synchronizer(std::chrono::nanoseconds(RAND_MAX / 2));
    mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::create();

    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    synchronizer.setInputs({&pub0, &pub1});

    mo::Time time(std::chrono::milliseconds(0));


    bool callback_invoked = false;
    auto callback = [&callback_invoked, &pub0, &pub1](const mo::Time* time_, const mo::FrameNumber* fn,
                                                               const aq::ParameterSynchronizer::PublisherVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub1) != vec.end());
        ASSERT_TRUE(time_);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 1; i < 40; ++i)
    {
        pub0.publish(i, mo::Header(time));
        pub1.publish(i + 1, mo::Header(time + std::chrono::nanoseconds(std::rand() / 10)));
        ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();


        callback_invoked = false;
        pub0.publish(i, mo::Header(time));
        ASSERT_FALSE(callback_invoked);
        time = mo::Time(std::chrono::milliseconds(i));
    }
}
