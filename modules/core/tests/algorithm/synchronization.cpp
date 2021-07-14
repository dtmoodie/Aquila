#include <Aquila/core/ParameterSynchronizer.hpp>

#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <gtest/gtest.h>

TEST(parameter_synchronizer, two_synchronized_inputs_timestamp)
{
    aq::ParameterSynchronizer synchronizer;

    mo::TPublisher<uint32_t> pub0;
    mo::TPublisher<uint32_t> pub1;

    synchronizer.setInputs({&pub0, &pub1});

    mo::Header header(std::chrono::milliseconds(0));

    bool callback_invoked = false;
    auto callback = [&callback_invoked, &pub0, &pub1, &header](const mo::Header& header_,
                                                               const aq::ParameterSynchronizer::PublisherVec_t& vec) {
        callback_invoked = true;
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub0) != vec.end());
        ASSERT_TRUE(std::find(vec.begin(), vec.end(), &pub1) != vec.end());
        ASSERT_TRUE(header.timestamp);
        ASSERT_EQ(header, header_);
    };

    synchronizer.setCallback(std::move(callback));
    for (uint32_t i = 0; i < 20; ++i)
    {
        pub0.publish(i, header);
        pub1.publish(i + 1, header);
        ASSERT_TRUE(callback_invoked);
        callback_invoked = false;
        header = mo::Header(std::chrono::milliseconds(i));
    }
}
