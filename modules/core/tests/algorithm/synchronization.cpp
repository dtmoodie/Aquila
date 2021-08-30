#include <Aquila/core/ParameterSynchronizer.hpp>

#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriber.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <gtest/gtest.h>

struct parameter_synchronizer: ::testing::Test
{
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream;

    std::vector<std::unique_ptr<mo::TPublisher<uint32_t>>> pubs;

    std::vector<std::unique_ptr<mo::TSubscriber<uint32_t>>> subs;

    mo::Header header;
    bool callback_invoked = false;
    bool callback_used = false;

    parameter_synchronizer()
    {
        stream = mo::IAsyncStream::create();
        header = mo::Header(std::chrono::milliseconds(0));
    }

    void init(const uint32_t N, const bool use_callback = true)
    {
        pubs.resize(N);
        subs.resize(N);
        std::vector<mo::ISubscriber*> sub_ptrs;
        sub_ptrs.reserve(N);
        for(uint32_t i = 0; i < N; ++i)
        {
            subs[i] = std::make_unique<mo::TSubscriber<uint32_t>>();
            pubs[i] = std::make_unique<mo::TPublisher<uint32_t>>();
            subs[i]->setInput(pubs[i].get());
            sub_ptrs.push_back(subs[i].get());
        }
        synchronizer.setInputs(std::move(sub_ptrs));
        callback_used = use_callback;
        if(use_callback)
        {
            synchronizer.setCallback(ct::variadicBind(&parameter_synchronizer::callback, this));
        }
    }

    void callback(const mo::Time* time, const mo::FrameNumber* fn,
                  const aq::ParameterSynchronizer::SubscriberVec_t& vec)
    {
        callback_invoked = true;
        for(const auto& sub : subs)
        {
            ASSERT_TRUE(std::find(vec.begin(), vec.end(), sub.get()) != vec.end());
        }

        ASSERT_TRUE(header.timestamp);
        ASSERT_TRUE(time);
        ASSERT_EQ(header.timestamp, *time);
    }
};


struct parameter_synchronizer_timestamp: parameter_synchronizer
{
    void iterate(const uint32_t N = 20)
    {
        const size_t sz = pubs.size();
        for (uint32_t i = 1; i < N; ++i)
        {
            for(uint32_t j = 0; j < sz; ++j)
            {
                pubs[j]->publish(i + j, header, stream.get());
            }
            check(i);
            post(i);
        }
    }

    void check(uint32_t i)
    {
        if(callback_used)
        {
            ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
        }else
        {
            auto timestamp = synchronizer.getNextSample();
            ASSERT_NE(boost::none, timestamp);
            ASSERT_EQ(*timestamp, *header.timestamp);
        }
    }

    void post(uint32_t i)
    {
        callback_invoked = false;
        pubs[0]->publish(i, header);
        ASSERT_FALSE(callback_invoked) << "i = " << i;
        header = mo::Header(std::chrono::milliseconds(i));
    }
};

TEST_F(parameter_synchronizer_timestamp, single_input_dedoup)
{
    this->init(1);
    iterate();
}


TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_direct)
{
    this->init(2);
    this->iterate();
}


TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_full_buffered)
{
    this->init(2);
    auto buf0 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf0->setInput(pubs[0].get());
    buf1->setInput(pubs[1].get());

    subs[0]->setInput(buf0);
    subs[1]->setInput(buf1);

    this->iterate();
}

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_half_buffered)
{
    this->init(2);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf1->setInput(pubs[1].get());

    subs[1]->setInput(buf1);
    this->iterate();
}

// Timestamp is the same but one only publishes half the time
TEST_F(parameter_synchronizer_timestamp, desynchronized_inputs)
{
    this->init(2);

    for (uint32_t i = 1; i < 40; ++i)
    {
        pubs[0]->publish(i, header);
        if(i % 2 == 0)
        {
            pubs[1]->publish(i + 1, header);
            ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
        }else
        {
            ASSERT_FALSE(callback_invoked);
        }
        post(i);
    }
}

int randi(int start, int end)
{
    return int((std::rand() / float(RAND_MAX)) * (end - start) + start);
}

// Jitter added to timestamp
TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_with_jitter)
{
    this->init(2);
    this->synchronizer.setSlop(std::chrono::milliseconds(10));
    mo::Time time(std::chrono::milliseconds(0));

    for (uint32_t i = 1; i < 40; ++i)
    {
        header = mo::Header(time);
        pubs[0]->publish(i, mo::Header(time));

        pubs[1]->publish(i + 1, mo::Header(time + std::chrono::milliseconds(randi(0, 6))));
        ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();

        check(i);
        post(i);
        time = mo::Time(std::chrono::milliseconds(i*33));
    }
}

TEST_F(parameter_synchronizer_timestamp, multiple_inputs)
{
    const int N = 100;
    this->synchronizer.setSlop(std::chrono::milliseconds(10));
    this->init(N);

    mo::Time time(std::chrono::milliseconds(0));

    for (uint32_t i = 1; i < 40; ++i)
    {
        header = mo::Header(time);
        pubs[0]->publish(i, mo::Header(time));
        for(int j = 1; j < N; ++j)
        {
            pubs[j]->publish(i + j, mo::Header(time + std::chrono::milliseconds(randi(0, 6))));
        }

        check(i);
        post(i);
        time = mo::Time(std::chrono::milliseconds(i*33));
    }
}

