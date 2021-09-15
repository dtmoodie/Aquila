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
            auto header = synchronizer.getNextSample();
            ASSERT_TRUE(header) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
            ASSERT_EQ(header->timestamp, this->header.timestamp);
        }
    }

    void post(uint32_t i)
    {
        callback_invoked = false;
        header = mo::Header(std::chrono::milliseconds(i));
    }
};

TEST_F(parameter_synchronizer_timestamp, single_input_dedoup_callback)
{
    this->init(1);
    iterate();
}

TEST_F(parameter_synchronizer_timestamp, single_input_dedoup_query)
{
    this->init(1, false);
    iterate();
}


TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_direct_callback)
{
    this->init(2);
    this->iterate();
}

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_direct_query)
{
    this->init(2, false);
    this->iterate();
}


TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_full_buffered_callback)
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

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_full_buffered_query)
{
    this->init(2, false);
    auto buf0 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf0->setInput(pubs[0].get());
    buf1->setInput(pubs[1].get());

    subs[0]->setInput(buf0);
    subs[1]->setInput(buf1);

    this->iterate();
}

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_half_buffered_callback)
{
    this->init(2);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf1->setInput(pubs[1].get());

    subs[1]->setInput(buf1);
    this->iterate();
}

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_half_buffered_query)
{
    this->init(2, false);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf1->setInput(pubs[1].get());

    subs[1]->setInput(buf1);
    this->iterate();
}

// Timestamp is the same but one only publishes half the time
TEST_F(parameter_synchronizer_timestamp, desynchronized_inputs_callback)
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

TEST_F(parameter_synchronizer_timestamp, desynchronized_inputs_query)
{
    this->init(2, false);

    for (uint32_t i = 1; i < 40; ++i)
    {
        pubs[0]->publish(i, header);
        if(i % 2 == 0)
        {
            pubs[1]->publish(i + 1, header);
            if(callback_used)
            {
                ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
            }else
            {
                ASSERT_TRUE(synchronizer.getNextSample());
            }
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
TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_with_jitter_callback)
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

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_with_jitter_query)
{
    this->init(2, false);
    this->synchronizer.setSlop(std::chrono::milliseconds(10));
    mo::Time time(std::chrono::milliseconds(0));

    for (uint32_t i = 1; i < 40; ++i)
    {
        header = mo::Header(time);
        pubs[0]->publish(i, mo::Header(time));

        pubs[1]->publish(i + 1, mo::Header(time + std::chrono::milliseconds(randi(0, 6))));
        if(this->callback_used)
        {
            ASSERT_TRUE(callback_invoked) << "i = " << i << " " << synchronizer.findEarliestCommonTimestamp();
        }

        check(i);
        post(i);
        time = mo::Time(std::chrono::milliseconds(i*33));
    }
}

TEST_F(parameter_synchronizer_timestamp, multiple_inputs_callback)
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

TEST_F(parameter_synchronizer_timestamp, multiple_inputs_query)
{
    const int N = 100;
    this->synchronizer.setSlop(std::chrono::milliseconds(10));
    this->init(N, false);

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


TEST_F(parameter_synchronizer_timestamp, typed_query)
{
    mo::TPublisher<int> pub0;
    mo::TPublisher<float> pub1;
    mo::TPublisher<double> pub2;
    mo::TPublisher<std::vector<int>> pub3;
    mo::TPublisher<std::string> pub4;

    mo::TSubscriber<int> sub0;
    sub0.setInput(&pub0);
    mo::TSubscriber<float> sub1;
    sub1.setInput(&pub1);
    mo::TSubscriber<double> sub2;
    sub2.setInput(&pub2);
    mo::TSubscriber<std::vector<int>> sub3;
    sub3.setInput(&pub3);
    mo::TSubscriber<std::string> sub4;
    sub4.setInput(&pub4);

    aq::ParameterSynchronizer synchronizer;

    synchronizer.setInputs(std::vector<mo::ISubscriber*>{&sub0, &sub1, &sub2, &sub3, &sub4});

    std::tuple<mo::TDataContainerConstPtr_t<int>, mo::TDataContainerConstPtr_t<float>, mo::TDataContainerConstPtr_t<double>, mo::TDataContainerConstPtr_t<std::vector<int>>, mo::TDataContainerConstPtr_t<std::string>> tup0;
    std::tuple<int, float, double, mo::vector<int>, std::string> tup1;
    mo::IAsyncStreamPtr_t stream = mo::IAsyncStream::create();
    ASSERT_FALSE(synchronizer.getNextSample(tup0, stream.get()));
    mo::Header header(10 * mo::ms);
    pub0.publish(5, header, stream.get());
    pub1.publish(15.0F, header, stream.get());
    pub2.publish(25.0, header, stream.get());
    auto vec = pub3.create(4);
    vec->data[0] = 4; vec->data[1] = 5; vec->data[2] = 6; vec->data[3] = 7;
    vec->header = header;
    pub3.publish(std::move(vec), stream.get());
    pub4.publish("asdf", header, stream.get());

    bool success = synchronizer.getNextSample(tup0, stream.get());
    ASSERT_TRUE(success);
    ASSERT_EQ(std::get<0>(tup0)->data, 5);
    ASSERT_EQ(std::get<1>(tup0)->data, 15.0F);
    ASSERT_EQ(std::get<2>(tup0)->data, 25.0);
    ASSERT_EQ(std::get<4>(tup0)->data, "asdf");

    header = mo::Header(15 * mo::ms);
    pub0.publish(5, header, stream.get());
    pub1.publish(12.0F, header, stream.get());
    pub2.publish(25.0, header, stream.get());
    vec->data[0] = 4; vec->data[1] = 5; vec->data[2] = 6; vec->data[3] = 7;
    vec->header = header;
    pub3.publish(std::move(vec), stream.get());
    pub4.publish("asdf", header, stream.get());

    success = synchronizer.getNextSample(tup1, stream.get());
    ASSERT_TRUE(success);
    ASSERT_EQ(std::get<0>(tup1), 5);
    ASSERT_EQ(std::get<1>(tup1), 12.0F);
    ASSERT_EQ(std::get<2>(tup1), 25.0);
    ASSERT_EQ(std::get<4>(tup1), "asdf");
}

