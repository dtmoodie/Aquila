#include <Aquila/core/ParameterSynchronizer.hpp>

#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriber.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <gtest/gtest.h>

int randi(int start, int end)
{
    return int((std::rand() / float(RAND_MAX)) * (end - start) + start);
}

struct parameter_synchronizer: ::testing::Test
{
    std::shared_ptr<spdlog::logger> m_logger;
    aq::ParameterSynchronizer synchronizer;
    mo::IAsyncStream::Ptr_t stream;

    std::vector<std::unique_ptr<mo::TPublisher<uint32_t>>> pubs;

    std::vector<std::unique_ptr<mo::TSubscriber<uint32_t>>> subs;

    mo::Header header;
    bool callback_invoked = false;
    bool callback_used = false;

    parameter_synchronizer():
        m_logger(mo::getLogger()),
        synchronizer(*m_logger)
    {
        stream = mo::IAsyncStream::create();
        header = mo::Header(std::chrono::milliseconds(0));
    }

    void init(const uint32_t N, const bool use_callback = true, const bool buffered = true)
    {
        pubs.resize(N);
        subs.resize(N);
        std::vector<mo::ISubscriber*> sub_ptrs;
        sub_ptrs.reserve(N);
        for(uint32_t i = 0; i < N; ++i)
        {
            subs[i] = std::make_unique<mo::TSubscriber<uint32_t>>();
            pubs[i] = std::make_unique<mo::TPublisher<uint32_t>>();
            if(buffered)
            {
                auto buffer = mo::buffer::IBuffer::create(mo::BufferFlags::STREAM_BUFFER);
                buffer->setInput(pubs[i].get());
                subs[i]->setInput(buffer);
            }else
            {
                subs[i]->setInput(pubs[i].get());
            }
            sub_ptrs.push_back(subs[i].get());
        }
        synchronizer.setInputs(std::move(sub_ptrs));
        callback_used = use_callback;
        if(use_callback)
        {
            synchronizer.setCallback(ct::variadicBind(&parameter_synchronizer::callback, this));
        }
    }

    virtual void callback(const mo::Time* time, const mo::FrameNumber* fn,
                  const aq::ParameterSynchronizer::SubscriberVec_t& vec) = 0;
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

    void callback(const mo::Time* time, const mo::FrameNumber* fn,
                  const aq::ParameterSynchronizer::SubscriberVec_t& vec) override
    {
        callback_invoked = true;
        for(const auto& sub : subs)
        {
            ASSERT_TRUE(std::find(vec.begin(), vec.end(), sub.get()) != vec.end());
        }

        ASSERT_TRUE(header.timestamp);
        ASSERT_TRUE(time);
        ASSERT_EQ(*header.timestamp, *time);
    }
};


TEST_F(parameter_synchronizer_timestamp, find_direct_timestamp_0)
{
    pubs.resize(2);
    subs.resize(2);
    std::vector<mo::ISubscriber*> sub_ptrs;
    sub_ptrs.reserve(2);

    // Setup a buffered connection
    subs[0] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[0] = std::make_unique<mo::TPublisher<uint32_t>>();
    auto buffer = mo::buffer::IBuffer::create(mo::BufferFlags::STREAM_BUFFER);
    buffer->setInput(pubs[0].get());
    subs[0]->setInput(buffer);
    sub_ptrs.push_back(subs[0].get());

    // setup a direct connection
    subs[1] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[1] = std::make_unique<mo::TPublisher<uint32_t>>();
    subs[1]->setInput(pubs[1].get());
    sub_ptrs.push_back(subs[1].get());

    synchronizer.setInputs(std::move(sub_ptrs));

    mo::Header hdr(0 * mo::ms);
    auto ts = synchronizer.findDirectTimestamp();
    ASSERT_FALSE(ts);

    pubs[0]->publish(0, hdr);
    ts = synchronizer.findDirectTimestamp();
    ASSERT_FALSE(ts);

    pubs[1]->publish(0, hdr);
    ts = synchronizer.findDirectTimestamp();
    ASSERT_TRUE(ts);
    //ASSERT_EQ(*ts, 0 * mo::ms);
}

// Do it backwards from above to make sure we don't have any weird ordering issues
TEST_F(parameter_synchronizer_timestamp, find_direct_timestamp_1)
{
    pubs.resize(2);
    subs.resize(2);
    std::vector<mo::ISubscriber*> sub_ptrs;
    sub_ptrs.reserve(2);

    // Setup a buffered connection
    subs[1] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[1] = std::make_unique<mo::TPublisher<uint32_t>>();
    auto buffer = mo::buffer::IBuffer::create(mo::BufferFlags::STREAM_BUFFER);
    buffer->setInput(pubs[1].get());
    subs[1]->setInput(buffer);
    sub_ptrs.push_back(subs[1].get());

    // setup a direct connection
    subs[0] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[0] = std::make_unique<mo::TPublisher<uint32_t>>();
    subs[0]->setInput(pubs[0].get());
    sub_ptrs.push_back(subs[0].get());

    synchronizer.setInputs(std::move(sub_ptrs));

    mo::Header hdr(0 * mo::ms);
    auto ts = synchronizer.findDirectTimestamp();
    ASSERT_FALSE(ts);

    pubs[1]->publish(0, hdr);
    ts = synchronizer.findDirectTimestamp();
    ASSERT_FALSE(ts);

    pubs[0]->publish(0, hdr);
    ts = synchronizer.findDirectTimestamp();
    ASSERT_TRUE(ts);
    //ASSERT_EQ(*ts, 0 * mo::ms);
}

TEST_F(parameter_synchronizer_timestamp, find_earlist_timestamp)
{
    this->init(2, false);
    mo::Time earliest(1000 * mo::ms);
    std::vector<int> times{5,6,7,4, 8,9,10};
    for(auto i : times)
    {
        const mo::Time time = i * mo::ms;
        earliest = std::min(time, earliest);
        pubs[0]->publish(i, mo::Header(time));
        pubs[1]->publish(i, mo::Header(time));
    }
    mo::OptionalTime found = synchronizer.findEarliestTimestamp();
    ASSERT_TRUE(found);
    ASSERT_EQ(*found, earliest);
}

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
    this->init(2, true, false);
    this->iterate();
}

TEST_F(parameter_synchronizer_timestamp, synchronized_inputs_direct_query)
{
    this->init(2, false, false);
    const size_t sz = pubs.size();
    for (uint32_t i = 1; i < 2; ++i)
    {
        for(uint32_t j = 0; j < sz; ++j)
        {
            pubs[j]->publish(i + j, header, stream.get());
            if(j == 0)
            {
                auto header = synchronizer.getNextSample();
                ASSERT_FALSE(header);
            }
        }
        check(i);
        post(i);
    }
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
    mo::TPublisher<mo::vector<int>> pub3;
    mo::TPublisher<std::string> pub4;

    mo::TSubscriber<int> sub0;
    sub0.setInput(&pub0);
    mo::TSubscriber<float> sub1;
    sub1.setInput(&pub1);
    mo::TSubscriber<double> sub2;
    sub2.setInput(&pub2);
    mo::TSubscriber<mo::vector<int>> sub3;
    sub3.setInput(&pub3);
    mo::TSubscriber<std::string> sub4;
    sub4.setInput(&pub4);
    auto logger = mo::getLogger();
    aq::ParameterSynchronizer synchronizer(*logger);

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

struct parameter_synchronizer_framenumber: parameter_synchronizer
{

    parameter_synchronizer_framenumber():
        parameter_synchronizer()
    {
        header = mo::Header(mo::FrameNumber(0));
    }
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
            ASSERT_EQ(header->frame_number, this->header.frame_number);
        }
    }

    void post(uint32_t i)
    {
        callback_invoked = false;
        header = mo::Header(mo::FrameNumber(i));
    }

    void callback(const mo::Time* time, const mo::FrameNumber* fn,
                  const aq::ParameterSynchronizer::SubscriberVec_t& vec) override
    {
        callback_invoked = true;
        for(const auto& sub : subs)
        {
            ASSERT_TRUE(std::find(vec.begin(), vec.end(), sub.get()) != vec.end());
        }

        ASSERT_TRUE(header.frame_number.valid());
        ASSERT_TRUE(fn);
        ASSERT_EQ(header.frame_number, *fn);
    }
};

TEST_F(parameter_synchronizer_framenumber, find_direct_framenumber_0)
{
    pubs.resize(2);
    subs.resize(2);
    std::vector<mo::ISubscriber*> sub_ptrs;
    sub_ptrs.reserve(2);

    // Setup a buffered connection
    subs[0] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[0] = std::make_unique<mo::TPublisher<uint32_t>>();
    auto buffer = mo::buffer::IBuffer::create(mo::BufferFlags::STREAM_BUFFER);
    buffer->setInput(pubs[0].get());
    subs[0]->setInput(buffer);
    sub_ptrs.push_back(subs[0].get());

    // setup a direct connection
    subs[1] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[1] = std::make_unique<mo::TPublisher<uint32_t>>();
    subs[1]->setInput(pubs[1].get());
    sub_ptrs.push_back(subs[1].get());

    synchronizer.setInputs(std::move(sub_ptrs));

    mo::Header hdr(mo::FrameNumber(0));
    auto ts = synchronizer.findDirectFrameNumber();
    ASSERT_FALSE(ts.valid());

    pubs[0]->publish(0, hdr);
    ts = synchronizer.findDirectFrameNumber();
    ASSERT_FALSE(ts.valid());

    pubs[1]->publish(0, hdr);
    ts = synchronizer.findDirectFrameNumber();
    ASSERT_TRUE(ts.valid());
}

// Do it backwards from above to make sure we don't have any weird ordering issues
TEST_F(parameter_synchronizer_framenumber, find_direct_framenumber_1)
{
    pubs.resize(2);
    subs.resize(2);
    std::vector<mo::ISubscriber*> sub_ptrs;
    sub_ptrs.reserve(2);

    // Setup a buffered connection
    subs[1] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[1] = std::make_unique<mo::TPublisher<uint32_t>>();
    auto buffer = mo::buffer::IBuffer::create(mo::BufferFlags::STREAM_BUFFER);
    buffer->setInput(pubs[1].get());
    subs[1]->setInput(buffer);
    sub_ptrs.push_back(subs[1].get());

    // setup a direct connection
    subs[0] = std::make_unique<mo::TSubscriber<uint32_t>>();
    pubs[0] = std::make_unique<mo::TPublisher<uint32_t>>();
    subs[0]->setInput(pubs[0].get());
    sub_ptrs.push_back(subs[0].get());

    synchronizer.setInputs(std::move(sub_ptrs));

    mo::Header hdr(mo::FrameNumber(0));
    auto ts = synchronizer.findDirectFrameNumber();
    ASSERT_FALSE(ts.valid());

    pubs[1]->publish(0, hdr);
    ts = synchronizer.findDirectFrameNumber();
    ASSERT_FALSE(ts.valid());

    pubs[0]->publish(0, hdr);
    ts = synchronizer.findDirectFrameNumber();
    ASSERT_TRUE(ts.valid());
    //ASSERT_EQ(*ts, 0 * mo::ms);
}

TEST_F(parameter_synchronizer_framenumber, find_earlist_framenumber)
{
    this->init(2, false);
    mo::FrameNumber earliest(1000);
    std::vector<int> times{5,6,7,4,8,9,10};
    for(auto i : times)
    {
        const mo::FrameNumber fn(i);
        earliest = std::min(fn, earliest);
        pubs[0]->publish(i, mo::Header(fn));
        pubs[1]->publish(i, mo::Header(fn));
    }
    mo::FrameNumber found = synchronizer.findEarliestFrameNumber();
    ASSERT_TRUE(found);
    ASSERT_EQ(found, earliest);
}

TEST_F(parameter_synchronizer_framenumber, single_input_dedoup_callback)
{
    this->init(1);
    iterate();
}

TEST_F(parameter_synchronizer_framenumber, single_input_dedoup_query)
{
    this->init(1, false);
    iterate();
}


TEST_F(parameter_synchronizer_framenumber, synchronized_inputs_direct_callback)
{
    this->init(2, false, false);
    this->iterate();
}

TEST_F(parameter_synchronizer_framenumber, synchronized_inputs_direct_query)
{
    this->init(2, false, false);
    this->iterate();
}


TEST_F(parameter_synchronizer_framenumber, synchronized_inputs_full_buffered_callback)
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

TEST_F(parameter_synchronizer_framenumber, synchronized_inputs_full_buffered_query)
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

TEST_F(parameter_synchronizer_framenumber, synchronized_inputs_half_buffered_callback)
{
    this->init(2);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf1->setInput(pubs[1].get());

    subs[1]->setInput(buf1);
    this->iterate();
}

TEST_F(parameter_synchronizer_framenumber, synchronized_inputs_half_buffered_query)
{
    this->init(2, false);
    auto buf1 = mo::buffer::IBuffer::create(mo::BufferFlags::DROPPING_STREAM_BUFFER);

    buf1->setInput(pubs[1].get());

    subs[1]->setInput(buf1);
    this->iterate();
}

// Timestamp is the same but one only publishes half the time
TEST_F(parameter_synchronizer_framenumber, desynchronized_inputs_callback)
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

TEST_F(parameter_synchronizer_framenumber, desynchronized_inputs_query)
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


TEST_F(parameter_synchronizer_framenumber, multiple_inputs_callback)
{
    const int N = 100;
    this->init(N);

    mo::FrameNumber fn(0);

    for (uint32_t i = 1; i < 40; ++i)
    {
        header = mo::Header(fn);
        pubs[0]->publish(i, mo::Header(fn));
        for(int j = 1; j < N; ++j)
        {
            pubs[j]->publish(i + j, mo::Header(fn));
        }

        check(i);
        post(i);
        fn = mo::FrameNumber(i);
    }
}

TEST_F(parameter_synchronizer_framenumber, multiple_inputs_query)
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


TEST_F(parameter_synchronizer_framenumber, typed_query)
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

    auto logger = mo::getLogger();
    aq::ParameterSynchronizer synchronizer(*logger);

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

