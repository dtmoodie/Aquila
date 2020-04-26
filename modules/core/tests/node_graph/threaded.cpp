#include "common.hpp"

struct MultiThreadedProducer : public aq::nodes::Node
{
    MO_DERIVE(MultiThreadedProducer, aq::nodes::Node)
        OUTPUT(int, out_a, 0)
        // APPEND_FLAGS(out_a, mo::ParamFlags::kSOURCE)
    MO_END;

    bool processImpl() override
    {
        producer_ready = true;
        if (!consumer_ready)
        {
            setModified();
            return false;
        }
        if (timestamp_mode)
        {
            out_a.publish(iterations, mo::Time(mo::ms * iterations));
        }
        else
        {
            out_a.publish(iterations, mo::Header(static_cast<uint64_t>(iterations)));
        }
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        setModified();
        ++iterations;
        aq::nodes::INode* This = this;
        sig_node_updated(This);
        return true;
    }
    int iterations = 0;
    boost::condition_variable_any cv;
    volatile bool consumer_ready = false;
    volatile bool producer_ready = false;
};

/*
struct ThreadedFixture
{
    ThreadedFixture()
    {
        a = a.create();
        d = d.create();
        ds = ds.create();
        ds->addNode(a);
        ds->startThread();
        while (!a->producer_ready)
        {
        }
    }
    rcc::shared_ptr<mt_a> a;
    rcc::shared_ptr<node_d> d;
    rcc::shared_ptr<IGraph> ds;
};

BOOST_FIXTURE_TEST_SUITE(ThreadedSuite, ThreadedFixture)
// This case represents a producer (a) and a consumer (c) on different threads
BOOST_AUTO_TEST_CASE(linear_threaded)
{
    auto stream = mo::AsyncStreamFactory::instance()->create();
    d->setStream(stream);
    BOOST_REQUIRE(d->connectInput("in_d", a.get(), "out_a"));
    a->consumer_ready = true;
    nodes::INode* ptr = a.get();
    a->sig_node_updated(ptr);
    for (int i = 0; i < 1000; ++i)
    {
        d->process();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
    std::cout << a->iterations << " " << d->iterations << std::endl;
}

BOOST_AUTO_TEST_SUITE_END() // ThreadedSuite
*/

MO_REGISTER_CLASS(MultiThreadedProducer)
