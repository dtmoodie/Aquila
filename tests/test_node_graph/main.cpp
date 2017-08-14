#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include <Aquila/core/Aquila.hpp>
#include <Aquila/core/IDataStream.hpp>
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/ThreadedNode.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include "Aquila/rcc/SystemTable.hpp"

#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/thread/ThreadPool.hpp"
#include "MetaObject/core/detail/Allocator.hpp"

#include "Aquila/core/detail/AlgorithmImpl.hpp"
#include "Aquila/core/Logging.hpp"
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "AquilaNodes"
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/parameterized_test.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/common.hpp>
#include <boost/log/attributes.hpp>
#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <type_traits>
//#include "../unit_test.hpp"

using namespace aq;
using namespace aq::nodes;
#define TEST_FRAME_NUMBER 0

bool timestamp_mode = true;
#if BOOST_VERSION > 105800
#define MY_BOOST_TEST_ADD_ARGS __FILE__, __LINE__,
#define MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR ,boost::unit_test::decorator::collector::instance()
#else
#define MY_BOOST_TEST_ADD_ARGS
#define MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR
#endif

#define BOOST_FIXTURE_PARAM_TEST_CASE( test_name, F, mbegin, mend )     \
struct test_name : public F {                                           \
   typedef ::std::remove_const< ::std::remove_reference< decltype(*(mbegin)) >::type>::type param_t; \
   void test_method(const param_t &);                                   \
};                                                                      \
                                                                        \
void BOOST_AUTO_TC_INVOKER( test_name )(const test_name::param_t &param) \
{                                                                       \
    test_name t;                                                        \
    t.test_method(param);                                               \
}                                                                       \
                                                                        \
BOOST_AUTO_TU_REGISTRAR( test_name )(                                   \
    boost::unit_test::make_test_case(                                   \
       &BOOST_AUTO_TC_INVOKER( test_name ), #test_name,                 \
       MY_BOOST_TEST_ADD_ARGS                                           \
       (mbegin), (mend))                                                \
       MY_BOOST_TEST_DEFAULT_DEC_COLLECTOR);                            \
                                                                        \
void test_name::test_method(const param_t &param)                       \




#define BOOST_AUTO_PARAM_TEST_CASE( test_name, mbegin, mend )           \
   BOOST_FIXTURE_PARAM_TEST_CASE( test_name,                            \
                                  BOOST_AUTO_TEST_CASE_FIXTURE,         \
                                  mbegin, mend)

struct node_a: public nodes::Node
{
    MO_DERIVE(node_a, nodes::Node)
        OUTPUT(int, out_a, 0)
    MO_END;

    bool processImpl()
    {
        if(timestamp_mode == true)
        {
            out_a_param.updateData(iterations, mo::Time_t(mo::ms * iterations));
        }else
        {
            out_a_param.updateData(iterations, mo::tag::_frame_number = iterations);
        }

        _modified = true;
        ++iterations;
        return true;
    }
    int iterations = 0;
};

struct node_b: public nodes::Node
{
    MO_DERIVE(node_b, nodes::Node)
        OUTPUT(int, out_b, 0)
    MO_END;

    bool processImpl()
    {
        if(timestamp_mode == true)
        {
            out_b_param.updateData(iterations, mo::Time_t(mo::ms * iterations));
        }else
        {
            out_b_param.updateData(iterations, mo::tag::_frame_number = iterations);
        }

        _modified = true;
        ++iterations;
        return true;
    }
    int iterations = 0;
};


struct node_c: public nodes::Node
{
    MO_DERIVE(node_c, nodes::Node)
        INPUT(int, in_a, nullptr)
        INPUT(int, in_b, nullptr)
    MO_END;

    bool processImpl()
    {
        BOOST_REQUIRE_EQUAL(*in_a, *in_b);

        sum = *in_a + *in_b;
        ++iterations;
        return true;
    }
    void check_timestamps()
    {
        Algorithm::impl* impl = _pimpl;
        MO_LOG(debug) << impl->_ts_processing_queue.size() << " frames left to process";
    }
    int sum = 0;
    int iterations = 0;
};

struct node_d : public nodes::Node
{
    MO_DERIVE(node_d, nodes::Node)
        INPUT(int, in_d, nullptr)
        OUTPUT(int, out_d, 0)
    MO_END;
    bool processImpl()
    {
        if(timestamp_mode == true)
        {
            BOOST_REQUIRE_EQUAL(mo::Time_t(*in_d * mo::ms), *in_d_param.getTimestamp());
            out_d_param.updateData(*in_d, *in_d_param.getTimestamp());
        }else
        {
            BOOST_REQUIRE_EQUAL(*in_d, in_d_param.getFrameNumber());
            out_d_param.updateData(*in_d, in_d_param.getFrameNumber());
        }

        ++iterations;
        return true;
    }
    int iterations = 0;
};

MO_REGISTER_CLASS(node_a)
MO_REGISTER_CLASS(node_b)
MO_REGISTER_CLASS(node_c)
MO_REGISTER_CLASS(node_d)

struct GlobalFixture
{
    GlobalFixture()
    {
        mo::MetaObjectFactory::instance(&table);
        aq::Init();
        aq::SetupLogging();
        mo::MetaObjectFactory::instance()->registerTranslationUnit();

        boost::filesystem::path currentDir = boost::filesystem::path("./").parent_path();
    #ifdef _MSC_VER
        currentDir = boost::filesystem::path("./");
    #else
        currentDir = boost::filesystem::path("./Plugins");
    #endif
        MO_LOG(info) << "Looking for plugins in: " << currentDir.string();
        boost::filesystem::directory_iterator end_itr;
        if(boost::filesystem::is_directory(currentDir))
        {
            for(boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
            {
                if(boost::filesystem::is_regular_file(itr->path()))
                {
    #ifdef _MSC_VER
                    if(itr->path().extension() == ".dll")
    #else
                    if(itr->path().extension() == ".so")
    #endif
                    {
                        std::string file = itr->path().string();
                        mo::MetaObjectFactory::instance()->loadPlugin(file);
                    }
                }
            }
        }
        g_allocator = mo::Allocator::getThreadSafeAllocator();
        cv::cuda::GpuMat::setDefaultAllocator(g_allocator);
        cv::Mat::setDefaultAllocator(g_allocator);
        g_allocator->setName("Global Allocator");
        mo::GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(g_allocator);
        mo::CpuThreadAllocatorSetter<cv::Mat>::Set(g_allocator);
        //boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);
    }
    ~GlobalFixture()
    {
        mo::ThreadPool::instance()->cleanup();
        mo::ThreadSpecificQueue::cleanup();
        mo::Allocator::cleanupThreadSpecificAllocator();
        delete g_allocator;
    }
    mo::Allocator* g_allocator;
    SystemTable table;
};
BOOST_GLOBAL_FIXTURE(GlobalFixture);



struct BranchingFixture
{
    BranchingFixture()
    {

        ds = ds.create();
        ds->stopThread();
        a = a.create();
        b = b.create();
        c = c.create();
        a->setDataStream(ds.get());
    }

    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_b> b;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IDataStream> ds;
};
using namespace mo;
static const std::pair<mo::ParamType, bool> settings[] =
{
    {CircularBuffer_e, true},
    {Map_e, true},
    {StreamBuffer_e, true},
    {BlockingStreamBuffer_e, true},
    {NNStreamBuffer_e, true}
#if TEST_FRAME_NUMBER
    ,{ CircularBuffer_e, false } ,
    { Map_e, false },
    { StreamBuffer_e, false },
    { BlockingStreamBuffer_e, false },
    { NNStreamBuffer_e, false }
#endif
};
#if TEST_FRAME_NUMBER
static const int num_settings = 10;
#else
static const int num_settings = 5;
#endif
BOOST_FIXTURE_TEST_SUITE(suite, BranchingFixture)
/*     a
*     | \
*     |  \
*     |  b
*     | /
*     c
*/
BOOST_AUTO_TEST_CASE(branching_direct_ts)
{
    timestamp_mode = true;
    a->addChild(b);
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->connectInput(b, "out_b", "in_b"));
    for(int i = 0; i < 100; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(a->out_a, i);
        BOOST_REQUIRE_EQUAL(b->out_b, i);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}
#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(branching_direct_fn)
{
    timestamp_mode = false;
    a->addChild(b);
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->connectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(a->out_a, i);
        BOOST_REQUIRE_EQUAL(b->out_b, i);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}
#endif

BOOST_AUTO_PARAM_TEST_CASE(branching_buffered, settings, settings + num_settings)
{
    timestamp_mode = param.second;
    a->addChild(b);
    std::cout << "Buffer: " << mo::paramTypeToString(param.first) << " ts: " << (timestamp_mode ? "on" : "off" )<< std::endl;
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->connectInput(b, "out_b", "in_b", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    for (int i = 0; i < 1000; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(a->out_a, i);
        BOOST_REQUIRE_EQUAL(b->out_b, i);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}

/*    a     b
*     |    /
*     |   /
*     |  /
*     | /
*     c
*/
BOOST_AUTO_TEST_CASE(merging_direct_ts)
{
    timestamp_mode = true;
    b->setDataStream(ds.get());
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->connectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        b->process();
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}
#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(merging_direct_fn)
{
    timestamp_mode = false;
    b->setDataStream(ds.get());
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->connectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        b->process();
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}
#endif


BOOST_AUTO_TEST_SUITE_END()

struct node_e: public nodes::Node
{
    MO_DERIVE(node_e, nodes::Node)
        OUTPUT(int, out, 0)
    MO_END;
    bool processImpl()
    {
        if(timestamp_mode)
        {
            out_param.updateData(iterations*2, mo::Time_t(mo::ms * (iterations * 2)));
        }else
        {
            out_param.updateData(iterations* 2, mo::tag::_frame_number = iterations*2);
        }
        _modified = true;
        ++iterations;
        return true;
    }
    int iterations = 0;
};
MO_REGISTER_CLASS(node_e)

BOOST_AUTO_TEST_CASE(merging_direct_desynced_ts)
{
    timestamp_mode = true;
    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_e> e;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IDataStream> ds;
    ds = ds.create();
    ds->stopThread();
    a = a.create();
    e = e.create();
    c = c.create();
    a->setDataStream(ds.get());

    e->setDataStream(ds.get());
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a", mo::ParamType(mo::ForceBufferedConnection_e | mo::CircularBuffer_e)));
    BOOST_REQUIRE(c->connectInput(e, "out", "in_b"));
    int e_iters = 0;
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(e->iterations, e_iters);
        BOOST_REQUIRE_EQUAL(c->iterations, e_iters);
        if (i % 2 == 0)
        {
            e->process();
            ++e_iters;
            BOOST_REQUIRE_EQUAL(e->iterations, e_iters);
            BOOST_REQUIRE_EQUAL(c->iterations, e_iters);
        }
    }
}
#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(merging_direct_desynced_fn)
{
    timestamp_mode = false;
    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_e> e;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IDataStream> ds;
    ds = ds.create();
    ds->stopThread();
    a = a.create();
    e = e.create();
    c = c.create();
    a->setDataStream(ds.get());

    e->setDataStream(ds.get());
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a", mo::ParamType(mo::ForceBufferedConnection_e | mo::CircularBuffer_e)));
    BOOST_REQUIRE(c->connectInput(e, "out", "in_b"));
    int e_iters = 0;
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(e->iterations, e_iters);
        BOOST_REQUIRE_EQUAL(c->iterations, e_iters);
        if (i % 2 == 0)
        {
            e->process();
            ++e_iters;
            BOOST_REQUIRE_EQUAL(e->iterations, e_iters);
            BOOST_REQUIRE_EQUAL(c->iterations, e_iters);
        }
    }
}
#endif

struct DiamondFixture
{
    DiamondFixture()
    {
        ds = ds.create();
        ds->stopThread();
        a = a.create();
        d1 = d1.create();
        d2 = d1.create();
        c = c.create();
        a->setDataStream(ds.get());
    }

    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_d> d1;
    rcc::shared_ptr<node_d> d2;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IDataStream> ds;
};

BOOST_FIXTURE_TEST_SUITE(DiamondSuite, DiamondFixture)

BOOST_AUTO_TEST_CASE(diamond_direct_ts)
{
    timestamp_mode = true;
    BOOST_REQUIRE(d1->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->connectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->connectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}

#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(diamond_direct_fn)
{
    timestamp_mode = false;
    BOOST_REQUIRE(d1->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->connectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->connectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}
#endif

BOOST_AUTO_PARAM_TEST_CASE(diamond_buffered_top, settings, settings + num_settings)
{
    timestamp_mode = param.second;
    BOOST_REQUIRE(d1->connectInput(a, "out_a", "in_d", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(d2->connectInput(a, "out_a", "in_d", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->connectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->connectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}

BOOST_AUTO_PARAM_TEST_CASE(diamond_buffered_bottom, settings, settings + num_settings)
{
    timestamp_mode = param.second;
    std::cout << "Buffer: " << mo::paramTypeToString(param.first)
              << " sync: " << (timestamp_mode ? "timestamp" : "framenumber") << std::endl;
    BOOST_REQUIRE(d1->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->connectInput(d1, "out_d", "in_a", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->connectInput(d2, "out_d", "in_b", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}

BOOST_AUTO_PARAM_TEST_CASE(diamond_buffered_left, settings, settings + num_settings)
{
    timestamp_mode = param.second;
    //std::cout << "Setting timestamp mode to " << (timestamp_mode ? "on\n" : "off\n");
    //std::cout << "Using buffer " << mo::ParameterTypeFlagsToString(param.first) << std::endl;
    std::cout << "Buffer: " << mo::paramTypeToString(param.first) << " ts: " << (timestamp_mode ? "on" : "off") << std::endl;
    BOOST_REQUIRE(d1->connectInput(a, "out_a", "in_d", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(d2->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->connectInput(d1, "out_d", "in_a", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->connectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}

BOOST_AUTO_PARAM_TEST_CASE(diamond_buffered_right, settings, settings + num_settings)
{
    timestamp_mode = param.second;
    BOOST_REQUIRE(d1->connectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->connectInput(a, "out_a", "in_d", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->connectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->connectInput(d2, "out_d", "in_b", mo::ParamType(mo::ForceBufferedConnection_e | param.first)));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}

BOOST_AUTO_TEST_SUITE_END() // DiamondSuite

struct mt_a: public aq::nodes::Node
{
    MO_DERIVE(mt_a, aq::nodes::Node)
        OUTPUT(int, out_a, 0)
        APPEND_FLAGS(out_a, mo::ParamFlags::Source_e)
    MO_END;

    bool processImpl()
    {
        producer_ready = true;
        if(!consumer_ready)
        {
            _modified = true;
            return false;
        }
        if(timestamp_mode == true)
        {
            out_a_param.updateData(iterations, mo::Time_t(mo::ms * iterations));
        }else
        {
            out_a_param.updateData(iterations, mo::tag::_frame_number = iterations);
        }
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        _modified = true;
        ++iterations;
        aq::nodes::Node* This = this;
        sig_node_updated(This);
        return true;
    }
    int iterations = 0;
    boost::condition_variable_any cv;
    volatile bool consumer_ready = false;
    volatile bool producer_ready = false;
};
MO_REGISTER_CLASS(mt_a)
struct ThreadedFixture
{
    ThreadedFixture()
    {
        a = a.create();
        d = d.create();
        ds = ds.create();
        ds->addNode(a);
        ds->startThread();
        while(!a->producer_ready)
        {

        }
    }

    ~ThreadedFixture()
    {

    }
    rcc::shared_ptr<mt_a> a;
    rcc::shared_ptr<node_d> d;
    rcc::shared_ptr<IDataStream> ds;
};

BOOST_FIXTURE_TEST_SUITE(ThreadedSuite, ThreadedFixture)
// This case represents a producer (a) and a consumer (c) on different threads
BOOST_AUTO_TEST_CASE(linear_threaded)
{
    d->setContext(mo::Context::getDefaultThreadContext());
    BOOST_REQUIRE(d->connectInput(a, "out_a", "in_d"));
    a->consumer_ready = true;
    nodes::Node* ptr = a.get();
    a->sig_node_updated(ptr);
    for(int i = 0; i < 1000; ++i)
    {
        d->process();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
    std::cout << a->iterations << " " << d->iterations << std::endl;
}

BOOST_AUTO_TEST_SUITE_END() // ThreadedSuite

BOOST_AUTO_TEST_CASE(finish)
{
    mo::ThreadSpecificQueue::cleanup();
    mo::ThreadPool::instance()->cleanup();
    mo::Allocator::cleanupThreadSpecificAllocator();
}
