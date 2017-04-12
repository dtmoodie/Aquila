#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "Aquila/IDataStream.hpp"
#include "Aquila/Nodes/Node.h"
#include "Aquila/Nodes/ThreadedNode.h"
#include "Aquila/Nodes/NodeInfo.hpp"

#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
#include "MetaObject/Detail/Allocator.hpp"

#include "Aquila/Detail/AlgorithmImpl.hpp"
#include "Aquila/Logging.h"
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
using namespace aq::Nodes;
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

struct node_a: public Nodes::Node
{
    MO_DERIVE(node_a, Nodes::Node)
        OUTPUT(int, out_a, 0)
    MO_END;

    bool ProcessImpl()
    {
        if(timestamp_mode == true)
        {
            out_a_param.UpdateData(iterations, mo::time_t(mo::ms * iterations));
        }else
        {
            out_a_param.UpdateData(iterations, mo::tag::_frame_number = iterations);
        }

        _modified = true;
        ++iterations;
        return true;
    }
    int iterations = 0;
};

struct node_b: public Nodes::Node
{
    MO_DERIVE(node_b, Nodes::Node)
        OUTPUT(int, out_b, 0)
    MO_END;

    bool ProcessImpl()
    {
        if(timestamp_mode == true)
        {
            out_b_param.UpdateData(iterations, mo::time_t(mo::ms * iterations));
        }else
        {
            out_b_param.UpdateData(iterations, mo::tag::_frame_number = iterations);
        }

        _modified = true;
        ++iterations;
        return true;
    }
    int iterations = 0;
};


struct node_c: public Nodes::Node
{
    MO_DERIVE(node_c, Nodes::Node)
        INPUT(int, in_a, nullptr)
        INPUT(int, in_b, nullptr)
    MO_END;

    bool ProcessImpl()
    {
        BOOST_REQUIRE_EQUAL(*in_a, *in_b);

        sum = *in_a + *in_b;
        ++iterations;
        return true;
    }
    void check_timestamps()
    {
        Algorithm::impl* impl = _pimpl;
        LOG(debug) << impl->_ts_processing_queue.size() << " frames left to process";
    }
    int sum = 0;
    int iterations = 0;
};

struct node_d : public Nodes::Node
{
    MO_DERIVE(node_d, Nodes::Node)
        INPUT(int, in_d, nullptr)
        OUTPUT(int, out_d, 0)
    MO_END;
    bool ProcessImpl()
    {
        if(timestamp_mode == true)
        {
            BOOST_REQUIRE_EQUAL(mo::time_t(*in_d * mo::ms), *in_d_param.GetTimestamp());
            out_d_param.UpdateData(*in_d, *in_d_param.GetTimestamp());
        }else
        {
            BOOST_REQUIRE_EQUAL(*in_d, in_d_param.GetFrameNumber());
            out_d_param.UpdateData(*in_d, in_d_param.GetFrameNumber());
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
        aq::SetupLogging();

        mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
        boost::filesystem::path currentDir = boost::filesystem::path("./").parent_path();
    #ifdef _MSC_VER
        currentDir = boost::filesystem::path("./");
    #else
        currentDir = boost::filesystem::path("./Plugins");
    #endif
        LOG(info) << "Looking for plugins in: " << currentDir.string();
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
                        mo::MetaObjectFactory::Instance()->LoadPlugin(file);
                    }
                }
            }
        }
        g_allocator = mo::Allocator::GetThreadSafeAllocator();
        cv::cuda::GpuMat::setDefaultAllocator(g_allocator);
        cv::Mat::setDefaultAllocator(g_allocator);
        g_allocator->SetName("Global Allocator");
        mo::GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(g_allocator);
        mo::CpuThreadAllocatorSetter<cv::Mat>::Set(g_allocator);
        //boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::trace);
    }
    ~GlobalFixture()
    {
        mo::ThreadPool::Instance()->Cleanup();
        mo::ThreadSpecificQueue::Cleanup();
        mo::Allocator::CleanupThreadSpecificAllocator();
        delete g_allocator;
    }
    mo::Allocator* g_allocator;
};
BOOST_GLOBAL_FIXTURE(GlobalFixture);



struct BranchingFixture
{
    BranchingFixture()
    {
        ds = ds.Create();
        ds->StopThread();
        a = a.Create();
        b = b.Create();
        c = c.Create();
        a->SetDataStream(ds.Get());
    }

    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_b> b;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IDataStream> ds;
};
using namespace mo;
static const std::pair<mo::ParameterTypeFlags, bool> settings[] =
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
    a->AddChild(b);
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b"));
    for(int i = 0; i < 100; ++i)
    {
        a->Process();
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
    a->AddChild(b);
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->Process();
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
    a->AddChild(b);
    std::cout << "Buffer: " << mo::ParameterTypeFlagsToString(param.first) << " ts: " << (timestamp_mode ? "on" : "off" )<< std::endl;
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    for (int i = 0; i < 1000; ++i)
    {
        a->Process();
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
    b->SetDataStream(ds.Get());
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->Process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        b->Process();
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}
#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(merging_direct_fn)
{
    timestamp_mode = false;
    b->SetDataStream(ds.Get());
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->Process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        b->Process();
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}
#endif


BOOST_AUTO_TEST_SUITE_END()

struct node_e: public Nodes::Node
{
    MO_DERIVE(node_e, Nodes::Node)
        OUTPUT(int, out, 0)
    MO_END;
    bool ProcessImpl()
    {
        if(timestamp_mode)
        {
            out_param.UpdateData(iterations*2, mo::time_t(mo::ms * iterations*2));
        }else
        {
            out_param.UpdateData(iterations* 2, mo::tag::_frame_number = iterations*2);
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
    ds = ds.Create();
    ds->StopThread();
    a = a.Create();
    e = e.Create();
    c = c.Create();
    a->SetDataStream(ds.Get());

    e->SetDataStream(ds.Get());
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | mo::CircularBuffer_e)));
    BOOST_REQUIRE(c->ConnectInput(e, "out", "in_b"));
    int e_iters = 0;
    for (int i = 0; i < 100; ++i)
    {
        a->Process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(e->iterations, e_iters);
        BOOST_REQUIRE_EQUAL(c->iterations, e_iters);
        if (i % 2 == 0)
        {
            e->Process();
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
    ds = ds.Create();
    ds->StopThread();
    a = a.Create();
    e = e.Create();
    c = c.Create();
    a->SetDataStream(ds.Get());

    e->SetDataStream(ds.Get());
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | mo::CircularBuffer_e)));
    BOOST_REQUIRE(c->ConnectInput(e, "out", "in_b"));
    int e_iters = 0;
    for (int i = 0; i < 100; ++i)
    {
        a->Process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(e->iterations, e_iters);
        BOOST_REQUIRE_EQUAL(c->iterations, e_iters);
        if (i % 2 == 0)
        {
            e->Process();
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
        ds = ds.Create();
        ds->StopThread();
        a = a.Create();
        d1 = d1.Create();
        d2 = d1.Create();
        c = c.Create();
        a->SetDataStream(ds.Get());
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
    BOOST_REQUIRE(d1->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->ConnectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->Process();
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
    BOOST_REQUIRE(d1->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->ConnectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->Process();
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
    BOOST_REQUIRE(d1->ConnectInput(a, "out_a", "in_d", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(d2->ConnectInput(a, "out_a", "in_d", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->ConnectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->Process();
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
    std::cout << "Buffer: " << mo::ParameterTypeFlagsToString(param.first)
              << " sync: " << (timestamp_mode ? "timestamp" : "framenumber") << std::endl;
    BOOST_REQUIRE(d1->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->ConnectInput(d1, "out_d", "in_a", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->ConnectInput(d2, "out_d", "in_b", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->Process();
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
    std::cout << "Buffer: " << mo::ParameterTypeFlagsToString(param.first) << " ts: " << (timestamp_mode ? "on" : "off") << std::endl;
    BOOST_REQUIRE(d1->ConnectInput(a, "out_a", "in_d", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(d2->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(c->ConnectInput(d1, "out_d", "in_a", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->ConnectInput(d2, "out_d", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->Process();
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
    BOOST_REQUIRE(d1->ConnectInput(a, "out_a", "in_d"));
    BOOST_REQUIRE(d2->ConnectInput(a, "out_a", "in_d", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    BOOST_REQUIRE(c->ConnectInput(d1, "out_d", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(d2, "out_d", "in_b", mo::ParameterTypeFlags(mo::ForceBufferedConnection_e | param.first)));
    for (int i = 0; i < 100; ++i)
    {
        BOOST_REQUIRE_EQUAL(a->iterations, i);
        BOOST_REQUIRE_EQUAL(d1->iterations, i);
        BOOST_REQUIRE_EQUAL(d2->iterations, i);
        BOOST_REQUIRE_EQUAL(c->iterations, i);
        a->Process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d1->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(d2->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + a->out_a);
    }
}

BOOST_AUTO_TEST_SUITE_END() // DiamondSuite

struct mt_a: public aq::Nodes::Node
{
    MO_DERIVE(mt_a, aq::Nodes::Node)
        OUTPUT(int, out_a, 0)
        APPEND_FLAGS(out_a, mo::Source_e)
    MO_END;

    bool ProcessImpl()
    {
        producer_ready = true;
        if(!consumer_ready)
        {
            return false;
        }
        if(timestamp_mode == true)
        {
            out_a_param.UpdateData(iterations, mo::time_t(mo::ms * iterations));
        }else
        {
            out_a_param.UpdateData(iterations, mo::tag::_frame_number = iterations);
        }
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        _modified = true;
        ++iterations;
        aq::Nodes::Node* This = this;
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
        a = a.Create();
        d = d.Create();
        ds = ds.Create();
        ds->AddNode(a);
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
    d->SetContext(mo::Context::GetDefaultThreadContext());
    BOOST_REQUIRE(d->ConnectInput(a, "out_a", "in_d"));
    a->consumer_ready = true;
    Nodes::Node* ptr = a.Get();
    a->sig_node_updated(ptr);
    for(int i = 0; i < 1000; ++i)
    {
        d->Process();
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
    std::cout << a->iterations << " " << d->iterations << std::endl;
}

BOOST_AUTO_TEST_SUITE_END() // ThreadedSuite

BOOST_AUTO_TEST_CASE(finish)
{
    mo::ThreadSpecificQueue::Cleanup();
    mo::ThreadPool::Instance()->Cleanup();
    mo::Allocator::CleanupThreadSpecificAllocator();
}
