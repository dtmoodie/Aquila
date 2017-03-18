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

#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

using namespace aq;
using namespace aq::Nodes;

struct node_a: public Nodes::Node
{
    MO_BEGIN(node_a)
        OUTPUT(int, out_a, 0);
    MO_END;

    bool ProcessImpl()
    {
        ++ts;
        out_a_param.UpdateData(ts, ts);
        _modified = true;
        ++iterations;
        return true;
    }
    int ts = 0;
    int iterations = 0;
};

struct node_b: public Nodes::Node
{
    MO_BEGIN(node_b)
        OUTPUT(int, out_b, 0);
    MO_END;

    bool ProcessImpl()
    {
        ++ts;
        out_b_param.UpdateData(ts, ts);
        _modified = true;
        ++iterations;
        return true;
    }
    int ts = 0;
    int iterations = 0;
};

struct node_c: public Nodes::Node
{
    MO_BEGIN(node_c)
        INPUT(int, in_a, nullptr);
        INPUT(int, in_b, nullptr);
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

MO_REGISTER_CLASS(node_a)
MO_REGISTER_CLASS(node_b)
MO_REGISTER_CLASS(node_c)

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


/*     a
 *     | \
 *     |  \
 *     |  b
 *     | /
 *     c
*/
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

BOOST_FIXTURE_TEST_SUITE(suite, BranchingFixture)

BOOST_AUTO_TEST_CASE(branching_direct)
{
    a->AddChild(b);
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b"));
    for(int i = 0; i < 100; ++i)
    {
        a->Process();
        BOOST_REQUIRE_EQUAL(a->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(b->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(c->iterations, i + 1);
        BOOST_REQUIRE_EQUAL(a->out_a, i + 1);
        BOOST_REQUIRE_EQUAL(b->out_b, i + 1);
        BOOST_REQUIRE_EQUAL(c->sum, a->out_a + b->out_b);
    }
}

BOOST_AUTO_TEST_CASE(merging_direct)
{
    BOOST_REQUIRE(c->ConnectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->ConnectInput(b, "out_b", "in_b"));
    for(int i = 0; i < 100; ++i)
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

BOOST_AUTO_TEST_SUITE_END()






BOOST_AUTO_TEST_CASE(diamond)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b1 = rcc::shared_ptr<node_b>::Create();
    auto b2 = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    // B1 and B2 don't have inputs, thus for them to be a child of A, we need to manually add them
    a->AddChild(b1);
    a->AddChild(b2);
    // C is added as a child of B1 and B2 here
    BOOST_REQUIRE(c->ConnectInput(b1, "in_a", "out_b"));
    BOOST_REQUIRE(c->ConnectInput(b2, "in_b", "out_b"));
    // Calling A, calls B1, then B2.  When B1 Process() is called, it tries to call C->Process()
    // Since C's inputs aren't ready yet, it is not called, but when B2 Process is called, C->Process() does get
    // called since B2 finished prepping the inputs for C.

    a->Process();
    BOOST_REQUIRE_EQUAL(c->sum, 2);
}

BOOST_AUTO_TEST_CASE(threaded_child_sync_every)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();

    auto thread = aq::Nodes::ThreadedNode::Create();
    mo::Context ctx;
    a->SetContext(&ctx);
    b->SetContext(&ctx);
    thread->AddChild(c);
    BOOST_REQUIRE(c->ConnectInput(b, "in_b", "out_b"));
    BOOST_REQUIRE(c->ConnectInput(a, "in_a", "out_a"));

    c->SetSyncInput("in_b");
    int sum = 0;

    a->Process();
    b->Process();
    sum += a->out_a;
    sum += b->out_b;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    BOOST_REQUIRE_EQUAL(c->sum, 2);
    
    for(int i = 0; i < 100; ++i)
    {
        a->Process();
        b->Process();
        sum += a->out_a;
        sum += b->out_b;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    thread->StopThread();
    if(c->sum != sum)
    {
        c->check_timestamps();
    }
    std::cout << "Dropped " << 1.0 -  double(c->sum) / (double)sum << " % of data\n";
    BOOST_REQUIRE_EQUAL(c->sum, sum);
}

BOOST_AUTO_TEST_CASE(threaded_child_sync_newest)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
    auto thread = rcc::shared_ptr<aq::Nodes::ThreadedNode>::Create();
    mo::Context ctx;
    a->SetContext(&ctx);
    b->SetContext(&ctx);
    thread->AddChild(c);
    BOOST_REQUIRE(c->ConnectInput(b, "in_b", "out_b"));
    BOOST_REQUIRE(c->ConnectInput(a, "in_a", "out_a"));

    c->SetSyncInput("in_b");
    c->SetSyncMethod(Algorithm::SyncNewest);
    int sum = 0;

    a->Process();
    b->Process();
    sum += a->out_a;
    sum += b->out_b;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    BOOST_REQUIRE_EQUAL(c->sum, 2);

    for(int i = 0; i < 100; ++i)
    {
        a->Process();
        b->Process();
        sum += a->out_a;
        sum += b->out_b;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    thread->StopThread();
    BOOST_REQUIRE_LE(c->sum, sum);
    std::cout << "Dropped " << 1.0 -  double(c->sum) / (double)sum << " % of data\n";
}


BOOST_AUTO_TEST_CASE(threaded_parent)
{
    auto a = rcc::shared_ptr<node_a>::Create();
    auto b = rcc::shared_ptr<node_b>::Create();
    auto c = rcc::shared_ptr<node_c>::Create();
}

