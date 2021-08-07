#include "common.hpp"
#include "objects.hpp"
#include "gtest/gtest.h"

struct BranchingFixture : public ::testing::Test
{
    BranchingFixture()
    {
    }
    virtual void SetUp()
    {
        init();
    }
    virtual void TearDown()
    {
    }
    void init()
    {
        graph = graph.create();
        graph->stop();
        a = a.create();
        b = b.create();
        c = c.create();
        a->setGraph(graph);
    }

    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_b> b;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IGraph> graph;
};
using namespace mo;
static const std::vector<std::pair<mo::BufferFlags, bool>> settings = {{mo::BufferFlags::MAP_BUFFER, true},
                                                                       {mo::BufferFlags::STREAM_BUFFER, true},
                                                                       {mo::BufferFlags::BLOCKING_STREAM_BUFFER, true},
                                                                       {mo::BufferFlags::NEAREST_NEIGHBOR_BUFFER, true}
#ifdef TEST_FRAME_NUMBER
                                                                       ,
                                                                       {CircularParamFlags::kBUFFER, false},
                                                                       {Map_e, false},
                                                                       {StreamParamFlags::kBUFFER, false},
                                                                       {BlockingStreamParamFlags::kBUFFER, false},
                                                                       {NNStreamParamFlags::kBUFFER, false}
#endif
};

/*     a
 *     | \
 *     |  \
 *     |  b
 *     | /
 *     c
 */
TEST_F(BranchingFixture, direct_ts)
{
    init();
    auto stream = graph->getStream();
    ASSERT_NE(stream, nullptr);
    timestamp_mode = true;
    a->addChild(b);
    EXPECT_EQ(c->connectInput("in_a", a.get(), "out_a"), true);
    EXPECT_EQ(c->connectInput("in_b", b.get(), "out_b"), true);
    for (int i = 0; i < 100; ++i)
    {
        a->process(*stream);
        ASSERT_EQ(a->iterations, i + 1);
        ASSERT_EQ(b->iterations, i + 1);
        ASSERT_EQ(c->iterations, i + 1);
        auto dataa = a->out_a.getData();
        auto datab = b->out_b.getData();
        ASSERT_TRUE(dataa);
        ASSERT_TRUE(datab);
        auto ptra = dataa->template ptr<int>();
        auto ptrb = datab->template ptr<int>();
        ASSERT_TRUE(ptra);
        ASSERT_TRUE(ptrb);
        EXPECT_EQ(*ptra, i);
        EXPECT_EQ(*ptrb, i);
        EXPECT_EQ(c->sum, *ptra + *ptrb);
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
        EXPECT_EQ(a->iterations, i + 1);
        EXPECT_EQ(b->iterations, i + 1);
        EXPECT_EQ(c->iterations, i + 1);
        EXPECT_EQ(a->out_a, i);
        EXPECT_EQ(b->out_b, i);
        EXPECT_EQ(c->sum, a->out_a + b->out_b);
    }
}
#endif

TEST_F(BranchingFixture, buffered)
{
    for (const auto& param : settings)
    {
        init();
        auto stream = this->graph->getStream();
        ASSERT_NE(stream, nullptr);
        /*auto sink = std::make_shared<spdlog::sinks::stdout_sink_st>();
        auto logger = std::make_shared<spdlog::logger>("logger", sink);
        logger->set_level(spdlog::level::level_enum::trace);
        c->setLogger(logger);*/
        timestamp_mode = param.second;
        a->addChild(b);
        std::cout << "Buffer: " << mo::bufferFlagsToString(param.first) << " ts: " << (timestamp_mode ? "on" : "off")
                  << std::endl;
        EXPECT_EQ(
            c->connectInput("in_a", a.get(), "out_a", mo::BufferFlags(mo::BufferFlags::FORCE_BUFFERED | param.first)),
            true);
        EXPECT_EQ(
            c->connectInput("in_b", b.get(), "out_b", mo::BufferFlags(mo::BufferFlags::FORCE_BUFFERED | param.first)),
            true);
        for (int i = 0; i < 1000; ++i)
        {
            a->process(*stream);
            EXPECT_EQ(a->iterations, i + 1);
            EXPECT_EQ(b->iterations, i + 1);
            EXPECT_EQ(c->iterations, i + 1);
            auto dataa = a->out_a.getData();
            auto datab = b->out_b.getData();
            ASSERT_TRUE(dataa);
            ASSERT_TRUE(datab);
            auto ptra = dataa->ptr<int>();
            auto ptrb = datab->ptr<int>();
            ASSERT_EQ(*ptra, i);
            ASSERT_EQ(*ptrb, i);
            ASSERT_EQ(c->sum, *ptra + *ptrb);
        }
    }
}

/*    a     b
 *     |    /
 *     |   /
 *     |  /
 *     | /
 *     c
 */
TEST_F(BranchingFixture, merging_direct_ts)
{
    init();
    timestamp_mode = true;
    b->setGraph(graph);
    auto stream = graph->getStream();
    EXPECT_EQ(c->connectInput("in_a", a.get(), "out_a"), true);
    EXPECT_EQ(c->connectInput("in_b", b.get(), "out_b"), true);
    for (int i = 0; i < 100; ++i)
    {
        a->process(*stream);
        EXPECT_EQ(a->iterations, i + 1);
        EXPECT_EQ(b->iterations, i);
        EXPECT_EQ(c->iterations, i);
        b->process(*stream);
        EXPECT_EQ(b->iterations, i + 1);
        EXPECT_EQ(c->iterations, i + 1);

        auto dataa = a->out_a.getData();
        auto datab = b->out_b.getData();
        ASSERT_TRUE(dataa);
        ASSERT_TRUE(datab);

        auto ptra = dataa->ptr<int>();
        auto ptrb = datab->ptr<int>();

        EXPECT_EQ(c->sum, *ptra + *ptrb);
    }
}
#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(merging_direct_fn)
{
    timestamp_mode = false;
    b->setGraph(ds.get());
    BOOST_REQUIRE(c->connectInput(a, "out_a", "in_a"));
    BOOST_REQUIRE(c->connectInput(b, "out_b", "in_b"));
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        EXPECT_EQ(a->iterations, i + 1);
        EXPECT_EQ(b->iterations, i);
        EXPECT_EQ(c->iterations, i);
        b->process();
        EXPECT_EQ(b->iterations, i + 1);
        EXPECT_EQ(c->iterations, i + 1);
        EXPECT_EQ(c->sum, a->out_a + b->out_b);
    }
}
#endif

TEST_F(BranchingFixture, merging_direct_desynced_ts)
{
    timestamp_mode = true;
    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_e> e;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IGraph> ds;
    ds = ds.create();
    ds->stop();
    a = a.create();
    e = e.create();
    c = c.create();
    a->setGraph(ds);

    e->setGraph(ds);
    ASSERT_TRUE(c->connectInput(
        "in_a", a.get(), "out_a", mo::BufferFlags(mo::BufferFlags::FORCE_BUFFERED | mo::BufferFlags::MAP_BUFFER)));
    ASSERT_TRUE(c->connectInput("in_b", e.get(), "out"));
    auto input = c->getInput("in_b");
    ASSERT_TRUE(input);
    input->appendFlags(mo::ParamFlags::kDESYNCED);

    int e_iters = 0;
    int c_iters = -1;
    auto stream = graph->getStream();
    ASSERT_NE(stream, nullptr);
    for (int i = 0; i < 100; ++i)
    {
        a->process(*stream);
        ASSERT_EQ(a->iterations, i + 1);
        ASSERT_EQ(e->iterations, e_iters);
        ++c_iters;

        if (i % 2 == 0)
        {
            e->process(*stream);
            ++e_iters;
            ++c_iters;
            ASSERT_EQ(e->iterations, e_iters);
        }
        // c should process everytime a is processed even though e is out of sync since
        // an update to a may still be relavent
        ASSERT_EQ(c->iterations, c_iters);
    }
}

#if TEST_FRAME_NUMBER
BOOST_AUTO_TEST_CASE(merging_direct_desynced_fn)
{
    timestamp_mode = false;
    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_e> e;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IGraph> ds;
    ds = ds.create();
    ds->stopThread();
    a = a.create();
    e = e.create();
    c = c.create();
    a->setGraph(ds.get());

    e->setGraph(ds.get());
    BOOST_REQUIRE(
        c->connectInput(a, "out_a", "in_a", mo::BufferFlags(mo::FORCE_BUFFERED | mo::CircularParamFlags::kBUFFER)));
    BOOST_REQUIRE(c->connectInput(e, "out", "in_b"));
    int e_iters = 0;
    for (int i = 0; i < 100; ++i)
    {
        a->process();
        EXPECT_EQ(a->iterations, i + 1);
        EXPECT_EQ(e->iterations, e_iters);
        EXPECT_EQ(c->iterations, e_iters);
        if (i % 2 == 0)
        {
            e->process();
            ++e_iters;
            EXPECT_EQ(e->iterations, e_iters);
            EXPECT_EQ(c->iterations, e_iters);
        }
    }
}
#endif
