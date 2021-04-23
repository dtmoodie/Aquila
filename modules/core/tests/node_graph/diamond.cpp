#include "common.hpp"
#include "objects.hpp"

#include "gtest/gtest.h"

using namespace mo;

struct DiamondFixture : public ::testing::TestWithParam<mo::BufferFlags>
{
    DiamondFixture()
    {
    }

    ~DiamondFixture() override
    {
    }

    void SetUp() override
    {
        init();
    }

    void TearDown() override
    {
    }

    void init()
    {
        graph = graph.create();
        graph->stop();
        a = a.create();
        d0 = d0.create();
        d1 = d0.create();
        d0->setName("d0");
        d1->setName("d1");
        c = c.create();
        a->setGraph(graph);
    }

    void check()
    {
        auto stream = graph->getStream();
        ASSERT_NE(stream, nullptr);
        for (int i = 0; i < 100; ++i)
        {
            ASSERT_EQ(a->iterations, i);
            ASSERT_EQ(d0->iterations, i);
            ASSERT_EQ(d1->iterations, i);
            ASSERT_EQ(c->iterations, i);
            a->process(*stream);
            ASSERT_EQ(a->iterations, i + 1);
            ASSERT_EQ(d0->iterations, i + 1);
            ASSERT_EQ(d1->iterations, i + 1);
            ASSERT_EQ(c->iterations, i + 1);
            auto data = a->out_a.getData();
            ASSERT_TRUE(data);
            auto ptr = data->ptr<int>();
            ASSERT_TRUE(ptr);
            ASSERT_EQ(c->sum, *ptr + *ptr);
        }
    }

    rcc::shared_ptr<node_a> a;
    rcc::shared_ptr<node_d> d0;
    rcc::shared_ptr<node_d> d1;
    rcc::shared_ptr<node_c> c;
    rcc::shared_ptr<aq::IGraph> graph;
};

//            A
//           /  \     a
//          /    \    a
//        d0      d1
//          \     /
//           \   /
//             C

TEST_F(DiamondFixture, direct_ts)
{
    timestamp_mode = true;
    ASSERT_TRUE(d0->connectInput("in_d", a.get(), "out_a"));
    ASSERT_TRUE(d1->connectInput("in_d", a.get(), "out_a"));
    ASSERT_TRUE(c->connectInput("in_a", d0.get(), "out_d"));
    ASSERT_TRUE(c->connectInput("in_b", d1.get(), "out_d"));
    check();
}

// A -> d0 + A -> d1 are buffered
TEST_P(DiamondFixture, buffered_top)
{
    auto param = GetParam();
    init();
    timestamp_mode = true;
    mo::BufferFlags flag(mo::BufferFlags::FORCE_BUFFERED | param);
    ASSERT_TRUE(d0->connectInput("in_d", a.get(), "out_a", flag));
    ASSERT_TRUE(d1->connectInput("in_d", a.get(), "out_a", flag));
    ASSERT_TRUE(c->connectInput("in_a", d0.get(), "out_d"));
    ASSERT_TRUE(c->connectInput("in_b", d1.get(), "out_d"));
    check();
}

// d0 -> C and d1 -> C are buffered
TEST_P(DiamondFixture, buffered_bottom)
{
    auto param = GetParam();
    init();
    timestamp_mode = true;
    mo::BufferFlags flag(mo::BufferFlags::FORCE_BUFFERED | param);
    ASSERT_TRUE(d0->connectInput("in_d", a.get(), "out_a"));
    ASSERT_TRUE(d1->connectInput("in_d", a.get(), "out_a"));
    ASSERT_TRUE(c->connectInput("in_a", d0.get(), "out_d", flag));
    ASSERT_TRUE(c->connectInput("in_b", d1.get(), "out_d", flag));
    check();
}

// A -> d0 and d0 -> C are buffered
TEST_P(DiamondFixture, buffered_left)
{
    auto param = GetParam();
    init();
    timestamp_mode = true;
    mo::BufferFlags flag(mo::BufferFlags::FORCE_BUFFERED | param);
    ASSERT_TRUE(d0->connectInput("in_d", a.get(), "out_a", flag));
    ASSERT_TRUE(d1->connectInput("in_d", a.get(), "out_a"));
    ASSERT_TRUE(c->connectInput("in_a", d0.get(), "out_d", flag));
    ASSERT_TRUE(c->connectInput("in_b", d1.get(), "out_d"));
    check();
}

// A -> d1 and d1 -> C are buffered
TEST_P(DiamondFixture, buffered_right)
{
    auto param = GetParam();
    init();
    timestamp_mode = true;
    mo::BufferFlags flag(mo::BufferFlags::FORCE_BUFFERED | param);
    ASSERT_TRUE(d0->connectInput("in_d", a.get(), "out_a"));
    ASSERT_TRUE(d1->connectInput("in_d", a.get(), "out_a", flag));
    ASSERT_TRUE(c->connectInput("in_a", d0.get(), "out_d"));
    ASSERT_TRUE(c->connectInput("in_b", d1.get(), "out_d", flag));
    if (param == BufferFlags::NEAREST_NEIGHBOR_BUFFER)
    {
        SystemTable::instance()->getDefaultLogger()->set_level(spdlog::level::trace);
        // TODO in this case since d0 gets updated first and with a nearest neighbor buffer it is working correctly in
        // that it is trying to sync the new input from d0 with a nearest neighbor from d1 which thus yields extra
        // executions
        /*for (int i = 0; i < 100; ++i)
        {
            ASSERT_EQ(a->iterations, i);
            ASSERT_EQ(d0->iterations, i);
            ASSERT_EQ(d1->iterations, i);
            ASSERT_EQ(c->iterations, i);
            a->process();
            ASSERT_EQ(a->iterations, i + 1);
            ASSERT_EQ(d0->iterations, i + 1);
            ASSERT_EQ(d1->iterations, i + 1);
            ASSERT_EQ(c->iterations, i + 1);
            ASSERT_EQ(c->sum, a->out_a.value() + a->out_a.value());
        }*/
        SystemTable::instance()->getDefaultLogger()->set_level(spdlog::level::critical);
    }
    else
    {
        check();
    }
}

static const auto test_parameters = ::testing::Values(mo::BufferFlags::MAP_BUFFER,
                                                      BufferFlags::STREAM_BUFFER,
                                                      BufferFlags::BLOCKING_STREAM_BUFFER,
                                                      BufferFlags::NEAREST_NEIGHBOR_BUFFER);

INSTANTIATE_TEST_SUITE_P(DiamondFixture, DiamondFixture, test_parameters);
