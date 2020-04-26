#include "objects.hpp"
#include "gtest/gtest.h"

TEST(pubsub, direct)
{
    auto graph = aq::IGraph::create();
    auto output_node = test_output_node::create();
    auto input_node = test_input_node::create();
    EXPECT_NE(output_node, nullptr);
    EXPECT_NE(input_node, nullptr);
    EXPECT_NE(graph, nullptr) << "Unable to create a graph";
    output_node->setGraph(graph);
    EXPECT_EQ(input_node->connectInput("value", output_node.get(), "value"), true);
    for (int i = 0; i < 10; ++i)
    {
        output_node->process();
    }
    EXPECT_EQ(output_node->process_count, 10);
    EXPECT_EQ(input_node->process_count, 10);
}

TEST(pubsub, buffered)
{
    auto graph = aq::IGraph::create();
    auto output_node = test_output_node::create();
    auto input_node = test_input_node::create();
    EXPECT_NE(output_node, nullptr);
    EXPECT_NE(input_node, nullptr);
    output_node->setGraph(graph);
    input_node->setGraph(graph);
    static const std::vector<mo::BufferFlags> test_cases{mo::BufferFlags::MAP_BUFFER,
                                                         mo::BufferFlags::STREAM_BUFFER,
                                                         mo::BufferFlags::BLOCKING_STREAM_BUFFER,
                                                         mo::BufferFlags::DROPPING_STREAM_BUFFER,
                                                         mo::BufferFlags::NEAREST_NEIGHBOR_BUFFER};
    for (auto test_case : test_cases)
    {
        std::cout << "Buffer type: " << mo::bufferFlagsToString(test_case) << std::endl;
        output_node->process_count = 0;
        input_node->process_count = 0;
        output_node->timestamp = 0;
        EXPECT_EQ(input_node->connectInput("value", output_node.get(), "value", test_case), true);

        for (int i = 0; i < 10; ++i)
        {
            output_node->process();
        }
        EXPECT_EQ(output_node->process_count, 10);
        EXPECT_EQ(input_node->process_count, 10);
    }
}
