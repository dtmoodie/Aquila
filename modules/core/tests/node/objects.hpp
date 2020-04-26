#pragma once
#include "common.hpp"

using namespace aq::nodes;

struct test_node : public Node
{
    static std::vector<std::string> getNodeCategory();

    bool processImpl() override;

    MO_DERIVE(test_node, Node)
        MO_SLOT(void, node_slot, int)
        MO_SIGNAL(void, node_signal, int)
        PARAM(int, test_param, 5)
    MO_END;
};

struct test_output_node : public Node
{
    MO_DERIVE(test_output_node, Node)
        SOURCE(int, value, 0)
    MO_END;
    bool processImpl() override;

    int timestamp = 0;
    int process_count = 0;
};

struct test_input_node : public Node
{
    MO_DERIVE(test_input_node, Node)
        INPUT(int, value)
    MO_END;

    bool processImpl() override;
    int process_count = 0;
};

struct test_multi_input_node : public Node
{
    MO_DERIVE(test_multi_input_node, Node)
        INPUT(int, value1)
        INPUT(int, value2)
    MO_END;
    bool processImpl() override;
};
