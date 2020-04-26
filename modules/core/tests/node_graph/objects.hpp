#include "common.hpp"

using namespace aq;

struct node_a : public nodes::Node
{
    MO_DERIVE(node_a, nodes::Node)
        OUTPUT(int, out_a)
    MO_END;

    bool processImpl() override;
    int iterations = 0;
};

struct node_b : public nodes::Node
{
    MO_DERIVE(node_b, nodes::Node)
        OUTPUT(int, out_b)
    MO_END;

    bool processImpl() override;
    int iterations = 0;
};

struct node_c : public nodes::Node
{
    MO_DERIVE(node_c, nodes::Node)
        INPUT(int, in_a)
        INPUT(int, in_b)
    MO_END;

    bool processImpl() override;
    void check_timestamps();
    int sum = 0;
    int iterations = 0;
};

struct node_d : public nodes::Node
{
    MO_DERIVE(node_d, nodes::Node)
        INPUT(int, in_d)
        OUTPUT(int, out_d)
    MO_END;

    bool processImpl() override;
    int iterations = 0;
};

struct node_e : public nodes::Node
{
    MO_DERIVE(node_e, nodes::Node)
        OUTPUT(int, out)
    MO_END;

    bool processImpl() override;
    int iterations = 0;
};
