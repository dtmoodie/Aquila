#include "objects.hpp"
#include "common.hpp"
#include "gtest/gtest.h"

bool node_a::processImpl()
{
    if (timestamp_mode)
    {
        out_a.publish(iterations, mo::Time(mo::ms * iterations));
    }
    else
    {
        out_a.publish(iterations, mo::tags::fn = iterations);
    }
    setModified();
    ++iterations;
    return true;
}

bool node_b::processImpl()
{
    if (timestamp_mode)
    {
        out_b.publish(iterations, mo::Time(mo::ms * iterations));
    }
    else
    {
        out_b.publish(iterations, mo::tags::fn = iterations);
    }
    setModified();
    ++iterations;
    return true;
}

bool node_c::processImpl()
{
    auto ts = this->getTimestamp();
    EXPECT_TRUE(ts) << "We expect to have a timestamp for the data that we are processing, but did not receive one";
    if (!ts)
    {
        return false;
    }
    EXPECT_NE(in_a, nullptr);
    EXPECT_NE(in_b, nullptr);
    if (!in_b_param.checkFlags(mo::ParamFlags::kDESYNCED))
    {
        EXPECT_EQ(*in_a, *in_b);
    }

    sum = *in_a + *in_b;
    ++iterations;
    return true;
}

void node_c::check_timestamps()
{
    // Algorithm::impl* impl = _algorithm_pimpl.get();
    // MO_LOG(debug) << impl->_ts_processing_queue.size() << " frames left to process";
}

bool node_d::processImpl()
{
    if (timestamp_mode)
    {
        auto ts = in_d_param.getNewestTimestamp();
        EXPECT_TRUE(ts);
        EXPECT_EQ(mo::Time(*in_d * mo::ms), *ts);
        out_d.publish(*in_d, *in_d_param.getNewestTimestamp());
    }
    else
    {
        auto hdr = in_d_param.getNewestHeader();
        EXPECT_TRUE(hdr);
        EXPECT_EQ(*in_d, hdr->frame_number);
        out_d.publish(*in_d, hdr->frame_number);
    }

    ++iterations;
    return true;
}

bool node_e::processImpl()
{
    if (timestamp_mode)
    {
        out.publish(iterations * 2, mo::Time(mo::ms * (iterations * 2)));
    }
    else
    {
        out.publish(iterations * 2, mo::tags::fn = iterations * 2);
    }
    setModified();
    ++iterations;
    return true;
}

MO_REGISTER_CLASS(node_a)
MO_REGISTER_CLASS(node_b)
MO_REGISTER_CLASS(node_c)
MO_REGISTER_CLASS(node_d)
MO_REGISTER_CLASS(node_e)
