#include "objects.hpp"
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/test/object_dynamic_reflection.hpp>
#include <MetaObject/params/ParamTags.hpp>

#include "gtest/gtest.h"

std::vector<std::string> test_node::getNodeCategory()
{
    return {"test1", "test2"};
}

bool test_node::processImpl()
{
    return true;
}

void test_node::node_slot(int)
{
}

bool test_output_node::processImpl()
{
    ++timestamp;
    auto stream = getStream();
    value.publish(timestamp * 10, mo::tags::timestamp = mo::ms * timestamp, mo::tags::stream = stream.get());
    ++process_count;
    // setModified(true);
    return true;
}

bool test_input_node::processImpl()
{
    auto header = value_param.getNewestHeader();
    EXPECT_TRUE(header);
    auto ts = header->timestamp;
    EXPECT_TRUE(ts);
    EXPECT_EQ(*ts, mo::Time((*value / 10) * mo::ms));
    ++process_count;
    return true;
}

bool test_multi_input_node::processImpl()
{
    EXPECT_EQ((*value1), (*value2));
    EXPECT_EQ(mo::Time(mo::ms * (*value1 * 10)), (*value1_param.getNewestTimestamp()));
    EXPECT_EQ(value1_param.getNewestTimestamp(), value2_param.getNewestTimestamp());
    return true;
}

MO_REGISTER_CLASS(test_node)
MO_REGISTER_CLASS(test_output_node)
MO_REGISTER_CLASS(test_input_node)
MO_REGISTER_CLASS(test_multi_input_node)

TEST(dynamic_reflection, test_node)
{
    auto stream = mo::IAsyncStream::create();
    auto inst = test_node::create();
    mo::testDynamicReflection(inst);
}

TEST(dynamic_reflection, output_node)
{
    auto stream = mo::IAsyncStream::create();
    auto inst = test_output_node::create();
    mo::testDynamicReflection(inst);
}

TEST(dynamic_reflection, input_node)
{
    auto stream = mo::IAsyncStream::create();
    auto inst = test_input_node::create();
    mo::testDynamicReflection(inst);
}

TEST(dynamic_reflection, multi_output)
{
    auto stream = mo::IAsyncStream::create();
    auto inst = test_multi_input_node::create();
    mo::testDynamicReflection(inst);
}

TEST(static_reflection, test_node)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("test_node");
    mo::testStaticReflection<test_node>(info);
}

TEST(static_reflection, output_node)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("test_output_node");
    mo::testStaticReflection<test_output_node>(info);
}

TEST(static_reflection, input_node)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("test_input_node");
    mo::testStaticReflection<test_input_node>(info);
}

TEST(static_reflection, multi_output)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("test_multi_input_node");
    mo::testStaticReflection<test_multi_input_node>(info);
}
