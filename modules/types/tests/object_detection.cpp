#include <Aquila/types/DetectionDescription.hpp>

#include <Aquila/types/ObjectDetection.hpp>

#include <Aquila/types/EntityComponentSystem.hpp>

#include <MetaObject/serialization/JSONPrinter.hpp>

#include <ct/reflect/print.hpp>
#include <ct/static_asserts.hpp>
#include <ctext/DataTable.hpp>

#include <gtest/gtest.h>

#include <iostream>

TEST(object_detection, detected_object)
{
    std::vector<std::string> cat_names({"cat0", "cat1", "cat2", "cat3", "cat4", "cat5"});
    aq::CategorySet::ConstPtr cats = std::make_shared<aq::CategorySet>(cat_names);

    auto cls = (*cats)[0]();
    cls = (*cats)[0](0.95);
    std::vector<aq::Classification> classes{cls};
    {
        aq::DetectedObject det;
    }
    {
        aq::DetectedObject det(cv::Rect2f(), classes);
    }

    aq::DetectedObjectSet dets(cats);
    dets.push_back(aq::DetectedObject(cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f), classes, 5));
}

TEST(object_detection, detected_object_data_table)
{
    ct::ext::DataTable<aq::DetectedObject> table;
}

TEST(object_detection, reflect_object)
{
    std::stringstream ss;
    ct::printStructInfo<aq::DetectedObject>(ss);
    std::cout << ss.str() << std::endl;
}

TEST(object_detection, reflect_ecs)
{
    using Bases_t = typename ct::ReflectImpl<aq::DetectedObjectSet>::BaseTypes;
    ct::StaticEqualTypes<Bases_t, ct::VariadicTypedef<aq::EntityComponentSystem>>{};
    std::vector<std::string> cat_names({"cat0", "cat1", "cat2", "cat3", "cat4", "cat5"});
    aq::CategorySet::ConstPtr cats = std::make_shared<aq::CategorySet>(cat_names);
    aq::DetectedObjectSet set(cats);
    std::stringstream ss;
    ct::printStructInfo<aq::DetectedObjectSet>(ss);
    std::cout << ss.str() << std::endl;
    std::cout << set << std::endl;
}

TEST(object_detection, serialize_ecs)
{
    std::vector<std::string> cat_names({"cat0", "cat1", "cat2", "cat3", "cat4", "cat5"});
    aq::CategorySet::ConstPtr cats = std::make_shared<aq::CategorySet>(cat_names);
    aq::DetectedObjectSet set(cats);
    std::stringstream ss;
    std::stringstream ss1;
    {
        mo::JSONSaver saver(ss1);
        saver(&set, "objects");
    }

    aq::DetectedObjectSet loaded;
    mo::JSONLoader loader(ss1);
    loader(&loaded, "objects");
}

aq::DetectedObjectSet makeDescriptorSet()
{
    std::vector<std::string> cat_names({"cat0", "cat1", "cat2", "cat3", "cat4", "cat5"});
    aq::CategorySet::ConstPtr cats = std::make_shared<aq::CategorySet>(cat_names);
    using Components_t = ct::VariadicTypedef<aq::detection::BoundingBox2d, aq::detection::Descriptor>;
    aq::TDetectedObjectSet<Components_t> set(cats);

    std::vector<float> descriptor_holder(20);

    for (size_t i = 0; i < 10; ++i)
    {
        aq::detection::BoundingBox2d bb(0.0, 0.1 * i, 0.2 * i, 0.3 * i);
        std::transform(descriptor_holder.begin(), descriptor_holder.end(), descriptor_holder.begin(), [](float) {
            return std::rand() / RAND_MAX;
        });

        set.pushComponents(bb, aq::detection::Descriptor(descriptor_holder.data(), descriptor_holder.size()));
    }
    return set;
}

TEST(object_detection, detection_descriptor)
{
    aq::DetectedObjectSet set = makeDescriptorSet();
    std::cout << set << std::endl;
    auto descriptors = set.getComponent<aq::detection::Descriptor>();
    EXPECT_EQ(descriptors.getShape()[0], 10);
    EXPECT_EQ(descriptors.getShape()[1], 20);
}

TEST(object_detection, untyped_publish_subscribe)
{
    std::shared_ptr<mo::TPublisher<aq::DetectedObjectSet>> pub =
        std::make_shared<mo::TPublisher<aq::DetectedObjectSet>>();

    mo::TSubscriber<aq::DetectedObjectSet> sub;
    auto types = pub->getOutputTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_TRUE(sub.acceptsType(types[0]));
    EXPECT_TRUE(sub.acceptsPublisher(*pub));
    EXPECT_TRUE(static_cast<mo::ISubscriber&>(sub).setInput(std::shared_ptr<mo::IPublisher>(pub)));
    EXPECT_TRUE(sub.setInput(pub.get()));

    aq::DetectedObjectSet set = makeDescriptorSet();
    pub->publish(std::move(set));

    auto data = sub.getData();
    EXPECT_TRUE(data);
    auto tdata = sub.getTypedData();
    EXPECT_TRUE(tdata);
}

TEST(object_detection, typed_publish_subscribe)
{
    using Type = aq::TDetectedObjectSet<ct::VariadicTypedef<aq::detection::BoundingBox2d>>;
    std::shared_ptr<mo::TPublisher<Type>> pub = std::make_shared<mo::TPublisher<Type>>();

    mo::TSubscriber<aq::DetectedObjectSet> sub;
    auto types = pub->getOutputTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_TRUE(sub.acceptsType(types[0]));
    EXPECT_TRUE(sub.acceptsPublisher(*pub));
    EXPECT_TRUE(static_cast<mo::ISubscriber&>(sub).setInput(std::shared_ptr<mo::IPublisher>(pub)));
    EXPECT_TRUE(sub.setInput(pub.get()));

    aq::DetectedObjectSet set = makeDescriptorSet();
    pub->publish(std::move(set));

    auto data = sub.getData();
    EXPECT_TRUE(data);
    auto tdata = sub.getTypedData();
    EXPECT_TRUE(tdata);
}

TEST(object_detection, publish_typed_subscribe)
{
    using Type = aq::TDetectedObjectSet<ct::VariadicTypedef<aq::detection::BoundingBox2d>>;
    std::shared_ptr<mo::TPublisher<Type>> pub = std::make_shared<mo::TPublisher<Type>>();

    mo::TSubscriber<aq::DetectedObjectSet> sub;
    auto types = pub->getOutputTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_TRUE(sub.acceptsType(types[0]));
    EXPECT_TRUE(sub.acceptsPublisher(*pub));
    EXPECT_TRUE(static_cast<mo::ISubscriber&>(sub).setInput(std::shared_ptr<mo::IPublisher>(pub)));
    EXPECT_TRUE(sub.setInput(pub.get()));

    aq::DetectedObjectSet set = makeDescriptorSet();
    pub->publish(std::move(set));

    auto data = sub.getData();
    EXPECT_TRUE(data);
    auto tdata = sub.getTypedData();
    EXPECT_TRUE(tdata);
}

TEST(object_detection, copy_components)
{
    aq::DetectedObjectSet set = makeDescriptorSet();

    aq::TDetectedObjectSet<ct::VariadicTypedef<aq::detection::Classifications>> cat_set = set;

    auto component = cat_set.getComponent<aq::detection::Classifications>();
    EXPECT_EQ(component.getShape()[0], 10);
}
