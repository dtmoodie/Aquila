#include <Aquila/types/DetectionDescription.hpp>

#include <Aquila/types/ObjectDetection.hpp>

#include <Aquila/types/EntityComponentSystem.hpp>

#include <MetaObject/serialization/JSONPrinter.hpp>

#include <ct/extensions/DataTable.hpp>
#include <ct/reflect/print.hpp>
#include <ct/static_asserts.hpp>

#include <gtest/gtest.h>

#include <iostream>

TEST(object_detection, detected_object)
{
    std::vector<std::string> cat_names({"cat0", "cat1", "cat2", "cat3", "cat4", "cat5"});
    aq::CategorySet::ConstPtr cats = std::make_shared<aq::CategorySet>(cat_names);

    auto cls = (*cats)[0]();
    cls = (*cats)[0](0.95);

    {
        aq::DetectedObject det;
    }
    {
        aq::DetectedObject det({}, cls);
    }

    aq::DetectedObjectSet dets(cats);
    dets.push_back(aq::DetectedObject(cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f), cls, 5));
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
    ct::StaticEqualTypes<Bases_t, ct::VariadicTypedef<aq::TEntityComponentSystem<aq::DetectedObject>>>{};
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

TEST(object_detection, detection_descriptor)
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

        set.push_back(bb, aq::detection::Descriptor(descriptor_holder.data(), descriptor_holder.size()));
    }
    std::cout << set << std::endl;
    auto descriptors = set.getComponent<aq::detection::Descriptor>();
}