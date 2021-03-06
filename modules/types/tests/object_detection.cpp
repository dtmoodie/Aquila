#include <ct/extensions/DataTable.hpp>
#include <ct/reflect/print.hpp>


#include <Aquila/types/ObjectDetection.hpp>

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
    dets.emplace_back(cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f), cls, 5);
}

TEST(object_detection, detected_object_data_table)
{
    ct::ext::DataTable<aq::DetectedObject> table;
}

TEST(object_detection, reflect)
{
    std::stringstream ss;
    ct::printStructInfo<aq::DetectedObject>(ss);
    std::cout << ss.str() << std::endl;
}