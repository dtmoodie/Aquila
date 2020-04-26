#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/ThreadedNode.hpp>

#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "AquilaFrameGrabbers"
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <iostream>

using namespace aq;
using namespace aq::nodes;

struct test_framegrabber : public IFrameGrabber
{
    bool processImpl()
    {
        current.create(128, 128, CV_8U);
        ++ts;
        current.setTo(ts);
        current_frame.updateData(current.clone(), ts);
        return true;
    }
    bool LoadFile(const std::string&)
    {
        return true;
    }
    long long getFrameNumber()
    {
        return ts;
    }
    long long GetNumFrames()
    {
        return 255;
    }

    MO_DERIVE(test_framegrabber, IFrameGrabber)
        OUTPUT(SyncedImage, current_frame, {})
    MO_END;
    int ts = 0;
    cv::Mat current;

    static int canLoadPath(const std::string& doc)
    {
        return 1;
    }
    static int loadTimeout()
    {
        return 1;
    }
};

struct img_node : public Node
{
    MO_DERIVE(img_node, Node)
        INPUT(SyncedImage, input)
    MO_END;

    bool processImpl()
    {
        BOOST_REQUIRE(input);
        auto mat = input->mat();
        // TODO update
        // EXPECT_EQ(mat.at<uchar>(0), (*input_param.getTimestamp()).value());
        return true;
    }
};

MO_REGISTER_CLASS(test_framegrabber);
MO_REGISTER_CLASS(img_node);

BOOST_AUTO_TEST_CASE(test_dummy_output)
{

    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    mo::MetaObjectFactory::instance()->loadPlugins("");
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("test_framegrabber");
    BOOST_REQUIRE(info);
    auto fg_info = dynamic_cast<const aq::nodes::FrameGrabberInfo*>(info);
    BOOST_REQUIRE(fg_info);
    std::cout << fg_info->Print();

    auto fg = rcc::shared_ptr<test_framegrabber>::create();
    auto node = rcc::shared_ptr<img_node>::create();
    BOOST_REQUIRE(node->connectInput("input", fg.get(), "current_frame"));
    for (int i = 0; i < 100; ++i)
    {
        fg->process();
    }
}
BOOST_AUTO_TEST_CASE(test_enumeration)
{
    // auto all_docs = aq::nodes::IFrameGrabber::ListAllLoadableDocuments();
    std::cout << mo::MetaObjectFactory::instance()->printAllObjectInfo();
}
