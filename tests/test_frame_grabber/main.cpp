#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "Aquila/nodes/Node.hpp"
#include <Aquila/types/SyncedMemory.hpp>
#include "Aquila/nodes/ThreadedNode.hpp"
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/framegrabbers/FrameGrabberInfo.hpp"
#include "Aquila/core/Logging.hpp"

#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "AquilaFrameGrabbers"
#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>


using namespace aq;
using namespace aq::Nodes;

struct test_framegrabber: public IFrameGrabber
{
    bool processImpl()
    {
        current.create(128,128, CV_8U);
        ++ts;
        current.setTo(ts);
        current_frame_param.updateData(current.clone(), ts, _ctx.get());
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
        OUTPUT(SyncedMemory, current_frame, {})
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

struct img_node: public Node
{
    MO_DERIVE(img_node, Node)
        INPUT(SyncedMemory, input, nullptr)
    MO_END;

    bool processImpl()
    {
        BOOST_REQUIRE(input);
        auto mat = input->getMat(stream());
        BOOST_REQUIRE_EQUAL(mat.at<uchar>(0), (*input_param.getTimestamp()).value());
        return true;
    }
};

MO_REGISTER_CLASS(test_framegrabber);
MO_REGISTER_CLASS(img_node);

BOOST_AUTO_TEST_CASE(test_dummy_output)
{
    aq::SetupLogging();
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    mo::MetaObjectFactory::instance()->loadPlugins("");
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("test_framegrabber");
    BOOST_REQUIRE(info);
    auto fg_info = dynamic_cast<aq::Nodes::FrameGrabberInfo*>(info);
    BOOST_REQUIRE(fg_info);
    std::cout << fg_info->Print();

    auto fg = rcc::shared_ptr<test_framegrabber>::create();
    auto node = rcc::shared_ptr<img_node>::create();
    BOOST_REQUIRE(node->connectInput(fg, "input", "current_frame"));
    for(int i = 0; i < 100; ++i)
    {
        fg->process();
    }
}
BOOST_AUTO_TEST_CASE(test_enumeration)
{
    //auto all_docs = aq::Nodes::IFrameGrabber::ListAllLoadableDocuments();
    std::cout << mo::MetaObjectFactory::instance()->printAllObjectInfo();
}
