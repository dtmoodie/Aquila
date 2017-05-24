#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/ThreadedNode.h"
#include "Aquila/nodes/IFrameGrabber.hpp"
#include "Aquila/Logging.h"
#include "Aquila/nodes/FrameGrabberInfo.hpp"

#include "MetaObject/params/ParameterMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
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
    bool ProcessImpl()
    {
        current.create(128,128, CV_8U);
        ++ts;
        current.setTo(ts);
        current_frame_param.UpdateData(current.clone(), ts, _ctx);
        return true;
    }
    bool LoadFile(const std::string&)
    {
        return true;
    }
    long long GetFrameNumber()
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
    
    static int canLoadDocument(const std::string& doc)
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
    MO_DERIVE(img_node, Node);
        INPUT(SyncedMemory, input, nullptr)
    MO_END;

    bool ProcessImpl()
    {
        BOOST_REQUIRE(input);
        auto mat = input->getMat(Stream());
        BOOST_REQUIRE_EQUAL(mat.at<uchar>(0), (*input_param.GetTimestamp()).value());
        return true;
    }
};

MO_REGISTER_CLASS(test_framegrabber);
MO_REGISTER_CLASS(img_node);

BOOST_AUTO_TEST_CASE(test_dummy_output)
{
    aq::SetupLogging();
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    mo::MetaObjectFactory::instance()->LoadPlugins("");
    auto info = mo::MetaObjectFactory::instance()->GetObjectInfo("test_framegrabber");
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
    std::cout << mo::MetaObjectFactory::instance()->PrintAllObjectInfo();
}
