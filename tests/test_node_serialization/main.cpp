#define BOOST_TEST_MAIN

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/core/Graph.hpp>
#include <Aquila/core/IGraph.hpp>

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <opencv2/core.hpp>

#include <fstream>
#include <iostream>

using namespace mo;

// MO_REGISTER_OBJECT(serializable_object);
BOOST_AUTO_TEST_CASE(initialize)
{
    boost::filesystem::path currentDir = boost::filesystem::current_path();
#ifdef _MSC_VER
#ifdef _DEBUG
    currentDir = boost::filesystem::path(currentDir.string() + "/../Debug/");
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/../RelWithDebInfo/");
#endif
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");
#endif
    MO_LOG(info, "Looking for plugins in: {}", currentDir.string());
    boost::filesystem::directory_iterator end_itr;
    if (boost::filesystem::is_directory(currentDir))
    {
        for (boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
#ifdef _MSC_VER
                if (itr->path().extension() == ".dll")
#else
                if (itr->path().extension() == ".so")
#endif
                {
                    std::string file = itr->path().string();
                    mo::MetaObjectFactory::instance()->loadPlugin(file);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(synced_mem_to_json)
{
    aq::SyncedImage synced_mem(cv::Mat(320, 240, CV_32FC3));

    mo::TParamPtr<aq::SyncedImage> param;
    param.setName("Matrix");
    param.updatePtr(&synced_mem);

    // TODO reimplement test for json serialization using dynamic visitation
    std::ofstream ofs("synced_memory_json.json");
    BOOST_REQUIRE(ofs.is_open());
    mo::JSONSaver saver(ofs);
    param.save(saver);
}

BOOST_AUTO_TEST_CASE(Graph)
{
    auto ds = aq::IGraph::create("", "TestFrameGrabber");
    std::ofstream ofs("Graph.json");
    BOOST_REQUIRE(ofs.is_open());

    /*aq::JSONOutputArchive ar(ofs);
    ds->addNode("QtImageDisplay");
    auto disp = ds->getNode("QtImageDisplay0");
    auto fg = ds->getNode("TestFrameGrabber0");
    disp->connectInput(fg, "current_frame", "image");
    ar(ds);*/
}

BOOST_AUTO_TEST_CASE(read_Graph)
{
    rcc::shared_ptr<aq::IGraph> stream = rcc::shared_ptr<aq::Graph>::create();
    std::ifstream ifs("Graph.json");
    BOOST_REQUIRE(ifs.is_open());
    /*std::map<std::string, std::string> dummy;
    aq::JSONInputArchive ar(ifs, dummy, dummy);
    ar(stream);*/
}
