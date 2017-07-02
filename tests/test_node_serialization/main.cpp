#define BOOST_TEST_MAIN
#include <Aquila/core/IDataStream.hpp>
#include <Aquila/serialization/cereal/JsonArchive.hpp>
#include <Aquila/core/DataStream.hpp>

#include <MetaObject/thread/ThreadPool.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/serialization/SerializationFactory.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <Aquila/types/SyncedMemory.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>

using namespace mo;

//MO_REGISTER_OBJECT(serializable_object);
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
    LOG(info) << "Looking for plugins in: " << currentDir.string();
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
    aq::SyncedMemory synced_mem(cv::Mat(320,240, CV_32FC3));
    
    mo::TParamPtr<aq::SyncedMemory> param;
    param.setName("Matrix");
    param.updatePtr(&synced_mem);
    auto func = mo::SerializationFactory::instance()->getJsonSerializationFunction(param.getTypeInfo());
    BOOST_REQUIRE(func);
    std::ofstream ofs("synced_memory_json.json");
    BOOST_REQUIRE(ofs.is_open());
    aq::JSONOutputArchive ar(ofs);
    func(&param,ar);
}

BOOST_AUTO_TEST_CASE(datastream)
{
    auto ds = aq::IDataStream::create("", "TestFrameGrabber");
    std::ofstream ofs("datastream.json");
    BOOST_REQUIRE(ofs.is_open());
    aq::JSONOutputArchive ar(ofs);
    ds->addNode("QtImageDisplay");
    auto disp = ds->getNode("QtImageDisplay0");
    auto fg = ds->getNode("TestFrameGrabber0");
    disp->connectInput(fg, "current_frame", "image");
    ar(ds);
}

BOOST_AUTO_TEST_CASE(read_datastream)
{
    rcc::shared_ptr<aq::IDataStream> stream = rcc::shared_ptr<aq::DataStream>::create();
    std::ifstream ifs("datastream.json");
    BOOST_REQUIRE(ifs.is_open());
    std::map<std::string, std::string> dummy;
    aq::JSONInputArchive ar(ifs, dummy, dummy);
    ar(stream);
}

BOOST_AUTO_TEST_CASE(cleanup_tests)
{
    mo::ThreadPool::Instance()->Cleanup();
}
