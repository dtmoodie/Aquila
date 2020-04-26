#include "gtest/gtest.h"

#include "common.hpp"
#include "objects.hpp"

using namespace aq;
using namespace aq::nodes;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto table = SystemTable::instance();
    table->getDefaultLogger()->set_level(spdlog::level::critical);
    auto factory = mo::MetaObjectFactory::instance();
    factory->registerTranslationUnit();
    boost::filesystem::path current_dir = boost::filesystem::path("./").parent_path();
#ifdef _MSC_VER
    current_dir = boost::filesystem::path("./");
#else
    current_dir = boost::filesystem::path("./Plugins");
#endif
    MO_LOG(debug, "Looking for plugins in: {}", current_dir.string());
    if (boost::filesystem::exists(current_dir))
    {
        factory->loadPlugins(current_dir.string());
    }

    auto allocator = table->getDefaultAllocator();

    auto cpu_proxy = std::make_shared<mo::CvAllocatorProxy>(allocator.get());

    // cv::cuda::GpuMat::setDefaultAllocator(&cuda_proxy);
    cv::Mat::setDefaultAllocator(cpu_proxy.get());
    allocator->setName("Global Allocator");
    mo::params::init(table.get());

    auto result = RUN_ALL_TESTS();

    return result;
}
