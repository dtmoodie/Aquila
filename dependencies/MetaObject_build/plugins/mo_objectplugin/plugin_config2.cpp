#include "mo_objectplugin_export.hpp"

namespace mo_objectplugin
{
const char** getPluginIncludes()
{
    static const char* paths[] = {
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/object/tests/plugin",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/object/src/",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/plugins/",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/plugins/mo_objectplugin/",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/object/src",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/core/src",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/include",
        "/home/dan/code/boost",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/RuntimeObjectSystem",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/rcc/Aurora/RuntimeCompiler",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/spdlog/include",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/include",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/include",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/params/src",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/types/src",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/modules/runtime_reflection/src",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/googletest/googletest/include",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject/dependencies/ct/dependencies/minitensor/googletest/googletest",
        nullptr
    };
    return paths;
}

const char** getPluginLinkDirsDebug()
{
    static const char* paths[] = {
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/bin/plugins",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build",
        "/home/dan/code/boost/stage/lib",
        "/usr/local/cuda/lib64",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/lib",
        nullptr
    };
    return paths;
}

const char** getPluginLinkDirsRelease()
{
    static const char* paths[] = {
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/bin/plugins",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build",
        "/home/dan/code/boost/stage/lib",
        "/usr/local/cuda/lib64",
        "/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/lib",
        nullptr
    };
    return paths;
}

const char** getPluginCompileOptions()
{
    static const char* paths[] = {
        "-std=c++14 ",
        "-g ",
        "-fno-omit-frame-pointer ",
        nullptr
    };
    return paths;
}

const char** getPluginCompileDefinitions()
{
    static const char* paths[] = {
        "-DMetaObject_EXPORTS ",
        "-DBOOST_NO_AUTO_PTR ",
        "-Dmo_objectplugin_EXPORTS ",
        nullptr
    };
    return paths;
}

const char** getPluginLinkLibs()
{
    static const char* paths[] = {
        "-lpthread ",
        "-lboost_thread ",
        "-lboost_chrono ",
        "-lboost_date_time ",
        "-lboost_atomic ",
        "-lboost_fiber ",
        "-lboost_context ",
        "-lboost_system ",
        "-lboost_filesystem ",
        "-lcudart ",
        "-lboost_filesystem ",
        nullptr
    };
    return paths;
}

const char** getPluginLinkLibsDebug()
{
    static const char* paths[] = {
        "-lmo_objectplugind ",
        "-lmetaobject_objectd ",
        "-lmetaobject_cored ",
        "-lRuntimeObjectSystemd ",
        "-lRuntimeCompilerd ",
        "-lmetaobject_paramsd ",
        "-lmetaobject_typesd ",
        "-lmetaobject_runtime_reflectiond ",
        "-lgtestd ",
        "-lgtest_maind ",
        nullptr
    };
    return paths;
}

const char** getPluginLinkLibsRelease()
{
    static const char* paths[] = {
        "-lmo_objectplugin ",
        "-lmetaobject_object ",
        "-lmetaobject_core ",
        "-lRuntimeObjectSystem ",
        "-lRuntimeCompiler ",
        "-lmetaobject_params ",
        "-lmetaobject_types ",
        "-lmetaobject_runtime_reflection ",
        "-lgtest ",
        nullptr
    };
    return paths;
}

int getPluginProjectId()
{
    return -1;
}

const char* getCompiler()
{
    return "/usr/bin/g++-5;/usr/local/cuda/bin/nvcc";
}
} // namespace mo_objectplugin
