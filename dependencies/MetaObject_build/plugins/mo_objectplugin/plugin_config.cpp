// clang-format off
#include "mo_objectplugin/mo_objectplugin_export.hpp"
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

const char* getPluginBuildInfo()
{
    return "mo_objectplugin\n"
           "  Compiler         : GNU\n"
           "    version        : 5.4.0\n"
           "  Build date       : 2021-10-22 17:44\n"
           "   branch       : \n"
           "   hash         : \n"
           "  Aquila version   : \n"
           "  Aquila branch    : \n"
           "  Aquila hash      : \n"
           "  MetaObject branch: \n"
           "  MetaObject hash  : \n"
           "  cxx flags        :  -g -fno-omit-frame-pointer\n"
#ifdef _DEBUG
           "  debug flags      : -g -D_DEBUG\n"
#else
           "  release flags    : -O3 -DNDEBUG\n"
#endif
           "  CUDA VERSION     : 9.0\n"
           "  builder          :  - \n";
}

void initPlugin(const int32_t id, mo::MetaObjectFactory* factory)
{
    mo_objectplugin::initPlugin(id, factory);
}

namespace mo_objectplugin
{
    void initPlugin(const int32_t id, mo::MetaObjectFactory* factory)
    {
        static bool initialized = false;
        if(!initialized)
        {
            mo::PluginCompilationOptions options;
            options.includes = getPluginIncludes();
            options.link_dirs_debug = getPluginLinkDirsDebug();
            options.link_dirs_release = getPluginLinkDirsRelease();
            options.compile_options = getPluginCompileOptions();
            options.compile_definitions = getPluginCompileDefinitions();
            options.link_libs = getPluginLinkLibs();
            options.link_libs_debug = getPluginLinkLibsDebug();
            options.link_libs_release = getPluginLinkLibsRelease();
            options.compiler = getCompiler();
            PerModuleInterface::GetInstance()->SetProjectIdForAllConstructors(id);
            factory->setupPluginCompilationOptions(id, options);
            factory->setupObjectConstructors(PerModuleInterface::GetInstance());
            initialized = true;
        }
    }

}

// clang-format on
