#ifndef mo_objectplugin_EXPORT_HPP
#define mo_objectplugin_EXPORT_HPP
// clang-format off
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined mo_objectplugin_EXPORTS
#  define mo_objectplugin_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define mo_objectplugin_EXPORT __attribute__ ((visibility ("default")))
#else
#  define mo_objectplugin_EXPORT
#endif

#if _WIN32
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#include <RuntimeObjectSystem/RuntimeLinkLibrary.h>

#ifdef WIN32
#ifdef _DEBUG
  RUNTIME_COMPILER_LINKLIBRARY("mo_objectplugind.lib")
#else
  RUNTIME_COMPILER_LINKLIBRARY("mo_objectplugin.lib")
#endif
#else // Unix
#ifdef NDEBUG
  RUNTIME_COMPILER_LINKLIBRARY("-lmo_objectplugin")
#else
  RUNTIME_COMPILER_LINKLIBRARY("-lmo_objectplugind")
#endif
#endif


namespace mo
{
    class MetaObjectFactory;
}

extern "C" mo_objectplugin_EXPORT const char* getPluginBuildInfo();
extern "C" mo_objectplugin_EXPORT void initPlugin(const int32_t id, mo::MetaObjectFactory* factory);

namespace mo_objectplugin
{
    const char** getPluginIncludes();
    const char** getPluginLinkDirsDebug();
    const char** getPluginLinkDirsRelease();
    const char** getPluginCompileOptions();
    const char** getPluginCompileDefinitions();
    const char** getPluginLinkLibs();
    const char** getPluginLinkLibsDebug();
    const char** getPluginLinkLibsRelease();
    const char* getCompiler();
    int mo_objectplugin_EXPORT getPluginProjectId();
    
    void mo_objectplugin_EXPORT initPlugin(const int32_t id, mo::MetaObjectFactory* factory);
}

// clang-format on

#endif // mo_objectplugin_EXPORT_HPP
