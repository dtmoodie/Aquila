#ifndef META_OBJECT_EXPORT
#define META_OBJECT_EXPORT

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined MetaObject_EXPORTS
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif

#ifndef MO_EXPORTS
#define MO_EXPORTS
#endif

#if _WIN32
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#ifndef MetaObject_EXPORTS
#include "RuntimeLinkLibrary.h"
  #ifdef WIN32
    #pragma comment(lib, "Advapi32.lib")
    #ifdef _DEBUG
      RUNTIME_COMPILER_LINKLIBRARY("MetaObjectd.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompilerd.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystemd.lib")
    #else
      RUNTIME_COMPILER_LINKLIBRARY("MetaObject.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompiler.lib")
      RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystem.lib")
    #endif
  #else // Unix
    #ifdef NDEBUG
      RUNTIME_COMPILER_LINKLIBRARY("-lMetaObject")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompiler")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystem")
    #else
      RUNTIME_COMPILER_LINKLIBRARY("-lMetaObjectd")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompilerd")
      RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystemd")
    #endif
  #endif
#endif

#endif
