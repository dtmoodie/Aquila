// clang-format off
#ifndef AQUILA_EXPORT
    #define AQUILA_EXPORT

    #if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(Aquila_EXPORTS)
        #define AQUILA_EXPORTS __declspec(dllexport)
    #elif (defined(__GNUC__) && (__GNUC__ >= 4))
        #define AQUILA_EXPORTS __attribute__((visibility("default")))
    #else
        #define AQUILA_EXPORTS
    #endif

    #ifndef AQUILA_EXPORTS
        #define AQUILA_EXPORTS
    #endif

    #ifdef _WIN32
        #pragma warning(disable : 4251)
        #pragma warning(disable : 4275)
    #endif

    #if !defined(Aquila_EXPORTS)
        #include "RuntimeObjectSystem/RuntimeLinkLibrary.h"

        #if defined(_WIN32)
            #pragma comment(lib, "Advapi32.lib")
            #if defined(_DEBUG)
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompilerd.lib")
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystemd.lib")
            #else
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeCompiler.lib")
                RUNTIME_COMPILER_LINKLIBRARY("RuntimeObjectSystem.lib")
            #endif // _DEBUG
        #else // UNIX
            #if defined(NDEBUG)
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompiler")
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystem")
            #else
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeCompilerd")
                RUNTIME_COMPILER_LINKLIBRARY("-lRuntimeObjectSystemd")
            #endif
        #endif
    #endif // Aquila_EXPORTS

#endif // AQUILA_EXPORT

#ifdef AQUILA_MODULE
    #ifndef Aquila_EXPORTS
        #ifdef _WIN32
            #ifdef _DEBUG
                RUNTIME_COMPILER_LINKLIBRARY("aquila_" AQUILA_MODULE ".lib")
            #else
                RUNTIME_COMPILER_LINKLIBRARY("aquila_" AQUILA_MODULE ".lib")
            #endif
        #else // Unix
            #ifdef NDEBUG
                RUNTIME_COMPILER_LINKLIBRARY("-laquila_" AQUILA_MODULE)
            #else
                RUNTIME_COMPILER_LINKLIBRARY("-laquila_" AQUILA_MODULE "d")
            #endif
        #endif // WIN32
    #endif // Aquila_EXPORTS
#endif // AQUILA_MODULE

// clang-format on
