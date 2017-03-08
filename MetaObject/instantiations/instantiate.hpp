#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined instantiations_EXPORTS
  #define INST_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
  #define INST_EXPORTS __attribute__ ((visibility ("default")))
#else
  #define INST_EXPORTS
#endif

#ifdef _MSC_VER
  #ifndef instantiations_EXPORTS
    #ifdef _DEBUG
      #pragma comment(lib, "instantiationsd.lib")
    #else
      #pragma comment(lib, "instantiations.lib")
    #endif
  #endif
#endif
namespace mo
{
    namespace instantiations
    {
        INST_EXPORTS void initialize();
    }
}
