#pragma once
/*
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "opencv2/hal.hpp"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_hal" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_hal" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_hal")
#define CALL
#endif
*/