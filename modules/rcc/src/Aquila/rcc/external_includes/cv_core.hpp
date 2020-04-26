#pragma once
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "cv_cudev.hpp"
#include "cv_hal.hpp"
#include "cv_link_config.hpp"
#include <opencv2/core.hpp>

#ifdef _WIN32
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_core" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_core" CV_VERSION_ ".lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")
#endif
