#pragma once
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "cv_link_config.hpp"
#include <opencv2/video.hpp>
#ifdef _WIN32
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_video" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_video" CV_VERSION_ ".lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_video")
#endif
