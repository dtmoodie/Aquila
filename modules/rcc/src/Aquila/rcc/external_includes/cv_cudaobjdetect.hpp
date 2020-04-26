#pragma once
#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "cv_link_config.hpp"
#include "opencv2/cudaobjdetect.hpp"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaobjdetect" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaobjdetect" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaobjdetect")
#define CALL
#endif
#endif