#pragma once
#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "cv_link_config.hpp"
#include "opencv2/cudafilters.hpp"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudafilters" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudafilters" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudafilters")
#define CALL
#endif
#endif