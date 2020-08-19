#pragma once
#include <MetaObject/core/metaobject_config.hpp>

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "cv_link_config.hpp"
#include "opencv2/cudaarithm.hpp"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaarithm" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaarithm" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")
#define CALL
#endif
