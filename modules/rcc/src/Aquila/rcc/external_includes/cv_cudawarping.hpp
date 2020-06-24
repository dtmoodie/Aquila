#ifndef RCC_EXTERNAL_CV_CUDAWARPING_HPP
#define RCC_EXTERNAL_CV_CUDAWARPING_HPP
#include <MetaObject/core/metaobject_config.hpp>

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "cv_link_config.hpp"
#include "opencv2/cudawarping.hpp"

#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudawarping" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudawarping" CV_VERSION_ ".lib")
#endif

#else // Linux

RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudawarping")
#define CALL

#endif

#endif // RCC_EXTERNAL_CV_CUDAWARPING_HPP