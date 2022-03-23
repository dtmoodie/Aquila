#pragma once
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <Aquila/detail/export.hpp>
#include <opencv2/core.hpp>
#ifdef _WIN32
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("aquila_utilitiesd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("aquila_utilities.lib")
#endif
#else
#ifdef NDEBUG
RUNTIME_COMPILER_LINKLIBRARY("-laquila_utilities")
#else
RUNTIME_COMPILER_LINKLIBRARY("-laquila_utilitiesd")
#endif
#endif

namespace cv
{
    namespace cuda
    {
        AQUILA_EXPORTS void rectangle(
            cv::cuda::GpuMat& img, const cv::Rect& rect, cv::Scalar color, int thickness, cv::cuda::Stream& stream);
    }
} // namespace cv
