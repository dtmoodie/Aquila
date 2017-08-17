#pragma once
#include <opencv2/core.hpp>
#include <Aquila/core/detail/Export.hpp>
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("aquila_utilitiesd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("aquila_utilities.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-laquila_utilities")
#endif

namespace cv{
namespace cuda{
	AQUILA_EXPORTS void rectangle(cv::cuda::GpuMat& img, const cv::Rect& rect, cv::Scalar color, int thickness, cv::cuda::Stream& stream);
}
}
