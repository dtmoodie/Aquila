#pragma once
#include <opencv2/core.hpp>
#include <Aquila/Detail/Export.hpp>
namespace cv
{
namespace cuda
{
	AQUILA_EXPORTS void rectangle(cv::cuda::GpuMat& img, const cv::Rect& rect, cv::Scalar color, int thickness, cv::cuda::Stream& stream);
}
}
