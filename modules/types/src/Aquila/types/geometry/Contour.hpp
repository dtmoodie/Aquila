#ifndef AQUILA_CONTOUR_HPP
#define AQUILA_CONTOUR_HPP
#include <opencv2/core/types.hpp>

#include <ct/reflect.hpp>
#include <ct/types/opencv.hpp>

#include <vector>
namespace aq
{
    using Contour = ct::TArrayView<cv::Point>;
} // namespace aq

#endif // AQUILA_CONTOUR_HPP