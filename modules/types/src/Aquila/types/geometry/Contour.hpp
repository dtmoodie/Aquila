#ifndef AQUILA_CONTOUR_HPP
#define AQUILA_CONTOUR_HPP
#include <opencv2/core/types.hpp>

#include <ct/reflect.hpp>
#include <ct/types/opencv.hpp>

#include <Aquila/types/EntityComponentSystem.hpp>

#include <vector>
namespace aq
{
    struct Contour_;
    using Contour = aq::ArrayComponent<cv::Point, Contour_>;
} // namespace aq

#endif // AQUILA_CONTOUR_HPP
