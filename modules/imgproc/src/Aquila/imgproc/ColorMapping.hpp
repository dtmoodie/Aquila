#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <Aquila/rcc/external_includes/cv_core.hpp>
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>
#include <functional>
#include <map>

namespace aq
{
    void AQUILA_EXPORTS createColormap(cv::Mat& lut, int num_classes, int ignore_class = -1);
}
