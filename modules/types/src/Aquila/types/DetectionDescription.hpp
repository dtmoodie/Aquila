#ifndef AQ_TYPES_DETECTION_DESCRIPTION_HPP
#define AQ_TYPES_DETECTION_DESCRIPTION_HPP
#include <Aquila/types/ObjectDetection.hpp>

#include <Aquila/types/TSyncedMemory.hpp>

namespace aq
{
    namespace detection
    {

        struct Descriptor_;
        using Descriptor = ArrayComponent<float, Descriptor_>;

        struct LandmarkDetection_;
        using LandmarkDetection = ArrayComponent<cv::Point2f, LandmarkDetection_>;
    } // namespace detection

} // namespace aq

#endif // AQ_TYPES_DETECTION_DESCRIPTION_HPP
