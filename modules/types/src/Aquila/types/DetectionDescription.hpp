#ifndef AQ_TYPES_DETECTION_DESCRIPTION_HPP
#define AQ_TYPES_DETECTION_DESCRIPTION_HPP
#include <Aquila/types/ObjectDetection.hpp>

#include <Aquila/types/TSyncedMemory.hpp>

namespace aq
{
    namespace detection
    {
        // using Descriptor = ct::TArrayView<float>;
        struct Descriptor_;
        using Descriptor = ArrayComponent<Descriptor_, float>;

        struct LandmarkDetection_;

        using LandmarkDetection = ArrayComponent<LandmarkDetection_, cv::Point2f>;
    } // namespace detection

} // namespace aq

#endif // AQ_TYPES_DETECTION_DESCRIPTION_HPP
