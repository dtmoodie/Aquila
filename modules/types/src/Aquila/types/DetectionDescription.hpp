#ifndef AQ_TYPES_DETECTION_DESCRIPTION_HPP
#define AQ_TYPES_DETECTION_DESCRIPTION_HPP
#include <Aquila/types/ObjectDetection.hpp>

#include <Aquila/types/TSyncedMemory.hpp>

namespace aq
{
    namespace detection
    {
        using Descriptor = ct::TArrayView<float>;
    } // namespace detection

} // namespace aq

#endif // AQ_TYPES_DETECTION_DESCRIPTION_HPP