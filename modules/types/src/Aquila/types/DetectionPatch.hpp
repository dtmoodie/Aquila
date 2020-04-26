#pragma once
#include "ObjectDetection.hpp"
#include "SyncedMemory.hpp"

namespace aq
{
    struct DetectionPatch
    {
        DetectedObject detection;
        aq::SyncedMemory source_image;
        aq::SyncedMemory patch;
    };
}

namespace ct
{
    REFLECT_BEGIN(aq::DetectionPatch)
        PUBLIC_ACCESS(detection)
        PUBLIC_ACCESS(source_image)
        PUBLIC_ACCESS(patch)
    REFLECT_END;
}
