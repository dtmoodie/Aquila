#pragma once
#include <Aquila/types/TSyncedMemory.hpp>

#include <Aquila/types/ObjectDetection.hpp>

namespace aq
{

    struct AQUILA_EXPORTS DetectionDescription : public DetectedObject
    {
    };

    struct AQUILA_EXPORTS DetectionDescriptionPatch : public DetectionDescription
    {
    };

    struct AQUILA_EXPORTS LandmarkDetection : public DetectedObject
    {
        LandmarkDetection() = default;
        LandmarkDetection(const DetectedObject&);
    };

    using DetectionDescriptionSet = TDetectedObjectSet<DetectionDescription>;
    using DetectionDescriptionPatchSet = TDetectedObjectSet<DetectionDescriptionPatch>;
    using LandmarkDetectionSet = TDetectedObjectSet<LandmarkDetection>;
} // namespace aq

namespace ct
{
    /*REFLECT_DERIVED(aq::DetectionDescription, aq::DetectedObject)
        //PUBLIC_ACCESS(descriptor)
    REFLECT_END;

    REFLECT_DERIVED(aq::DetectionDescriptionPatch, aq::DetectedObject)
        //PUBLIC_ACCESS(aligned_patch)
    REFLECT_END;

    REFLECT_DERIVED(aq::LandmarkDetection, aq::DetectedObject)
        //PUBLIC_ACCESS(landmark_keypoints)
    REFLECT_END;*/
} // namespace ct
