#pragma once
#include "ObjectDetection.hpp"
#include "SyncedImage.hpp"

namespace aq
{
    namespace detection
    {
        struct Patch
        {
            using DType = Patch;
            static const constexpr uint8_t D = 0;

            aq::SyncedImage source_image;
            aq::SyncedImage patch;
        };

        struct AlignedPatch
        {
            using DType = AlignedPatch;
            static const constexpr uint8_t D = 0;

            aq::SyncedImage source_image;
            aq::SyncedImage aligned_patch;
        };
    } // namespace detection

} // namespace aq

namespace ct
{
    REFLECT_BEGIN(aq::detection::Patch)
        PUBLIC_ACCESS(source_image)
        PUBLIC_ACCESS(patch)
    REFLECT_END;

    REFLECT_BEGIN(aq::detection::AlignedPatch)
        PUBLIC_ACCESS(source_image)
        PUBLIC_ACCESS(aligned_patch)
    REFLECT_END;
} // namespace ct
