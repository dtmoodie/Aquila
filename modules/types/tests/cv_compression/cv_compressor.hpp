#ifndef AQ_TYPES_CV_COMPRESSOR_HPP
#define AQ_TYPES_CV_COMPRESSOR_HPP

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/IImageCompressor.hpp>
#include <aq_cv_compression_export.hpp>
namespace aq
{
    struct aq_cv_compression_EXPORT OpenCVCompressor : aq::IImageCompressor
    {
        static int32_t priority(aq::ImageEncoding);

        MO_DERIVE(OpenCVCompressor, aq::IImageCompressor)

        MO_END;

        void compressImpl(const aq::SyncedImage& image,
                          aq::CompressedImage& compressed,
                          aq::ImageEncoding enc) const override;
    };
} // namespace aq
#endif // AQ_TYPES_CV_COMPRESSOR_HPP