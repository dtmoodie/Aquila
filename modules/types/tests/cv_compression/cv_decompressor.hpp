#ifndef AQ_TYPES_CV_DECOMPRESSOR_HPP
#define AQ_TYPES_CV_DECOMPRESSOR_HPP

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/IImageCompressor.hpp>
#include <aq_cv_compression_export.hpp>
namespace aq
{

    struct aq_cv_compression_EXPORT OpenCVDecompressor : public aq::IImageDecompressor
    {
        static int32_t priority(aq::ImageEncoding);

        MO_DERIVE(OpenCVDecompressor, aq::IImageDecompressor)

        MO_END;

        void decompressImpl(const aq::CompressedImage& compressed, aq::SyncedImage& out) const;

        void decompressImpl(std::shared_ptr<aq::CompressedImage> compressed, aq::SyncedImage& out) const;
    };
} // namespace aq
#endif // AQ_TYPES_CV_DECOMPRESSOR_HPP