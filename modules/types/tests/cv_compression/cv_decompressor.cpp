#include "cv_decompressor.hpp"
#include "CvMatAllocatorWrapper.hpp"
#include <opencv2/imgcodecs.hpp>
namespace aq
{
    int32_t OpenCVDecompressor::priority(aq::ImageEncoding)
    {
        return 1;
    }

    void OpenCVDecompressor::decompressImpl(const aq::CompressedImage& compressed, aq::SyncedImage& ret) const
    {
        std::shared_ptr<aq::SyncedImage> out = std::make_shared<aq::SyncedImage>(ret);

        auto compressed_data = compressed.data();
        // OpenCV's api doesn't allow wrapping of const data, even though the decompress operation is a const
        // operation
        cv::Mat wrap_data(compressed_data.size(),
                          1,
                          CV_8UC1,
                          const_cast<uint8_t*>(static_cast<const uint8_t*>(compressed_data.data())));
        auto wrapped = wrap(out);
        CvAllocatorContextManager ctx(wrapped.allocator);
        auto img = cv::imdecode(wrap_data, cv::IMREAD_UNCHANGED);
        ret = std::move(*out);
    }

    void OpenCVDecompressor::decompressImpl(std::shared_ptr<aq::CompressedImage> compressed, aq::SyncedImage& ret) const
    {
        decompress(*compressed, ret);
    }
} // namespace aq

using namespace aq;
MO_REGISTER_OBJECT(OpenCVDecompressor)
