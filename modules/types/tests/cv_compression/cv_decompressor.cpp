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
        MO_ASSERT(!compressed.empty());
        std::shared_ptr<aq::SyncedImage> out = std::make_shared<aq::SyncedImage>(ret);

        ce::shared_ptr<const SyncedMemory> compressed_data = compressed.getData();
        // OpenCV's api doesn't allow wrapping of const data, even though the decompress operation is a const
        // operation
        ct::TArrayView<const void> host_view = compressed_data->host();
        const uint8_t* ptr = ct::ptrCast<uint8_t>(host_view.data());

        cv::Mat wrap_data(host_view.size(), 1, CV_8UC1, const_cast<uint8_t*>(ptr));

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
