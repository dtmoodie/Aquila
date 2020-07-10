#include "cv_compressor.hpp"
#include "CvMatAllocatorWrapper.hpp"
#include <opencv2/imgcodecs.hpp>
namespace aq
{
    int32_t OpenCVCompressor::priority(aq::ImageEncoding)
    {
        return 1;
    }

    void
    OpenCVCompressor::compressImpl(const aq::SyncedImage& image, aq::CompressedImage& out, aq::ImageEncoding enc) const
    {
        auto mat = image.mat();

        std::stringstream ss;
        ss << ".";
        ss << enc;
        // Figure out how to reduce the need for this later
        std::vector<uint8_t> tmp;
        std::string ext = ss.str();
        if (cv::imencode(ext, mat, tmp, {}))
        {
            out = std::move(tmp);
        }
    }
} // namespace aq

using namespace aq;
MO_REGISTER_OBJECT(OpenCVCompressor)