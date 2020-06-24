#ifndef AQUILA_TYPES_CVSTREAM_HPP
#define AQUILA_TYPES_CVSTREAM_HPP
#include <Aquila/core/detail/Export.hpp>

#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/cuda/AsyncStream.hpp>

#include <opencv2/core/cuda.hpp>

namespace aq
{
    struct AQUILA_EXPORTS CVStream : mo::cuda::AsyncStream
    {
        CVStream(cv::cuda::Stream stream = cv::cuda::Stream(),
                 mo::DeviceAllocator::Ptr_t allocator = mo::DeviceAllocator::getDefault(),
                 mo::Allocator::Ptr_t host_alloc = mo::Allocator::getDefault());
        cv::cuda::Stream& getCVStream();

      private:
        cv::cuda::Stream m_stream;
    };
} // namespace aq

#endif // AQUILA_TYPES_CVSTREAM_HPP