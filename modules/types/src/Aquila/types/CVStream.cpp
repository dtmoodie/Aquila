#include "CVStream.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <MetaObject/core/AsyncStreamConstructor.hpp>
#include <MetaObject/core/AsyncStreamFactory.hpp>

namespace aq
{
    CVStream::CVStream(cv::cuda::Stream stream, mo::DeviceAllocator::Ptr_t allocator, mo::Allocator::Ptr_t host_alloc)
        : mo::cuda::AsyncStream(
              mo::cuda::Stream(cv::cuda::StreamAccessor::getStream(stream), false), allocator, host_alloc)
        , m_stream(stream)
    {
    }

    cv::cuda::Stream& CVStream::getCVStream()
    {
        return m_stream;
    }

    namespace
    {
        struct CVStreamConstructor : public mo::AsyncStreamConstructor
        {
            CVStreamConstructor()
            {
                SystemTable::dispatchToSystemTable(
                    [this](SystemTable* table) { mo::AsyncStreamFactory::instance(table)->registerConstructor(this); });
            }

            uint32_t priority(int32_t) override
            {
                return 3U;
            }

            Ptr_t
            create(const std::string& name, int32_t, mo::PriorityLevels, mo::PriorityLevels thread_priority) override
            {
                auto stream = std::make_shared<aq::CVStream>();
                stream->setName(name);
                stream->setHostPriority(thread_priority);
                return stream;
            }
        };

        CVStreamConstructor g_ctr;
    } // namespace
} // namespace aq