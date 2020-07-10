#include "CvMatAllocatorWrapper.hpp"
namespace aq
{
    struct CvMatAllocatorWrapper final : public cv::MatAllocator
    {
        CvMatAllocatorWrapper(std::shared_ptr<aq::SyncedImage> image);
        cv::UMatData*
        allocate(int dims, const int* sizes, int type, void*, size_t*, int, cv::UMatUsageFlags) const override;

        bool allocate(cv::UMatData*, int, cv::UMatUsageFlags) const override;

        void deallocate(cv::UMatData* data) const override;

      private:
        // Why did opencv decide the allocate function needed to be const -_-
        mutable std::shared_ptr<aq::SyncedImage> m_image;
    };

    cv::Mat wrap(std::shared_ptr<aq::SyncedImage> image)
    {
        auto ptr = new CvMatAllocatorWrapper(image);
        cv::Mat out;
        out.allocator = ptr;
        return out;
    }

    cv::UMatData*
    CvMatAllocatorWrapper::allocate(int dims, const int* sizes, int type, void*, size_t*, int, cv::UMatUsageFlags) const
    {
        int height, width;
        height = sizes[0];
        if (dims == 2)
        {
            width = sizes[1];
        }
        else
        {
            width = 1;
        }
        const auto channels = CV_MAT_CN(type);
        aq::PixelFormat fmt;
        switch (channels)
        {
        case 1:
            fmt = aq::PixelFormat::kGRAY;
            break;
        case 3:
            fmt = aq::PixelFormat::kRGB;
            break;
        default:
            THROW(warn, "Unsupported number of channels {}", channels);
        }
        aq::DataFlag dtype = aq::fromCvDepth(CV_MAT_DEPTH(type));

        m_image->create(height, width, fmt, dtype);

        auto host_data = m_image->data()->mutableHost();
        auto udata = new cv::UMatData(this);
        udata->data = ct::ptrCast<uchar>(host_data.data());
        udata->size = host_data.size();
        udata->origdata = ct::ptrCast<uchar>(host_data.data());
        return udata;
    }

    bool CvMatAllocatorWrapper::allocate(cv::UMatData*, int, cv::UMatUsageFlags) const
    {
        return false;
    }

    void CvMatAllocatorWrapper::deallocate(cv::UMatData* data) const
    {
        delete data;
        delete this;
    }

    CvMatAllocatorWrapper::CvMatAllocatorWrapper(std::shared_ptr<aq::SyncedImage> image)
        : m_image(image)
    {
    }

    CvAllocatorContextManager::CvAllocatorContextManager(cv::MatAllocator* alloc)
    {
        m_original = cv::Mat::getDefaultAllocator();
        cv::Mat::setDefaultAllocator(alloc);
    }

    CvAllocatorContextManager::~CvAllocatorContextManager()
    {
        cv::Mat::setDefaultAllocator(m_original);
    }

} // namespace aq