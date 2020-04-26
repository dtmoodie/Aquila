#ifndef AQ_TYPES_CV_MAT_ALLOCATOR_WRAPPER
#define AQ_TYPES_CV_MAT_ALLOCATOR_WRAPPER
#include <Aquila/types/SyncedImage.hpp>
#include <MetaObject/core/SystemTable.hpp>
#include <opencv2/core/mat.hpp>

// This allocator allows for a cv::Mat that is backed by a SyncedImage
// TODO: Figure out if OpenCV copies allocators when creating a new Mat. If so, the problem that we have here is that

// Wants:
// 1) Be able to wrap an existing SyncedImage with a cv mat such that allocations are backed by the synced image and
//    that the original owner of the synced image can
// 2) Be able to use the allocator generically for all opencv mats

// I think to do this we need two allocators, one global allocator that is used for all opencv mats and one
// allocator that is mad specifically for wrapping.

namespace aq
{

    // This is the one that is globally accessible such that all cv::Mats are allocated from it
    class CvMatAllocator : public cv::MatAllocator
    {
        struct UserData
        {
            SyncedImage::Ptr_t img;
            const CvMatAllocator* alloc;
        };

      public:
        CvMatAllocator(std::shared_ptr<mo::IDeviceStream> stream)
            : m_stream(stream)
        {
        }

        cv::UMatData*
        allocate(int dims, const int* sizes, int type, void*, size_t*, int, cv::UMatUsageFlags) const override
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
                fmt = aq::PixelFormat::kBGR;
                break;
            default:
                THROW(warn, "Unsupported number of channels {}", channels);
            }
            aq::DataFlag dtype = aq::fromCvDepth(CV_MAT_DEPTH(type));
            SyncedImage::Ptr_t image = std::make_shared<SyncedImage>();
            image->setStream(m_stream);
            image->create(static_cast<uint32_t>(height), static_cast<uint32_t>(width), fmt, dtype);

            auto host_data = image->data()->mutableHost();
            auto udata = new cv::UMatData(this);
            udata->data = ct::ptrCast<uchar>(host_data.data());
            udata->size = host_data.size();
            udata->origdata = ct::ptrCast<uchar>(host_data.data());

            auto user_data = new UserData;
            // -_-
            user_data->alloc = this;
            user_data->img = image;
            udata->userdata = user_data;
            return udata;
        }

        SyncedImage::Ptr_t getSyncedImage(cv::Mat mat) const
        {
            if (mat.u && mat.u->currAllocator == this)
            {
                auto data = ct::ptrCast<UserData>(mat.u->userdata);
                if (data)
                {
                    return data->img;
                }
            }
            return {};
        }

        bool allocate(cv::UMatData*, int, cv::UMatUsageFlags) const override
        {
            return false;
        }

        void deallocate(cv::UMatData* data) const override
        {
            auto user_data = ct::ptrCast<UserData>(data->userdata);
            delete user_data;
            delete data;
        }

      private:
        std::shared_ptr<mo::IDeviceStream> m_stream;
    };

    struct ScopedOpenCVAllocator
    {
        ScopedOpenCVAllocator(cv::MatAllocator* alloc)
        {
            m_prev_allocator = cv::Mat::getDefaultAllocator();
            cv::Mat::setDefaultAllocator(alloc);
        }

        ~ScopedOpenCVAllocator()
        {
            cv::Mat::setDefaultAllocator(m_prev_allocator);
        }

      private:
        cv::MatAllocator* m_prev_allocator = nullptr;
    };
} // namespace aq
#endif // AQ_TYPES_CV_MAT_ALLOCATOR_WRAPPER
