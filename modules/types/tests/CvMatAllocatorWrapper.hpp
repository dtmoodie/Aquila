#pragma once
#include <Aquila/types/SyncedImage.hpp>
#include <opencv2/core/mat.hpp>

struct CvMatAllocatorWrapper final : public cv::MatAllocator
{
    static cv::Mat wrap(std::shared_ptr<aq::SyncedImage> image);
    cv::UMatData*
    allocate(int dims, const int* sizes, int type, void*, size_t*, int, cv::UMatUsageFlags) const override;

    bool allocate(cv::UMatData*, int, cv::UMatUsageFlags) const override;

    void deallocate(cv::UMatData* data) const override;

  private:
    CvMatAllocatorWrapper(std::shared_ptr<aq::SyncedImage> image);
    // Why did opencv decide the allocate function needed to be const -_-

    mutable std::shared_ptr<aq::SyncedImage> m_image;
};
