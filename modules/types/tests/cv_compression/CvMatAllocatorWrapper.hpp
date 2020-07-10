#ifndef MO_TYPES_CV_MAT_ALLOCATOR_WRAPPER_HPP
#define MO_TYPES_CV_MAT_ALLOCATOR_WRAPPER_HPP
#include <Aquila/types/SyncedImage.hpp>
#include <aq_cv_compression_export.hpp>
#include <opencv2/core/mat.hpp>
namespace aq
{

    cv::Mat AQUILA_EXPORTS wrap(std::shared_ptr<aq::SyncedImage> image);

    struct CvAllocatorContextManager
    {
        CvAllocatorContextManager(cv::MatAllocator* alloc);

        ~CvAllocatorContextManager();

      private:
        cv::MatAllocator* m_original;
    };
} // namespace aq

#endif // MO_TYPES_CV_MAT_ALLOCATOR_WRAPPER_HPP