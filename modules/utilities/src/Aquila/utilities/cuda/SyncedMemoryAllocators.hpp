#pragma once
#include "Aquila/utilities/CpuMatAllocators.h"
#include "GpuMatAllocators.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
namespace aq
{
    class ISyncedMemoryAllocator:
            virtual public CpuPinnedAllocator,
            virtual public cv::cuda::GpuMat::Allocator
    {
    public:
        /*!
         * \brief Instance thread specific accessor
         * \return
         */
        static ISyncedMemoryAllocator* Instance();
        /*!
         * \brief SetInstance resets the thread specific pointer to the new allocator
         * \param allocator
         */
        static void SetInstance(ISyncedMemoryAllocator* allocator);
    };


}
