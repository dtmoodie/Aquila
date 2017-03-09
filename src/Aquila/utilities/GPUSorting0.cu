#include <Aquila/Thrust_interop.hpp>
#include "Aquila/utilities/GPUSortingPriv.hpp"
namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template AQUILA_EXPORTS void sortAscending<uchar>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<uchar>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<uchar>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<uchar>(cv::cuda::GpuMat&, cudaStream_t);

            template AQUILA_EXPORTS void sortAscending<char>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<char>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<char>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<char>(cv::cuda::GpuMat&, cudaStream_t);
        }
    }
}

