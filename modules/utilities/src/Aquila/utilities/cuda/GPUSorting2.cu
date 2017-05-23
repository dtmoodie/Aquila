#include <Aquila/core/detail/Export.hpp>
#include <Aquila/utilities/thrust/thrust_interop.hpp>
#include <Aquila/utilities/cuda/GPUSortingPriv.hpp>
namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template AQUILA_EXPORTS void sortAscending<float>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<float>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<float>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<float>(cv::cuda::GpuMat&, cudaStream_t);

            template AQUILA_EXPORTS void sortAscending<double>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<double>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<double>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<double>(cv::cuda::GpuMat&, cudaStream_t);
        }
    }
}
