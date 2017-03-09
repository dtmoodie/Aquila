#include <Aquila/Thrust_interop.hpp>
#include "Aquila/utilities/GPUSortingPriv.hpp"
namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template AQUILA_EXPORTS void sortAscending<ushort>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<ushort>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<ushort>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<ushort>(cv::cuda::GpuMat&, cudaStream_t);

            template AQUILA_EXPORTS void sortAscending<short>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<short>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<short>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<short>(cv::cuda::GpuMat&, cudaStream_t);

            template AQUILA_EXPORTS void sortAscending<int>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescending<int>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortAscendingEachRow<int>(cv::cuda::GpuMat&, cudaStream_t);
            template AQUILA_EXPORTS void sortDescendingEachRow<int>(cv::cuda::GpuMat&, cudaStream_t);
        }
    }
}
