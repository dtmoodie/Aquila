#pragma once

#include <Aquila/utilities/thrust/thrust_interop.hpp>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/utility.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>

namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template <typename T>
            struct UnarySortDescending
            {
                template <typename U1>
                __device__ __host__ void operator()(const U1& it1)
                {
                    thrust::sort(&thrust::get<0>(it1), &thrust::get<1>(it1), thrust::greater<T>());
                }
            };

            template <typename T>
            struct UnarySortAscending
            {
                template <typename U1>
                __device__ __host__ void operator()(const U1& it1)
                {
                    thrust::sort(&thrust::get<0>(it1), &thrust::get<1>(it1), thrust::less<T>());
                }
            };

            template <typename T>
            void sortAscending(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
#if THRUST_VERSION < 100802
                (void)stream;
#endif
                thrust::sort(
#if THRUST_VERSION >= 100802
                    thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream),
#endif
                    view.begin(),
                    view.end(),
                    thrust::less<float>());
            }

            template <typename T>
            void sortDescending(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
#if THRUST_VERSION < 100802
                (void)stream;
#endif
                thrust::sort(
#if THRUST_VERSION >= 100802
                    thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream),
#endif
                    view.begin(),
                    view.end(),
                    thrust::greater<float>());
            }

            template <typename T>
            void sortAscendingEachRow(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
                auto range = view.rowRange(0, -1);
#if THRUST_VERSION < 100802
                (void)stream;
#endif

                thrust::for_each(
#if THRUST_VERSION >= 100802
                    thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream),
#endif
                    range.first,
                    range.second,
                    UnarySortAscending<T>());
            }

            template <typename T>
            void sortDescendingEachRow(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
                auto range = view.rowRange(0, -1);

                thrust::for_each(
#if THRUST_VERSION >= 100802
                    thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream),
#endif
                    range.first,
                    range.second,
                    UnarySortDescending<T>());
            }
        } // namespace detail

        template <class T>
        void sort(cv::cuda::GpuMat& sorted,
                  cv::SortFlags flags = cv::SORT_ASCENDING,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null())
        {
            if (flags == cv::SORT_ASCENDING)
            {
                detail::sortAscending<T>(sorted, cv::cuda::StreamAccessor::getStream(stream));
            }
            else
            {
                detail::sortDescending<T>(sorted, cv::cuda::StreamAccessor::getStream(stream));
            }
        }

        template <class T>
        void sort(const cv::cuda::GpuMat& unsorted,
                  cv::cuda::GpuMat& sorted,
                  cv::SortFlags flags = cv::SORT_ASCENDING,
                  cv::cuda::Stream& stream = cv::cuda::Stream::Null())
        {
            unsorted.copyTo(sorted, stream);
            sort<T>(sorted, flags, stream);
        }

    } // namespace cuda
} // namespace cv
