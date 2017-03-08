#pragma once
#include "Allocator.hpp"
#include "MetaObject/Logging/Log.hpp"
#include <opencv2/cudev/common.hpp>
#include <opencv2/core/cuda.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>
#include <cuda_runtime.h>
namespace mo
{
#define MO_CUDA_ERROR_CHECK(expr, msg) \
{ \
    cudaError_t err = (expr); \
    if(err != cudaSuccess) \
    { \
       THROW(warning)  << #expr << " failed " << cudaGetErrorString(err) << " " msg; \
    } \
}

unsigned char* alignMemory(unsigned char* ptr, int elemSize)
{
    int i;
    for (i = 0; i < elemSize; ++i)
    {
        if (reinterpret_cast<size_t>(ptr + i) % elemSize == 0)
        {
            break;
        }
    }
    return ptr + i;  // Forces memory to be aligned to an element's byte boundary
}
int alignmentOffset(unsigned char* ptr, int elemSize)
{
    int i;
    for (i = 0; i < elemSize; ++i)
    {
        if (reinterpret_cast<size_t>(ptr + i) % elemSize == 0)
        {
            break;
        }
    }
    return i;
}

/// ==========================================================
/// PitchedPolicy
PitchedPolicy::PitchedPolicy()
{
    textureAlignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
}

void PitchedPolicy::SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride)
{
    if (rows == 1 || cols == 1)
    {
        stride = cols*elemSize;
    }
    else
    {
        if((cols*elemSize % textureAlignment) == 0)
            stride = cols*elemSize;
        else
            stride = cols*elemSize + textureAlignment - (cols*elemSize % textureAlignment);
    }
    sizeNeeded = stride*rows;
}

/// ==========================================================
/// ContinuousPolicy
void ContinuousPolicy::SizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride)
{
    stride = cols*elemSize;
    sizeNeeded = stride * rows;
}

/// ==========================================================
/// PoolPolicy
template<typename PaddingPolicy>
PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::PoolPolicy(size_t initialBlockSize):
    _initial_block_size(initialBlockSize)
{
    blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(_initial_block_size)));
}

template<typename PaddingPolicy>
bool PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    size_t sizeNeeded, stride;
    PaddingPolicy::SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
    unsigned char* ptr;
    for (auto itr : blocks)
    {
        ptr = itr->allocate(sizeNeeded, elemSize);
        if (ptr)
        {
            mat->data = ptr;
            mat->step = stride;
            mat->refcount = (int*)cv::fastMalloc(sizeof(int));
            memoryUsage += mat->step*rows;
            LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") "
                       << mat->step * rows / (1024 * 1024) << " MB from memory block. Total usage: "
                       << memoryUsage / (1024 * 1024) << " MB";
            return true;
        }
    }
    // If we get to this point, then no memory was found, need to allocate new memory
    blocks.push_back(std::shared_ptr<GpuMemoryBlock>(
                         new GpuMemoryBlock(
                             std::max(_initial_block_size / 2, sizeNeeded))));
    LOG(trace) << "[GPU] Expanding memory pool by " <<
                  std::max(_initial_block_size / 2, sizeNeeded) / (1024 * 1024)
               << " MB";
    if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, elemSize))
    {
        mat->data = ptr;
        mat->step = stride;
        mat->refcount = (int*)cv::fastMalloc(sizeof(int));
        memoryUsage += mat->step*rows;
        LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") "
                   << mat->step * rows / (1024 * 1024)
                   << " MB from memory block. Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
        return true;
    }
    return false;
}

template<typename PaddingPolicy>
void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
{
    for (auto itr : blocks)
    {
        if (itr->deAllocate(mat->data))
        {
            cv::fastFree(mat->refcount);
            memoryUsage -= mat->step*mat->rows;
            return;
        }
    }
    throw cv::Exception(0, "[GPU] Unable to find memory to deallocate", __FUNCTION__, __FILE__, __LINE__);
}

template<typename PaddingPolicy>
unsigned char* PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t sizeNeeded)
{
    unsigned char* ptr;
    for (auto itr : blocks)
    {
        ptr = itr->allocate(sizeNeeded, 1);
        if (ptr)
        {
            memoryUsage += sizeNeeded;

            return ptr;
        }
    }
    // If we get to this point, then no memory was found, need to allocate new memory
    blocks.push_back(std::shared_ptr<GpuMemoryBlock>(new GpuMemoryBlock(std::max(_initial_block_size / 2, sizeNeeded))));
    LOG(trace) << "[GPU] Expanding memory pool by "
               << std::max(_initial_block_size / 2, sizeNeeded) / (1024 * 1024)
               << " MB";
    if (unsigned char* ptr = (*blocks.rbegin())->allocate(sizeNeeded, 1))
    {
        memoryUsage += sizeNeeded;

        return ptr;
    }
    return nullptr;
}

template<typename PaddingPolicy>
void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    for (auto itr : blocks)
    {
        if (itr->deAllocate(ptr))
        {
            return;
        }
    }
}

template<typename PaddingPolicy>
void PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>::Release()
{
    blocks.clear();
}


/// ==========================================================
/// StackPolicy
template<typename PaddingPolicy>
StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::StackPolicy()
{
#ifdef _MSC_VER
    deallocateDelay = 1000;
#else
    deallocateDelay = 1000*1000;
#endif
}

template<typename PaddingPolicy>
bool StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    size_t sizeNeeded, stride;

    PaddingPolicy::SizeNeeded(rows, cols, elemSize, sizeNeeded, stride);
    for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
    {
        if(itr->size == sizeNeeded)
        {
            mat->data = itr->ptr;
            mat->step = stride;
            mat->refcount = (int*)cv::fastMalloc(sizeof(int));
            clock_t time = clock();
            LOG(trace) << "[GPU] Reusing block of size (" << rows << "," << cols << ") "
                       << mat->step * rows / (1024 * 1024) << " MB at " << (void*)itr->ptr
                       << " which was stale for "
                       << (time - itr->free_time) * 1000 / CLOCKS_PER_SEC
                       << " ms. total usage: "
                       << memoryUsage / (1024 * 1024) << " MB";
            this->memoryUsage += sizeNeeded;
            deallocateList.erase(itr);
            return true;
        }
    }
    if (rows > 1 && cols > 1)
    {

        MO_CUDA_ERROR_CHECK(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows),
                            << " while allocating " << rows << ", " << cols);

        this->memoryUsage += mat->step * rows;
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << mat->step * rows / (1024 * 1024) << " MB at "
                   << (void*)mat->data << ". Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
    }
    else
    {
        CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
        this->memoryUsage += mat->step * rows;
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << cols * rows / (1024 * 1024) << " MB at "
                   << (void*)mat->data << ". Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
        mat->step = elemSize * cols;
    }
    mat->refcount = (int*)cv::fastMalloc(sizeof(int));
    return true;
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
{
    //this->memoryUsage -= mat->rows*mat->step;
    LOG(trace) << "[GPU] Releasing mat of size (" << mat->rows << ","
               << mat->cols << ") " << (mat->dataend - mat->datastart)/(1024*1024) << " MB to the memory pool";
    deallocateList.emplace_back(mat->datastart, clock(), mat->dataend - mat->datastart);
    cv::fastFree(mat->refcount);
    clear();
}

template<typename PaddingPolicy>
unsigned char* StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t sizeNeeded)
{
    unsigned char* ptr = nullptr;
    auto time = clock();
    for (auto itr = deallocateList.begin(); itr != deallocateList.end(); ++itr)
    {
        if (itr->size == sizeNeeded)
        {
            ptr = itr->ptr;
            this->memoryUsage += sizeNeeded;
            current_allocations[ptr] = sizeNeeded;
            LOG(trace) << "[GPU] Allocating block of size " << itr->size / (1024 * 1024)
                       << "MB from the memory stack which was stale for " << (time - itr->free_time) * 1000 / CLOCKS_PER_SEC
                       << " ms at " << (void*) itr->ptr;
            deallocateList.erase(itr);
            return ptr;
        }
    }
    CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, sizeNeeded));
    this->memoryUsage += sizeNeeded;
    current_allocations[ptr] = sizeNeeded;
    LOG(trace) << "[GPU] Allocating block of size " << sizeNeeded / (1024 * 1024)
               << "MB at " <<  (void*) ptr;
    return ptr;
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    auto itr = current_allocations.find(ptr);
    if(itr != current_allocations.end())
    {
        current_allocations.erase(itr);
        deallocateList.emplace_back(ptr, clock(), current_allocations[ptr]);
    }

    clear();
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::clear()
{
    auto time = clock();
    for (auto itr = deallocateList.begin(); itr != deallocateList.end(); )
    {
        if((time - itr->free_time) > deallocateDelay)
        {
            memoryUsage -= itr->size;
            LOG(trace) << "[GPU] Deallocating block of size " << itr->size /(1024*1024)
                       << "MB. Which was stale for " << (time - itr->free_time) * 1000 / CLOCKS_PER_SEC << " ms at "
                       << (void*)itr->ptr;
            CV_CUDEV_SAFE_CALL(cudaFree(itr->ptr));
            itr = deallocateList.erase(itr);
        }else
        {
            ++itr;
        }
    }
}

template<typename PaddingPolicy>
void StackPolicy<cv::cuda::GpuMat, PaddingPolicy>::Release()
{
    for(auto& itr : deallocateList)
    {
        CV_CUDEV_SAFE_CALL(cudaFree(itr.ptr));
    }
    deallocateList.clear();
}

/// ==========================================================
/// NonCachingPolicy
template<typename PaddingPolicy>
bool NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    size_t size_needed, stride;
    PaddingPolicy::SizeNeeded(rows, cols, elemSize, size_needed, stride);
    if (rows > 1 && cols > 1)
    {
        CV_CUDEV_SAFE_CALL(cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows));
        memoryUsage += mat->step*rows;
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << mat->step * rows / (1024 * 1024) << " MB. Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
    }
    else
    {
        CV_CUDEV_SAFE_CALL(cudaMalloc(&mat->data, elemSize * cols * rows));
        memoryUsage += elemSize*cols*rows;
        LOG(trace) << "[GPU] Allocating block of size (" << rows << "," << cols << ") "
                   << cols * rows / (1024 * 1024) << " MB. Total usage: "
                   << memoryUsage / (1024 * 1024) << " MB";
        mat->step = elemSize * cols;
    }
    mat->refcount = (int*)cv::fastMalloc(sizeof(int));
    return true;
}

template<typename PaddingPolicy>
void NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::free(cv::cuda::GpuMat* mat)
{
    CV_CUDEV_SAFE_CALL(cudaFree(mat->data));
    cv::fastFree(mat->refcount);
}

template<typename PaddingPolicy>
unsigned char* NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::allocate(size_t num_bytes)
{
    unsigned char* ptr = nullptr;
    CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, num_bytes));
    memoryUsage += num_bytes;
    return ptr;
}

template<typename PaddingPolicy>
void NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    CV_CUDEV_SAFE_CALL(cudaFree(ptr));
}

/// ==========================================================
/// LockPolicy
template<class Allocator>
bool LockPolicyImpl<Allocator, cv::cuda::GpuMat>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::allocate(mat, rows, cols, elemSize);
}

template<class Allocator>
void LockPolicyImpl<Allocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::free(mat);
}

template<class Allocator>
unsigned char* LockPolicyImpl<Allocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::allocate(num_bytes);
}

template<class Allocator>
void LockPolicyImpl<Allocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::deallocate(ptr, num_bytes);
}

template<class Allocator>
cv::UMatData* LockPolicyImpl<Allocator, cv::Mat>::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
{
    return Allocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
}

template<class Allocator>
bool LockPolicyImpl<Allocator, cv::Mat>::allocate(cv::UMatData* data, int accessflags,
                                                  cv::UMatUsageFlags usageFlags) const
{
    return Allocator::allocate(data, accessflags, usageFlags);
}

template<class Allocator>
void LockPolicyImpl<Allocator, cv::Mat>::deallocate(cv::UMatData* data) const
{
    return Allocator::deallocate(data);
}
template<class Allocator>
unsigned char* LockPolicyImpl<Allocator, cv::Mat>::allocate(size_t num_bytes)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::allocate(num_bytes);
}

template<class Allocator>
void LockPolicyImpl<Allocator, cv::Mat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    boost::mutex::scoped_lock lock(mtx);
    return Allocator::deallocate(ptr, num_bytes);
}
/// ==========================================================
/// RefCountPolicy



template<class Allocator>
RefCountPolicyImpl<Allocator, cv::Mat>::~RefCountPolicyImpl()
{
    CV_Assert(ref_count == 0);
}
template<class Allocator>
cv::UMatData* RefCountPolicyImpl<Allocator, cv::Mat>::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
{
    auto ret = Allocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
    if(ret)
    {
        //++ref_count;
        return ret;
    }
    return nullptr;
}
template<class Allocator>
bool RefCountPolicyImpl<Allocator, cv::Mat>::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    if(Allocator::allocate(data, accessflags, usageFlags))
    {
        //++ref_count;
        return true;
    }
    return false;
}
template<class Allocator>
void RefCountPolicyImpl<Allocator, cv::Mat>::deallocate(cv::UMatData* data) const
{
    Allocator::deallocate(data);
    //--ref_count;
}

template<class Allocator>
unsigned char* RefCountPolicyImpl<Allocator, cv::Mat>::allocate(size_t num_bytes)
{
    ++ref_count;
    return Allocator::allocate(num_bytes);
}

template<class Allocator>
void RefCountPolicyImpl<Allocator, cv::Mat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    --ref_count;
    Allocator::deallocate(ptr, num_bytes);
}
/// =========================================================
/// GpuMat implementation

template<class Allocator>
RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>::~RefCountPolicyImpl()
{
    //CV_Assert(ref_count == 0 && "Warning, trying to delete allocator while cv::cuda::GpuMat's still reference it");
    if(ref_count != 0)
    {
        LOG(warning) << "Trying to delete allocator while cv::cuda::GpuMat's still reference it";
    }
}
template<class Allocator>
bool RefCountPolicyImpl<Allocator,cv::cuda::GpuMat>::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    if(Allocator::allocate(mat, rows, cols, elemSize))
    {
        ++ref_count;
        return true;
    }
    return false;
}
template<class Allocator>
void RefCountPolicyImpl<Allocator,cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
{
    Allocator::free(mat);
    --ref_count;
}
template<class Allocator>
unsigned char* RefCountPolicyImpl<Allocator,cv::cuda::GpuMat>::allocate(size_t num_bytes)
{
    if(auto ptr = Allocator::allocate(num_bytes))
    {
        ++ref_count;
        return ptr;
    }
    return nullptr;
}
template<class Allocator>
void RefCountPolicyImpl<Allocator,cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    Allocator::deallocate(ptr, num_bytes);
    --ref_count;
}


/// ==========================================================
/// ScopedDebugPolicy


template<class Allocator>
bool ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    if(Allocator::allocate(mat, rows, cols, elemSize))
    {
        scopeOwnership[mat->data] = GetScopeName();
        scopedAllocationSize[GetScopeName()] += mat->step * mat->rows;
        return true;
    }
    return false;
}

template<class Allocator>
void ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
{
    Allocator::free(mat);
    auto itr = scopeOwnership.find(mat->data);
    if (itr != scopeOwnership.end())
    {
        scopedAllocationSize[itr->second] -= mat->rows * mat->step;
    }

}

template<class Allocator>
unsigned char* ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
{
    if(auto ptr = Allocator::allocate(num_bytes))
    {
        scopeOwnership[ptr] = GetScopeName();
        scopedAllocationSize[GetScopeName()] += num_bytes;
        return ptr;
    }
    return nullptr;
}

template<class Allocator>
void ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    Allocator::free(ptr);
    auto itr = scopeOwnership.find(ptr);
    if (itr != scopeOwnership.end())
    {
        scopedAllocationSize[itr->second] -= this->current_allocations[ptr];
    }
}

template<class SmallAllocator, class LargeAllocator>
CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::CombinedPolicyImpl(size_t threshold_)
    : threshold(threshold_)
{

}

template<class SmallAllocator, class LargeAllocator>
bool CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::allocate(
        cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    if(rows*cols*elemSize < threshold)
    {
        return SmallAllocator::allocate(mat, rows, cols, elemSize);
    }else
    {
        return LargeAllocator::allocate(mat, rows, cols, elemSize);
    }
}

template<class SmallAllocator, class LargeAllocator>
void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::free(cv::cuda::GpuMat* mat)
{
    if(mat->rows * mat->cols * mat->elemSize() < threshold)
    {
        SmallAllocator::free(mat);

    }else
    {
        LargeAllocator::free(mat);
    }
}

template<class SmallAllocator, class LargeAllocator>
unsigned char* CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::allocate(size_t num_bytes)
{
    return SmallAllocator::allocate(num_bytes);
}

template<class SmallAllocator, class LargeAllocator>
void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    return SmallAllocator::deallocate(ptr, num_bytes);
}

template<class SmallAllocator, class LargeAllocator>
void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>::Release()
{
    SmallAllocator::Release();
    LargeAllocator::Release();
}

template<class SmallAllocator, class LargeAllocator>
CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::CombinedPolicyImpl(size_t threshold_):
    threshold(threshold_)
{

}

template<class SmallAllocator, class LargeAllocator>
cv::UMatData* CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::allocate(
                            int dims, const int* sizes, int type,
                            void* data, size_t* step, int flags,
                            cv::UMatUsageFlags usageFlags) const
{
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--)
    {
        if (step)
        {
            if (data && step[i] != CV_AUTOSTEP)
            {
                CV_Assert(total <= step[i]);
                total = step[i];
            }
            else
            {
                step[i] = total;
            }
        }

        total *= sizes[i];
    }
    if(total < threshold)
    {
        return SmallAllocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
    }else
    {
        return LargeAllocator::allocate(dims, sizes, type, data, step, flags, usageFlags);
    }
}

template<class SmallAllocator, class LargeAllocator>
bool CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::allocate(
                    cv::UMatData* data, int accessflags,
                    cv::UMatUsageFlags usageFlags) const
{
    return (data != NULL);
}

template<class SmallAllocator, class LargeAllocator>
void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::deallocate(cv::UMatData* data) const
{
    if(data->size < threshold)
    {
        SmallAllocator::deallocate(data);
    }else
    {
        LargeAllocator::deallocate(data);
    }
}

template<class SmallAllocator, class LargeAllocator>
unsigned char* CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::allocate(size_t num_bytes)
{
    if(num_bytes < threshold)
    {
        return SmallAllocator::allocate(num_bytes);
    }else
    {
        return LargeAllocator::allocate(num_bytes);
    }
}

template<class SmallAllocator, class LargeAllocator>
void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::deallocate(unsigned char* ptr, size_t num_bytes)
{
    if(num_bytes < threshold)
    {
        return SmallAllocator::deallocate(ptr, num_bytes);
    }else
    {
        return LargeAllocator::deallocate(ptr, num_bytes);
    }
}

template<class SmallAllocator, class LargeAllocator>
void CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>::Release()
{
    SmallAllocator::Release();
    LargeAllocator::Release();
}

template<class SmallAllocator, class LargeAllocator>
CombinedPolicy<SmallAllocator, LargeAllocator>::CombinedPolicy(size_t threshold)
    :CombinedPolicyImpl<SmallAllocator, LargeAllocator, typename LargeAllocator::MatType>(threshold)
{

}

template<class CPUAllocator, class GPUAllocator>
class ConcreteAllocator
        : virtual public GPUAllocator
        , virtual public CPUAllocator
        , virtual public mo::Allocator
{
public:
    // GpuMat allocate
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        return GPUAllocator::allocate(mat, rows, cols, elemSize);
    }

    void free(cv::cuda::GpuMat* mat)
    {
        return GPUAllocator::free(mat);
    }

    // Thrust allocate
    unsigned char* allocateGpu(size_t num_bytes)
    {
        return GPUAllocator::allocate(num_bytes);
    }

    void deallocateGpu(unsigned char* ptr, size_t num_bytes)
    {
        return GPUAllocator::deallocate(ptr, num_bytes);
    }

    unsigned char* allocateCpu(size_t num_bytes)
    {
        return CPUAllocator::allocate(num_bytes);
    }

    void deallocateCpu(unsigned char* ptr, size_t num_bytes)
    {
        CPUAllocator::deallocate(ptr, num_bytes);
    }

    // CPU allocate
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        return CPUAllocator::allocate(dims, sizes, type, data,
                                      step, flags, usageFlags);
    }

    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
    {
        return CPUAllocator::allocate(data, accessflags, usageFlags);
    }

    void deallocate(cv::UMatData* data) const
    {
        CPUAllocator::deallocate(data);
    }

    void Release()
    {
        CPUAllocator::Release();
        GPUAllocator::Release();
    }
};

typedef PoolPolicy<cv::cuda::GpuMat, PitchedPolicy>   d_TensorPoolAllocator_t;
typedef LockPolicy<d_TensorPoolAllocator_t>              d_mt_TensorPoolAllocator_t;

typedef StackPolicy<cv::cuda::GpuMat, PitchedPolicy>  d_TensorAllocator_t;
typedef StackPolicy<cv::cuda::GpuMat, PitchedPolicy>     d_TextureAllocator_t;

typedef LockPolicy<d_TensorAllocator_t>                  d_mt_TensorAllocator_t;
typedef LockPolicy<d_TextureAllocator_t>                 d_mt_TextureAllocator_t;

typedef CpuPoolPolicy h_PoolAllocator_t;
typedef CpuStackPolicy h_StackAllocator_t;

typedef mt_CpuPoolPolicy                    h_mt_PoolAllocator_t;
typedef mt_CpuStackPolicy                   h_mt_StackAllocator_t;
//#ifdef NDEBUG
//typedef CombinedPolicy<d_TensorPoolAllocator_t, d_TextureAllocator_t> d_UniversalAllocator_t;
//typedef LockPolicy<d_UniversalAllocator_t> d_mt_UniversalAllocator_t;
//#else
typedef RefCountPolicy<CombinedPolicy<d_TensorPoolAllocator_t, d_TextureAllocator_t>> d_UniversalAllocator_t;
typedef RefCountPolicy<LockPolicy<d_UniversalAllocator_t>> d_mt_UniversalAllocator_t;
//#endif
typedef CombinedPolicy<h_PoolAllocator_t, h_StackAllocator_t> h_UniversalAllocator_t;
typedef LockPolicy<h_UniversalAllocator_t> h_mt_UniversalAllocator_t;


typedef ConcreteAllocator<h_mt_PoolAllocator_t, d_mt_TensorPoolAllocator_t> mt_TensorAllocator_t;
typedef ConcreteAllocator<h_PoolAllocator_t, d_TensorAllocator_t>           TensorAllocator_t;

typedef ConcreteAllocator<h_mt_StackAllocator_t, d_mt_TextureAllocator_t>   mt_TextureAllocator_t;
typedef ConcreteAllocator<h_StackAllocator_t, d_TextureAllocator_t>         TextureAllocator_t;

typedef ConcreteAllocator<h_UniversalAllocator_t, d_UniversalAllocator_t>    UniversalAllocator_t;
typedef ConcreteAllocator<h_mt_UniversalAllocator_t, d_mt_UniversalAllocator_t>    mt_UniversalAllocator_t;
}
