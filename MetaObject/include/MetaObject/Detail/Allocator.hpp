#pragma once
#include "HelperMacros.hpp"
#include "Export.hpp"
#include "MemoryBlock.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <boost/thread/mutex.hpp>
#include <list>
#include <cuda.h>

namespace mo
{
MO_EXPORTS inline unsigned char* alignMemory(unsigned char* ptr, int elemSize);
MO_EXPORTS inline int alignmentOffset(unsigned char* ptr, int elemSize);
MO_EXPORTS void SetScopeName(const std::string& name);
MO_EXPORTS const std::string& GetScopeName();
MO_EXPORTS void InstallThrustPoolingAllocator();



template<class T>
class GpuThreadAllocatorSetter
{
    DEFINE_HAS_STATIC_FUNCTION(HasGpuDefaultAllocator, setDefaultThreadAllocator, cv::cuda::GpuMat::Allocator*);
public:
    static bool Set(cv::cuda::GpuMat::Allocator* allocator)
    {
        return helper<T>(allocator);
    }
private:
    template<class U> static bool helper(typename std::enable_if<!HasGpuDefaultAllocator<U>::value, cv::cuda::GpuMat::Allocator>::type* allocator)
    {
        return false;
    }
    template<class U> static bool helper(typename std::enable_if<HasGpuDefaultAllocator<U>::value, cv::cuda::GpuMat::Allocator>::type* allocator)
    {
        U::setDefaultThreadAllocator(allocator);
        return true;
    }
};

template<class T>
class CpuThreadAllocatorSetter
{
    DEFINE_HAS_STATIC_FUNCTION(HasCpuDefaultAllocator, setDefaultThreadAllocator, cv::MatAllocator*);
public:
    static bool Set(cv::MatAllocator* allocator)
    {
        return helper<T>(allocator);
    }
private:
    template<class U> static bool helper(typename std::enable_if<!HasCpuDefaultAllocator<U>::value, cv::MatAllocator>::type* allocator)
    {
        return false;
    }
    template<class U> static bool helper(typename std::enable_if<HasCpuDefaultAllocator<U>::value, cv::MatAllocator>::type* allocator)
    {
        U::setDefaultThreadAllocator(allocator);
        return true;
    }
};




class Allocator;
class MO_EXPORTS CpuAllocatorThreadAdapter: public cv::MatAllocator
{
public:
    static void SetThreadAllocator(cv::MatAllocator* allocator);
    static void SetGlobalAllocator(cv::MatAllocator* allocator);
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
};

class MO_EXPORTS GpuAllocatorThreadAdapter: public cv::cuda::GpuMat::Allocator
{
public:
    static void SetThreadAllocator(cv::cuda::GpuMat::Allocator* allocator);
    static void SetGlobalAllocator(cv::cuda::GpuMat::Allocator* allocator);

    virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    virtual void free(cv::cuda::GpuMat* mat);
};

class MO_EXPORTS Allocator
        : virtual public cv::cuda::GpuMat::Allocator
        , virtual public cv::MatAllocator
{
public:
    static Allocator* GetThreadSafeAllocator();
    static Allocator* GetThreadSpecificAllocator();

    // Used for stl allocators
    virtual unsigned char* allocateGpu(size_t num_bytes) = 0;
    virtual void deallocateGpu(uchar* ptr, size_t numBytes) = 0;

    virtual unsigned char* allocateCpu(size_t num_bytes) = 0;
    virtual void deallocateCpu(uchar* ptr, size_t numBytes) = 0;
    virtual void Release() {}
    void SetName(const std::string& name){this->name = name;}
    const std::string GetName(){return name;}
private:
    std::string name;
};

/// ========================================================
/// Memory layout policies
/// ========================================================

/*!
 * \brief The PitchedPolicy class allocates memory with padding
 *        such that a 2d array can be utilized as a texture reference
 */
class MO_EXPORTS PitchedPolicy
{
public:
    inline PitchedPolicy();
    inline void SizeNeeded(int rows, int cols, int elemSize,
                           size_t& sizeNeeded, size_t& stride);
private:
    size_t textureAlignment;
};

/*!
 * \brief The ContinuousPolicy class allocates memory with zero padding
 *        wich allows for nice reshaping operations
 *
 */
class MO_EXPORTS ContinuousPolicy
{
public:
    inline void SizeNeeded(int rows, int cols, int elemSize,
                           size_t& sizeNeeded, size_t& stride);
};

/// ========================================================
/// Allocation Policies
/// ========================================================

/*!
 * \brief The AllocationPolicy class is a base for all other allocation
 *        policies, it's members track memory usage by the allocator
 */
class MO_EXPORTS AllocationPolicy
{
public:
    /*!
     * \brief GetMemoryUsage
     * \return current estimated memory usage
     */
    inline size_t GetMemoryUsage() const;
    virtual void Release() {}
protected:
    size_t memoryUsage;
    /*!
     * \brief current_allocations keeps track of the stacks allocated by allocate(size_t)
     *        since a call to free(unsigned char*) will not return the size of the allocated
     *        data
     */
    std::map<unsigned char*, size_t> current_allocations;
};

/*!
 * \brief The PoolPolicy allocation policy uses a memory pool to cache
 *        memory usage of small amounts of data.  Best used for variable
 *        amounts of data
 */
template<typename T, typename PaddingPolicy>
class MO_EXPORTS PoolPolicy
{

};

/// ========================================================================================
template<typename PaddingPolicy>
class MO_EXPORTS PoolPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
        , public virtual PaddingPolicy
{
public:
    typedef cv::cuda::GpuMat MatType;
    PoolPolicy(size_t initialBlockSize = 1e7);

    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
    virtual void Release();
private:
    size_t _initial_block_size;
    std::list<std::shared_ptr<GpuMemoryBlock>> blocks;
};


class MO_EXPORTS CpuPoolPolicy: virtual public cv::MatAllocator
{
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
    uchar* allocate(size_t num_bytes);
    void deallocate(uchar* ptr, size_t num_bytes);
    void Release() {}
};

class MO_EXPORTS mt_CpuPoolPolicy : virtual public CpuPoolPolicy
{
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
    uchar* allocate(size_t num_bytes);
    void deallocate(uchar* ptr, size_t num_bytes);
};
class MO_EXPORTS PinnedAllocator : virtual public cv::MatAllocator
{
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
};


/*!
 *  \brief The StackPolicy class checks for a free memory stack of the exact
 *         requested size, if it is available it allocates from the free stack.
 *         If it wasn't able to find memory of the exact requested size, it will
 *         allocate the exact size and return it.  Since memory is not coelesced
 *         between stacks, this is best for large fixed size data, such as
 *         repeatedly allocated and deallocated images.
 */
template<typename T, typename PaddingPolicy> class MO_EXPORTS StackPolicy{};

/// =================================================================================
template<typename PaddingPolicy>
class MO_EXPORTS StackPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
        , public virtual PaddingPolicy
{
public:
    typedef cv::cuda::GpuMat MatType;
    StackPolicy();
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    void free(cv::cuda::GpuMat* mat);

    unsigned char* allocate(size_t num_bytes);
    void deallocate(unsigned char* ptr, size_t num_bytes);
    virtual void Release();
protected:
    void clear();
    struct FreeMemory
    {
        FreeMemory(unsigned char* ptr_, clock_t time_, size_t size_):
            ptr(ptr_), free_time(time_), size(size_){}
        unsigned char* ptr;
        clock_t free_time;
        size_t size;
    };
    std::list<FreeMemory> deallocateList;
    size_t deallocateDelay; // ms
};


class MO_EXPORTS CpuStackPolicy
    : public virtual AllocationPolicy
    , public virtual ContinuousPolicy
    , public virtual cv::MatAllocator
{
public:
    typedef cv::Mat MatType;
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    uchar* allocate(size_t total);
    void deallocate(cv::UMatData* data) const;
    void deallocate(uchar* ptr, size_t total);
    void Release(){}
};

class MO_EXPORTS mt_CpuStackPolicy: virtual public CpuStackPolicy
{
public:
    typedef cv::Mat MatType;
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    uchar* allocate(size_t total);
    bool deallocate(uchar* ptr, size_t total);
    void deallocate(cv::UMatData* data) const;
};


/*!
 *  \brief The NonCachingPolicy allocates and deallocates the same as
 *         OpenCV's default allocator.  It has the advantage of memory
 *         usage tracking.
 */

template<typename T, typename PaddingPolicy> class MO_EXPORTS NonCachingPolicy {};

template<typename PaddingPolicy>
class MO_EXPORTS NonCachingPolicy<cv::cuda::GpuMat, PaddingPolicy>
        : public virtual AllocationPolicy
{
public:
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    void free(cv::cuda::GpuMat* mat);
    unsigned char* allocate(size_t num_bytes);
    void deallocate(unsigned char* ptr, size_t num_bytes);
};

template<class Allocator, class MatType>
class LockPolicyImpl: public Allocator{};

template<class Allocator>
class LockPolicyImpl<Allocator, cv::cuda::GpuMat>: public Allocator
{
public:
    typedef cv::cuda::GpuMat MatType;
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
private:
    boost::mutex mtx;
};

template<class Allocator>
class LockPolicyImpl<Allocator, cv::Mat>: public Allocator
{
public:
    typedef cv::Mat MatType;
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
private:
    boost::mutex mtx;
};

template<class Allocator, class MatType>
class RefCountPolicyImpl
{
};

template<class Allocator>
class RefCountPolicyImpl<Allocator, cv::Mat>: public Allocator
{
public:
    typedef cv::Mat MatType;

    template<class... T>RefCountPolicyImpl(T... args):
        Allocator(args...){}

    ~RefCountPolicyImpl();
    cv::UMatData* allocate(int dims, const int* sizes, int type,
        void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const;
    bool allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const;
    void deallocate(cv::UMatData* data) const;
    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
private:
    int ref_count = 0;
};

template<class Allocator>
class RefCountPolicyImpl<Allocator, cv::cuda::GpuMat>: public Allocator
{
public:
    typedef cv::cuda::GpuMat MatType;

    template<class... T>RefCountPolicyImpl(T... args):
        Allocator(args...){}
    ~RefCountPolicyImpl();
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
private:
    int ref_count = 0;
};

class MO_EXPORTS CpuMemoryPool
{
public:
    virtual ~CpuMemoryPool() {}
    static CpuMemoryPool* GlobalInstance();
    static CpuMemoryPool* ThreadInstance();
    virtual bool allocate(void** ptr, size_t total, size_t elemSize) = 0;
    virtual uchar* allocate(size_t total) = 0;
    virtual bool deallocate(void* ptr, size_t total) = 0;
};

class MO_EXPORTS CpuMemoryStack
{
public:
    virtual ~CpuMemoryStack() {}
    static CpuMemoryStack* GlobalInstance();
    static CpuMemoryStack* ThreadInstance();
    virtual bool allocate(void** ptr, size_t total, size_t elemSize) = 0;
    virtual uchar* allocate(size_t total) = 0;
    virtual bool deallocate(void* ptr, size_t total) = 0;
};


/*!
 * \brief The LockPolicy class locks calls to the given allocator
 */
template<class Allocator>
class MO_EXPORTS LockPolicy
        : public LockPolicyImpl<Allocator, typename Allocator::MatType>
{
};

/*!
 *  \brief The ref count policy keeps a count of the number of mats that have been
 *         allocated and deallocated so that you can debug when deleting an allocator
 *         prior to releasing all allocated mats
 */
template<class Allocator>
class MO_EXPORTS RefCountPolicy
        : public RefCountPolicyImpl<Allocator, typename Allocator::MatType>
{
public:
    template<class... T> RefCountPolicy(T... args):
        RefCountPolicyImpl<Allocator, typename Allocator::MatType>(args...)
    {
    }
};

template<class SmallAllocator, class LargeAllocator, class MatType>
class MO_EXPORTS CombinedPolicyImpl
        : virtual public SmallAllocator
        , virtual public LargeAllocator
{

};

template<class SmallAllocator, class LargeAllocator>
class MO_EXPORTS CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::cuda::GpuMat>
        : virtual public SmallAllocator
        , virtual public LargeAllocator
{
public:
    CombinedPolicyImpl(size_t threshold);
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
    void Release();
private:
    size_t threshold;
};

template<class SmallAllocator, class LargeAllocator>
class MO_EXPORTS CombinedPolicyImpl<SmallAllocator, LargeAllocator, cv::Mat>
        : virtual public SmallAllocator
        , virtual public LargeAllocator
{
public:
    CombinedPolicyImpl(size_t threshold);
    inline cv::UMatData* allocate(int dims, const int* sizes, int type,
                                  void* data, size_t* step, int flags,
                                  cv::UMatUsageFlags usageFlags) const;
    inline bool allocate(cv::UMatData* data, int accessflags,
                         cv::UMatUsageFlags usageFlags) const;
    inline void deallocate(cv::UMatData* data) const;
    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t num_bytes);
    void Release();
private:
    size_t threshold;
};

template<class SmallAllocator, class LargeAllocator>
class MO_EXPORTS CombinedPolicy
        : public CombinedPolicyImpl<SmallAllocator, LargeAllocator, typename LargeAllocator::MatType>
{
public:
    typedef typename LargeAllocator::MatType MatType;
    CombinedPolicy(size_t threshold = 1*1024*512);
};

/*!
 * \brief The ScopeDebugPolicy class
 */
template<class Allocator, class MatType>
class MO_EXPORTS ScopeDebugPolicy: public virtual Allocator{};

template<class Allocator>
class MO_EXPORTS ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>: public virtual Allocator
{
public:
    inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
    inline void free(cv::cuda::GpuMat* mat);

    inline unsigned char* allocate(size_t num_bytes);
    inline void deallocate(unsigned char* ptr, size_t numBytes);
private:
    std::map<unsigned char*, std::string> scopeOwnership;
    std::map<std::string, size_t> scopedAllocationSize;
};



template<class T> class PinnedStlAllocator
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    template< class U > struct rebind { typedef PinnedStlAllocator<U> other; };
    pointer allocate(size_type n, std::allocator<void>::const_pointer hint)
    {
        return allocate(n);
    }

    pointer allocate(size_type n)
    {
        pointer output = nullptr;
        cudaSafeCall(cudaMallocHost(&output, n*sizeof(pointer)));
        return output;
    }

    void deallocate(pointer ptr, size_type n)
    {
        cudaSafeCall(cudaFreeHost(ptr));
    }
};


template<class T> bool operator==(const PinnedStlAllocator<T>& lhs, const PinnedStlAllocator<T>& rhs)
{
    return &lhs == &rhs;
}
template<class T> bool operator!=(const PinnedStlAllocator<T>& lhs, const PinnedStlAllocator<T>& rhs)
{
    return &lhs != &rhs;
}

// Share pinned pool with CpuPoolPolicy
template<class T> class PinnedStlAllocatorPoolThread
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    template< class U > struct rebind { typedef PinnedStlAllocatorPoolThread<U> other; };
    pointer allocate(size_type n, std::allocator<void>::const_pointer hint)
    {
        return allocate(n);
    }

    pointer allocate(size_type n)
    {
        pointer ptr = nullptr;
        CpuMemoryPool::ThreadInstance()->allocate((void**)&ptr, n*sizeof(T), sizeof(T));
        return ptr;
    }

    void deallocate(pointer ptr, size_type n)
    {
        CpuMemoryPool::ThreadInstance()->deallocate(ptr, n*sizeof(T));
    }
};

template<class T> class PinnedStlAllocatorPoolGlobal
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    template< class U > struct rebind { typedef PinnedStlAllocatorPoolGlobal<U> other; };

    pointer allocate(size_type n, std::allocator<void>::const_pointer hint)
    {
        return allocate(n);
    }

    pointer allocate(size_type n)
    {
        pointer ptr = nullptr;
        CpuMemoryPool::GlobalInstance()->allocate(&ptr, n*sizeof(T), sizeof(T));
        return ptr;
    }

    void deallocate(pointer ptr, size_type n)
    {
        CpuMemoryPool::GlobalInstance()->deallocate(ptr, n*sizeof(T));
    }
};

} // namespace mo
