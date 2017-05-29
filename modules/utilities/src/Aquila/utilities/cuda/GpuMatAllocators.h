#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <opencv2/core/cuda.hpp>
#include <mutex>
#include <tuple>
#include <list>
#include <map>
#include <memory>
#include <boost/thread.hpp>
#include "MetaObject/core/detail/MemoryBlock.hpp"
namespace aq
{
    AQUILA_EXPORTS cv::cuda::GpuMat::Allocator* GetDefaultBlockMemoryAllocator();
    AQUILA_EXPORTS cv::cuda::GpuMat::Allocator* GetDefaultDelayedDeallocator();
    AQUILA_EXPORTS cv::cuda::GpuMat::Allocator* CreateBlockMemoryAllocator();


    AQUILA_EXPORTS void SetScopeName(const std::string& name);
    AQUILA_EXPORTS const std::string& getScopeName();

    template<class Allocator, class MatType>
    class AQUILA_EXPORTS ScopeDebugPolicy: virtual public Allocator
    {
    };

    template<class Allocator>
    class AQUILA_EXPORTS ScopeDebugPolicy<Allocator, cv::cuda::GpuMat>:
            virtual public Allocator
    {
    public:
        inline bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        inline void free(cv::cuda::GpuMat* mat);

        inline unsigned char* allocate(size_t num_bytes);
        inline void free(unsigned char* ptr);
    protected:
        std::map<std::string, size_t> scopedAllocationSize;
        std::map<unsigned char*, std::string> scopeOwnership;
    };



    class AQUILA_EXPORTS PitchedAllocator : public virtual cv::cuda::GpuMat::Allocator
    {
    public:
        PitchedAllocator();
        void sizeNeeded(int rows, int cols, int elemSize, size_t& sizeNeeded, size_t& stride);
        void Increment(unsigned char* ptr, size_t size);
        void Decrement(unsigned char* ptr, size_t size);
        void SetScope(const std::string& name);
    protected:
        size_t textureAlignment;
        size_t memoryUsage;
        std::map<std::string, size_t> scopedAllocationSize;
        std::map<boost::thread::id, std::string> currentScopeName;
        std::map<unsigned char*, std::string> scopeOwnership;
    };

    class AQUILA_EXPORTS BlockMemoryAllocator: public virtual PitchedAllocator
    {
    public:
        static BlockMemoryAllocator* Instance(size_t initial_size = 10*1024*1024);
        BlockMemoryAllocator(size_t initialBlockSize);
        virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        virtual void free(cv::cuda::GpuMat* mat);
        bool free_impl(cv::cuda::GpuMat* mat);
        virtual unsigned char* allocate(size_t num_bytes);
        virtual void free(unsigned char* ptr);

        size_t initialBlockSize_;
    protected:
        std::list<std::shared_ptr<mo::GpuMemoryBlock>> blocks;
    };

    class AQUILA_EXPORTS DelayedDeallocator : public virtual PitchedAllocator
    {
    public:
        DelayedDeallocator();
        virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        virtual void free(cv::cuda::GpuMat* mat);
        virtual unsigned char* allocate(size_t num_bytes);
        virtual void free(unsigned char* ptr);
        size_t deallocateDelay; // ms
    protected:
        virtual void clear();
        std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocateList;
        std::map<unsigned char*, size_t> current_allocations;
    };

    class AQUILA_EXPORTS CombinedAllocator : public DelayedDeallocator, public BlockMemoryAllocator
    {
    public:
        // Initial memory pool of 10MB
        // Anything over 1MB is allocated by DelayedDeallocator
        static CombinedAllocator* Instance(size_t initial_pool_size = 10*1024*1024,
                                           size_t threshold_level = 1*1024*1024);
        CombinedAllocator(size_t initial_pool_size = 10*1024*1024 ,
                          size_t threshold_level = 1*1024*1024);
        virtual bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
        virtual void free(cv::cuda::GpuMat* mat);
        virtual unsigned char* allocate(size_t num_bytes);
        virtual void free(unsigned char* ptr);
        size_t _threshold_level;
        size_t initialBlockSize_;
    protected:
        std::list<std::shared_ptr<mo::GpuMemoryBlock>> blocks;
    };
}
