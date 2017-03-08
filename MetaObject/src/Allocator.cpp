#include "MetaObject/Detail/AllocatorImpl.hpp"
#include <ctime>


using namespace mo;
boost::thread_specific_ptr<Allocator> thread_specific_allocator;
boost::thread_specific_ptr<std::string> current_scope;

thread_local cv::MatAllocator* t_cpuAllocator = nullptr;
             cv::MatAllocator* g_cpuAllocator = nullptr;

thread_local cv::cuda::GpuMat::Allocator* t_gpuAllocator = nullptr;
             cv::cuda::GpuMat::Allocator* g_gpuAllocator = nullptr;

cv::UMatData* CpuAllocatorThreadAdapter::allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step, int flags,
                       cv::UMatUsageFlags usageFlags) const
{
    if(t_cpuAllocator)
    {
        return t_cpuAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
    }
    else
    {
        if(g_cpuAllocator == nullptr)
        {
            g_cpuAllocator = mo::Allocator::GetThreadSafeAllocator();
        }
        return g_cpuAllocator->allocate(dims, sizes, type, data, step, flags, usageFlags);
    }
}

bool CpuAllocatorThreadAdapter::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    if(t_cpuAllocator)
    {
        return t_cpuAllocator->allocate(data, accessflags, usageFlags);
    }
    else
    {
        if(g_cpuAllocator == nullptr)
        {
            g_cpuAllocator = mo::Allocator::GetThreadSafeAllocator();
        }
        return g_cpuAllocator->allocate(data, accessflags, usageFlags);
    }
}

void CpuAllocatorThreadAdapter::deallocate(cv::UMatData* data) const
{
    if(t_cpuAllocator)
    {
        t_cpuAllocator->deallocate(data);
    }
    else
    {
        if(g_cpuAllocator == nullptr)
        {
            g_cpuAllocator = mo::Allocator::GetThreadSafeAllocator();
        }
        g_cpuAllocator->deallocate(data);
    }
}

void CpuAllocatorThreadAdapter::SetThreadAllocator(cv::MatAllocator* allocator)
{
    t_cpuAllocator = allocator;
}

void CpuAllocatorThreadAdapter::SetGlobalAllocator(cv::MatAllocator* allocator)
{
    g_cpuAllocator = allocator;
}

bool GpuAllocatorThreadAdapter::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    if(t_gpuAllocator)
    {
        return t_gpuAllocator->allocate(mat, rows, cols, elemSize);
    }else
    {
        if(g_gpuAllocator == nullptr)
        {
            g_gpuAllocator = mo::Allocator::GetThreadSafeAllocator();
        }
        return g_gpuAllocator->allocate(mat, rows, cols, elemSize);
    }
}


void GpuAllocatorThreadAdapter::free(cv::cuda::GpuMat* mat)
{
    if(t_gpuAllocator)
    {
        t_gpuAllocator->free(mat);
    }else
    {
        if(g_gpuAllocator == nullptr)
        {
            g_gpuAllocator = mo::Allocator::GetThreadSafeAllocator();
        }
        g_gpuAllocator->free(mat);
    }
}

void GpuAllocatorThreadAdapter::SetThreadAllocator(cv::cuda::GpuMat::Allocator* allocator)
{
    t_gpuAllocator = allocator;
}

void GpuAllocatorThreadAdapter::SetGlobalAllocator(cv::cuda::GpuMat::Allocator* allocator)
{
    g_gpuAllocator = allocator;
}

void mo::SetScopeName(const std::string& name)
{
    if (current_scope.get() == nullptr)
    {
        current_scope.reset(new std::string());
    }
    *current_scope = name;
}

const std::string& mo::GetScopeName()
{
    if (current_scope.get() == nullptr)
    {
        current_scope.reset(new std::string());
    }
    return *current_scope;
}

class CpuMemoryPoolImpl: public CpuMemoryPool
{
public:
    CpuMemoryPoolImpl(size_t initial_size = 1e8):
        _initial_block_size(initial_size),
        total_usage(0)
    {
        blocks.emplace_back(new mo::CpuMemoryBlock(_initial_block_size));
    }

    bool allocate(void** ptr, size_t total, size_t elemSize)
    {
        int index = 0;
        unsigned char* _ptr;
        for (auto& block : blocks)
        {
            _ptr = block->allocate(total, elemSize);
            if (_ptr)
            {
                *ptr = _ptr;
                LOG(trace) << "Allocating " << total << " bytes from pre-allocated memory block number "
                           << index << " at address: " << (void*)_ptr;
                return true;
            }
            ++index;
        }
        LOG(trace) << "Creating new block of page locked memory for allocation.";
        blocks.push_back(
            std::shared_ptr<mo::CpuMemoryBlock>(
                new mo::CpuMemoryBlock(std::max(_initial_block_size / 2, total))));
        _ptr = (*blocks.rbegin())->allocate(total, elemSize);
        if (_ptr)
        {
            LOG(debug) << "Allocating " << total
                       << " bytes from newly created memory block at address: " << (void*)_ptr;
            *ptr = _ptr;
            return true;
        }
        return false;
    }
    uchar* allocate(size_t num_bytes)
    {
        int index = 0;
        unsigned char* _ptr;
        for (auto& block : blocks)
        {
            _ptr = block->allocate(num_bytes, sizeof(uchar));
            if (_ptr)
            {
                LOG(trace) << "Allocating " << num_bytes << " bytes from pre-allocated memory block number "
                           << index << " at address: " << (void*)_ptr;
                return _ptr;
            }
            ++index;
        }
        LOG(trace) << "Creating new block of page locked memory for allocation.";
        blocks.push_back(
            std::shared_ptr<mo::CpuMemoryBlock>(
                new mo::CpuMemoryBlock(std::max(_initial_block_size / 2, num_bytes))));
        _ptr = (*blocks.rbegin())->allocate(num_bytes, sizeof(uchar));
        if (_ptr)
        {
            LOG(debug) << "Allocating " << num_bytes
                       << " bytes from newly created memory block at address: " << (void*)_ptr;
            return _ptr;
        }
        return nullptr;
    }
    bool deallocate(void* ptr, size_t total)
    {
        for (auto itr : blocks)
        {
            if (ptr >= itr->Begin() && ptr < itr->End())
            {
                LOG(trace) << "Releasing memory block of size "
                                 << total << " at address: " << ptr;
                if (itr->deAllocate((unsigned char*)ptr))
                {
                    return true;
                }
            }
        }
        return false;
    }
private:
    size_t total_usage;
    size_t _initial_block_size;
    std::list<std::shared_ptr<mo::CpuMemoryBlock>> blocks;
};

class mt_CpuMemoryPoolImpl: public CpuMemoryPoolImpl
{
public:
    bool allocate(void** ptr, size_t total, size_t elemSize)
    {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryPoolImpl::allocate(ptr, total, elemSize);
    }
    uchar* allocate(size_t total)
    {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryPoolImpl::allocate(total);
    }

    bool deallocate(void* ptr, size_t total)
    {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryPoolImpl::deallocate(ptr, total);
    }
private:
    boost::mutex mtx;
};
CpuMemoryPool* CpuMemoryPool::GlobalInstance()
{
    static CpuMemoryPool* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new mt_CpuMemoryPoolImpl();
    }
    return g_inst;
}

CpuMemoryPool* CpuMemoryPool::ThreadInstance()
{
    static boost::thread_specific_ptr<CpuMemoryPool> g_inst;
    if(g_inst.get() == nullptr)
    {
        g_inst.reset(new CpuMemoryPoolImpl());
    }
    return g_inst.get();
}

class CpuMemoryStackImpl: public CpuMemoryStack
{
public:
    typedef cv::Mat MatType;
    CpuMemoryStackImpl(size_t delay):
        deallocation_delay(delay) {}

    ~CpuMemoryStackImpl()
    {
        cleanup(true, true);
    }

    bool allocate(void** ptr, size_t total, size_t elemSize)
    {
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr)
        {
            if(std::get<2>(*itr) == total)
            {
                *ptr = std::get<0>(*itr);
                deallocate_stack.erase(itr);
                LOG(trace) << "[CPU] Reusing memory block of size "
                           << total / (1024 * 1024) << " MB. Total usage: "
                           << total_usage /(1024*1024) << " MB";
                return true;
            }
        }
        this->total_usage += total;
        LOG(trace) << "[CPU] Allocating block of size "
                   << total / (1024 * 1024) << " MB. Total usage: "
                   << total_usage / (1024 * 1024) << " MB";
        CV_CUDEV_SAFE_CALL(cudaMallocHost(ptr, total));
        return true;
    }
    uchar* allocate(size_t total)
    {
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end(); ++itr)
        {
            if(std::get<2>(*itr) == total)
            {

                deallocate_stack.erase(itr);
                LOG(trace) << "[CPU] Reusing memory block of size "
                           << total / (1024 * 1024) << " MB. Total usage: "
                           << total_usage /(1024*1024) << " MB";
                return std::get<0>(*itr);
            }
        }
        this->total_usage += total;
        LOG(trace) << "[CPU] Allocating block of size "
                   << total / (1024 * 1024) << " MB. Total usage: "
                   << total_usage / (1024 * 1024) << " MB";
        uchar* ptr = nullptr;
        CV_CUDEV_SAFE_CALL(cudaMallocHost(&ptr, total));
        return ptr;
    }

    bool deallocate(void* ptr, size_t total)
    {
        LOG(trace) << "Releasing " << total / (1024 * 1024) << " MB to lazy deallocation pool";
        deallocate_stack.emplace_back((unsigned char*)ptr, clock(), total);
        cleanup();
        return true;
    }
private:
    void cleanup(bool force  = false, bool destructor = false)
    {
        auto time = clock();
        if (force)
            time = 0;
        for (auto itr = deallocate_stack.begin(); itr != deallocate_stack.end();)
        {
            if((time - std::get<1>(*itr)) > deallocation_delay)
            {
                total_usage -= std::get<2>(*itr);
                if(!destructor)
                {
#ifdef _MSC_VER
                LOG(trace) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
                    << " MB. Which was stale for " << time - std::get<1>(*itr)
                    << " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
#else
                LOG(trace) << "[CPU] DeAllocating block of size " << std::get<2>(*itr) / (1024 * 1024)
                    << " MB. Which was stale for " << (time - std::get<1>(*itr)) / 1000
                    << " ms. Total usage: " << total_usage / (1024 * 1024) << " MB";
#endif
                }
                CV_CUDEV_SAFE_CALL(cudaFreeHost((void*)std::get<0>(*itr)));
                itr = deallocate_stack.erase(itr);
            }else
            {
                ++itr;
            }
        }
    }
    size_t total_usage;
    size_t deallocation_delay;
    std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_stack;
};

class mt_CpuMemoryStackImpl: public CpuMemoryStackImpl
{
public:
    mt_CpuMemoryStackImpl(size_t delay):
        CpuMemoryStackImpl(delay){}
    bool allocate(void** ptr, size_t total, size_t elemSize)
    {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryStackImpl::allocate(ptr, total, elemSize);
    }
    uchar* allocate(size_t total)
    {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryStackImpl::allocate(total);
    }

    bool deallocate(void* ptr, size_t total)
    {
        boost::mutex::scoped_lock lock(mtx);
        return CpuMemoryStackImpl::deallocate(ptr, total);
    }
private:
    boost::mutex mtx;
};

CpuMemoryStack* CpuMemoryStack::GlobalInstance()
{
    static CpuMemoryStack* g_inst = nullptr;
    if(g_inst == nullptr)
    {
/*#ifdef _MSC_VER
        g_inst = new mt_CpuMemoryStackImpl(1000);
#else

        g_inst = new RefCountPolicy<mt_CpuMemoryStackImpl>(1000*1000);
#endif*/
        g_inst = new mt_CpuMemoryStackImpl(1.5 * CLOCKS_PER_SEC);
    }
    return g_inst;
}

CpuMemoryStack* CpuMemoryStack::ThreadInstance()
{
    static boost::thread_specific_ptr<CpuMemoryStack> g_inst;
    if(g_inst.get() == nullptr)
    {
/*#ifdef _MSC_VER
        g_inst.reset(new CpuMemoryStackImpl(1000));
#else
        g_inst.reset(new RefCountPolicy<CpuMemoryStackImpl>(1000*1000));
#endif*/
        g_inst.reset(new CpuMemoryStackImpl(1.5 * CLOCKS_PER_SEC));

    }
    return g_inst.get();
}

Allocator* Allocator::GetThreadSafeAllocator()
{
    static Allocator* g_inst = nullptr;
    if(g_inst == nullptr)
    {
        g_inst = new mt_UniversalAllocator_t();
    }
    return g_inst;
}

Allocator* Allocator::GetThreadSpecificAllocator()
{
    if(thread_specific_allocator.get() == nullptr)
    {
        thread_specific_allocator.reset(new mt_UniversalAllocator_t());
    }
    return thread_specific_allocator.get();
}

// ================================================================
// CpuStackPolicy
cv::UMatData* CpuStackPolicy::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
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

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data)
    {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    }
    else
    {
        void* ptr = 0;
        //CpuDelayedDeallocationPool::instance()->allocate(&ptr, total, CV_ELEM_SIZE(type));
        CpuMemoryStack::ThreadInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}
uchar* CpuStackPolicy::allocate(size_t total)
{
    return CpuMemoryStack::ThreadInstance()->allocate(total);
}

bool CpuStackPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}
void CpuStackPolicy::deallocate(uchar* ptr, size_t total)
{
    CpuMemoryStack::ThreadInstance()->deallocate(ptr, total);
}

void CpuStackPolicy::deallocate(cv::UMatData* u) const
{
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0)
    {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            //cudaFreeHost(u->origdata);
            CpuMemoryStack::ThreadInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}

// ================================================================
// mt_CpuStackPolicy

cv::UMatData* mt_CpuStackPolicy::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
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

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data)
    {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    }
    else
    {
        void* ptr = 0;
        //CpuDelayedDeallocationPool::instance()->allocate(&ptr, total, CV_ELEM_SIZE(type));
        CpuMemoryStack::GlobalInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool mt_CpuStackPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}

void mt_CpuStackPolicy::deallocate(cv::UMatData* u) const
{
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0)
    {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            //cudaFreeHost(u->origdata);
            CpuMemoryStack::GlobalInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}

// ================================================================
// CpuPoolPolicy
cv::UMatData* CpuPoolPolicy::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
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

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data)
    {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    }
    else
    {
        void* ptr = 0;
        CpuMemoryPool::ThreadInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool CpuPoolPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}

void CpuPoolPolicy::deallocate(cv::UMatData* u) const
{
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0)
    {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            //cudaFreeHost(u->origdata);
            CpuMemoryPool::ThreadInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}
uchar* CpuPoolPolicy::allocate(size_t num_bytes)
{
    return CpuMemoryPool::ThreadInstance()->allocate(num_bytes);
}

void CpuPoolPolicy::deallocate(uchar* ptr, size_t num_bytes)
{

}

// ================================================================
// mt_CpuPoolPolicy
cv::UMatData* mt_CpuPoolPolicy::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
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

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data)
    {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    }
    else
    {
        void* ptr = 0;
        CpuMemoryPool::GlobalInstance()->allocate(&ptr, total, CV_ELEM_SIZE(type));

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool mt_CpuPoolPolicy::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}

void mt_CpuPoolPolicy::deallocate(cv::UMatData* u) const
{
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0)
    {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            //cudaFreeHost(u->origdata);
            CpuMemoryPool::GlobalInstance()->deallocate(u->origdata, u->size);
            u->origdata = 0;
        }

        delete u;
    }
}


cv::UMatData* PinnedAllocator::allocate(int dims, const int* sizes, int type,
    void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
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

    cv::UMatData* u = new cv::UMatData(this);
    u->size = total;

    if (data)
    {
        u->data = u->origdata = static_cast<uchar*>(data);
        u->flags |= cv::UMatData::USER_ALLOCATED;
    }
    else
    {
        void* ptr = 0;
        cudaMallocHost(&ptr, total);

        u->data = u->origdata = static_cast<uchar*>(ptr);
    }

    return u;
}

bool PinnedAllocator::allocate(cv::UMatData* data, int accessflags, cv::UMatUsageFlags usageFlags) const
{
    return false;
}

void PinnedAllocator::deallocate(cv::UMatData* u) const
{
    if (!u)
        return;

    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);

    if (u->refcount == 0)
    {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED))
        {
            cudaFreeHost(u->origdata);   
            u->origdata = 0;
        }

        delete u;
    }
}
