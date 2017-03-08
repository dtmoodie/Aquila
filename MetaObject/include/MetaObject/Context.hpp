#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include "MetaObject/Detail/Allocator.hpp"
#include <opencv2/core/cuda.hpp>
#include <string>


namespace mo
{
    class Allocator;
    class MO_EXPORTS Context
    {
    public:
        static Context* GetDefaultThreadContext();
        static void SetDefaultThreadContext(Context*  ctx);
        inline Context(const std::string& name = "")
        {
            thread_id = GetThisThread();
            allocator = Allocator::GetThreadSpecificAllocator();
            GpuThreadAllocatorSetter<cv::cuda::GpuMat>::Set(allocator);
            CpuThreadAllocatorSetter<cv::Mat>::Set(allocator);
            if(name.size())
                SetName(name);
        }

        ~Context();
        void SetName(const std::string& name);
        cv::cuda::Stream&      GetStream();
        void                  SetStream(cv::cuda::Stream stream);

        size_t process_id = 0;
        size_t thread_id = 0;
        std::string host_name;
        Allocator* allocator;
    private:
        cv::cuda::Stream stream;
        std::string name;
    };
}
