#pragma once
#include <MetaObject/Detail/Export.hpp>
#include "ThreadHandle.hpp"
#include "Thread.hpp"
namespace mo
{
    class Thread;
    class MO_EXPORTS ThreadPool
    {
    public:
        static ThreadPool* Instance();
        ThreadHandle RequestThread();
        void Cleanup();
    protected:
        friend class ThreadHandle;
        void ReturnThread(Thread* thread);
    private:
        struct PooledThread
        {
            PooledThread(bool available_, Thread* thread_):
                available(available_), thread(thread_){}
            ~PooledThread();
            bool available = true;
            int ref_count = 0;
            Thread* thread;
        };
        std::list<PooledThread> _threads;
    };
}
