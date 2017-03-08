#pragma once

#include <MetaObject/Detail/Export.hpp>
#include <memory>
#include <functional>

namespace mo
{
    class Thread;
    class ThreadPool;
    class Context;
    class ISlot;
    class Connection;
    template<class T> class TypedSlot;
    class MO_EXPORTS ThreadHandle
    {
    public:
        ThreadHandle();
        ThreadHandle(const ThreadHandle& other);
        ThreadHandle(ThreadHandle&& other);

        ~ThreadHandle();
        ThreadHandle& operator=(ThreadHandle&& other);
        ThreadHandle& operator=(const ThreadHandle& other);

        Context* GetContext();
        size_t GetId() const;
        bool IsOnThread() const;
        void PushEventQueue(const std::function<void(void)>& f);
        // Work can be stolen and can exist on any thread
        void PushWork(const std::function<void(void)>& f);
        void Start();
        void Stop();
        bool GetIsRunning() const;
        void SetExitCallback(const std::function<void(void)>& f);
        void SetStartCallback(const std::function<void(void)>& f);
        void SetThreadName(const std::string& name);
        std::shared_ptr<Connection> SetInnerLoop(TypedSlot<int(void)>* slot);
    protected:
        friend class ThreadPool;
        ThreadHandle(Thread* thread, int* ref_count);
        Thread* _thread;
        int* _ref_count;
        void decrement();
        void increment();
    };
}
