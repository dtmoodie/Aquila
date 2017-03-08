#include "MetaObject/Thread/ThreadHandle.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
#include "MetaObject/Thread/Thread.hpp"
#include "MetaObject/Logging/Profiling.hpp"
using namespace mo;
ThreadHandle::ThreadHandle(Thread* thread, int* ref_count)
{
    _ref_count = ref_count;
    _thread = thread;
    increment();
}
ThreadHandle::~ThreadHandle()
{
    decrement();
}
ThreadHandle::ThreadHandle()
{
    _ref_count = nullptr;
    _thread = nullptr;
}
ThreadHandle::ThreadHandle(const ThreadHandle& other)
{
    _ref_count = other._ref_count;
    _thread = other._thread;
    increment();
}
ThreadHandle::ThreadHandle(ThreadHandle&& other)
{
    _ref_count = other._ref_count;
    _thread = other._thread;
    other._ref_count = nullptr;
    other._thread = nullptr;
}
ThreadHandle& ThreadHandle::operator=(ThreadHandle&& other)
{
    _thread = other._thread;
    this->_ref_count = other._ref_count;
    other._ref_count = nullptr;
    other._thread = nullptr;
    return *this;
}
ThreadHandle& ThreadHandle::operator=(const ThreadHandle& other)
{
    decrement();
    this->_ref_count = other._ref_count;
    this->_thread = other._thread;
    increment();
    return *this;
}
Context* ThreadHandle::GetContext()
{
    if(_thread)
    {
        return _thread->GetContext();
    }
    return nullptr;
}
void ThreadHandle::PushEventQueue(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->PushEventQueue(f);
    }
}
// Work can be stolen and can exist on any thread
void ThreadHandle::PushWork(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->PushWork(f);
    }
}
void ThreadHandle::Start()
{
    if (_thread)
    {
        _thread->Start();
    }
}
void ThreadHandle::Stop()
{
    if(_thread)
    {
        _thread->Stop();
    }
}
void ThreadHandle::SetExitCallback(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->SetExitCallback(f);
    }
}
void ThreadHandle::SetStartCallback(const std::function<void(void)>& f)
{
    if(_thread)
    {
        _thread->SetStartCallback(f);
    }
}

std::shared_ptr<Connection> ThreadHandle::SetInnerLoop(TypedSlot<int(void)>* slot)
{
    if(_thread)
    {
        return _thread->SetInnerLoop(slot);
    }
    return std::shared_ptr<Connection>();
}
void ThreadHandle::decrement()
{
    if (_ref_count)
    {
        --(*_ref_count);
        if (*_ref_count <= 0 && _thread)
     {
            ThreadPool* pool = _thread->GetPool();
            if (pool)
            {
                pool->ReturnThread(_thread);
            }
            else
            {
                delete _thread;
            }
        }
    }
}
void ThreadHandle::increment()
{
    if(_ref_count)
    {
        ++(*_ref_count);
    }
}
size_t ThreadHandle::GetId() const
{
    return _thread->GetId();
}

bool ThreadHandle::IsOnThread() const
{
    return _thread->IsOnThread();
}
bool ThreadHandle::GetIsRunning() const
{
    if(_thread == nullptr)
        return false;
    return _thread->_run && !_thread->_paused;
}
void ThreadHandle::SetThreadName(const std::string& name)
{
    if(_thread)
    {
        if(_thread->IsOnThread())
        {
            mo::SetThreadName(name.c_str());
            mo::SetStreamName(name.c_str(), _thread->GetContext()->GetStream());
            _thread->_name = name;
            _thread->GetContext()->SetName(name);
        }else
        {
            Thread* thread = _thread;
            std::string name_ = name;
            _thread->PushEventQueue([name_, thread, this]()
            {
                mo::SetThreadName(name_.c_str());
                mo::SetStreamName(name_.c_str(), thread->GetContext()->GetStream());
                _thread->_name = name_;
                _thread->GetContext()->SetName(name_);
            });
        }
    }
}
