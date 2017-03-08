#include "MetaObject/Thread/Thread.hpp"
#include "MetaObject/Signals/TypedSignalRelay.hpp"
#include "MetaObject/Context.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"
#include "MetaObject/Thread/BoostThread.h"
#include "MetaObject/Thread/ThreadRegistry.hpp"
using namespace mo;


void Thread::PushEventQueue(const std::function<void(void)>& f)
{
    boost::recursive_mutex::scoped_lock lock(_mtx);
    _event_queue.push(f);
    _cv.notify_all();
}
// Work can be stolen and can exist on any thread
void Thread::PushWork(const std::function<void(void)>& f)
{
    boost::recursive_mutex::scoped_lock lock(_mtx);
    _work_queue.push(f);
    _cv.notify_all();
}
void Thread::Start()
{
    _run = true;
    _cv.notify_all();
}
void Thread::Stop()
{
    _run = false;
    _thread.interrupt();
    int wait_count = 0;
    while(!_paused)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        ++wait_count;
        if(wait_count % 1000 == 0)
            LOG(warning) << "Waited 1000 seconds for " << this->_name << " to stop";
    }
}
void Thread::SetExitCallback(const std::function<void(void)>& f)
{
    _on_exit = f;
}
void Thread::SetStartCallback(const std::function<void(void)>& f)
{
    _on_start = f;
}
std::shared_ptr<Connection> Thread::SetInnerLoop(TypedSlot<int(void)>* slot)
{
    return slot->Connect(_inner_loop);
}
ThreadPool* Thread::GetPool() const
{
    return _pool;
}
Context* Thread::GetContext()
{
    boost::recursive_mutex::scoped_lock lock(_mtx);
    if(!_ctx)
    {
        _cv.wait_for(lock, boost::chrono::seconds(5));
    }
    return _ctx;
}

Thread::Thread()
{
    _pool = nullptr;
    _ctx = nullptr;
    _inner_loop.reset(new mo::TypedSignalRelay<int(void)>());
    _quit = false;
    _thread = boost::thread(&Thread::Main, this);
    _paused = false;

}
Thread::Thread(ThreadPool* pool)
{
    _inner_loop.reset(new mo::TypedSignalRelay<int(void)>());
    _pool = pool;
    _ctx = nullptr;
    _quit = false;
    _thread = boost::thread(&Thread::Main, this);

}

Thread::~Thread()
{
    _quit = true;
    _run = false;
    LOG(info) << "Shutting down " << this->_name << " thread";
    _thread.interrupt();
    _thread.timed_join(boost::posix_time::time_duration(0,0,10));
}

void Thread::Main()
{
    mo::Context ctx;
    {
        boost::recursive_mutex::scoped_lock lock(_mtx);
        _ctx = &ctx;
        lock.unlock();
        _cv.notify_all();
    }

    mo::Context::SetDefaultThreadContext(_ctx);
    if(_on_start)
        _on_start();

    while(!boost::this_thread::interruption_requested())
    {
        {
            boost::recursive_mutex::scoped_lock lock(_mtx);
            while (_work_queue.size())
            {
                _work_queue.back()();
                _work_queue.pop();
            }
            while(_event_queue.size())
            {
                _event_queue.back()();
                _event_queue.pop();
            }
            mo::ThreadSpecificQueue::RunOnce();
        }
        if(_run)
        {
            try
            {
                if (_inner_loop->HasSlots())
                {
                    int delay = (*_inner_loop)();
                    if (delay)
                    {
                        boost::this_thread::sleep_for(boost::chrono::milliseconds(delay));
                    }
                }
            }catch(boost::thread_interrupted& e)
            {
                if(_quit)
                {
                    throw e;
                }
            }catch(...)
            {
                boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            }
        }else
        {
            while(!_run)
            {

                try
                {
                    _paused = true;
                    boost::recursive_mutex::scoped_lock lock(_mtx);
                    _cv.wait_for(lock, boost::chrono::milliseconds(10));
                    {
                        while (_work_queue.size())
                        {
                            _work_queue.back()();
                            _work_queue.pop();
                        }
                        while(_event_queue.size())
                        {
                            _event_queue.back()();
                            _event_queue.pop();
                        }
                        mo::ThreadSpecificQueue::RunOnce();
                    }
                }catch(boost::thread_interrupted& e)
                {
                    if(_quit)
                    {
                        throw e;
                    }
                }
            }
            if(_on_start)
                _on_start();
            _paused = false;
        }
    }
    _paused = true;
    LOG(debug) << _name << " Thread exiting";
    if(_on_exit)
        _on_exit();
}
size_t Thread::GetId() const
{
    return GetThreadId(_thread);
}
bool Thread::IsOnThread() const
{
    return GetId() == GetThisThread();
}
