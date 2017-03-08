#include "MetaObject/Thread/ThreadHandle.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"

int main()
{
    int call_count = 0;
    mo::TypedSlot<int(void)> inner_loop(
        std::bind([&call_count]()->int
    {
        ++call_count;
        return 100;
    }));
    {
        mo::ThreadHandle handle = mo::ThreadPool::Instance()->RequestThread();
        mo::ThreadHandle handle2 = handle;
        auto connection = handle.SetInnerLoop(&inner_loop);
        handle.Start();
        boost::this_thread::sleep_for(boost::chrono::seconds(10));
        handle.Stop();
    }
    boost::this_thread::sleep_for(boost::chrono::seconds(10));
    mo::ThreadPool::Instance()->Cleanup();
    return 0;
}