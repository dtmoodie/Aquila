#include "MetaObject/Thread/ThreadRegistry.hpp"

#include <thread>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <map>
#include <vector>
using namespace mo;

struct ThreadRegistry::impl
{
    std::map<int, std::vector<size_t>> _thread_map;
    std::mutex mtx;
};

using namespace mo;

size_t mo::GetThisThread()
{
    std::stringstream ss;
    ss << std::this_thread::get_id();
    size_t output;
    ss >> output;
    return output;
}



ThreadRegistry::ThreadRegistry()
{
    _pimpl = new impl();
}
ThreadRegistry::~ThreadRegistry()
{
    delete _pimpl;
}

void ThreadRegistry::RegisterThread(ThreadType type, size_t id)
{
    std::lock_guard<std::mutex> lock(_pimpl->mtx);
    auto& threads = _pimpl->_thread_map[type];
    if (std::count(threads.begin(), threads.end(), id) == 0)
        threads.push_back(id);
}

size_t ThreadRegistry::GetThread(int type)
{
    std::lock_guard<std::mutex> lock(_pimpl->mtx);
    // TODO some kind of load balancing for multiple threads of a specific type
    auto current_thread = GetThisThread();
    auto itr = _pimpl->_thread_map.find(type);
    if (itr != _pimpl->_thread_map.end())
    {
        if (itr->second.size())
        {
            if (std::count(itr->second.begin(), itr->second.end(), current_thread) == 0) // If the current thread is not of appropriate type
                return itr->second.back();
        }
    }
    return current_thread;
}

ThreadRegistry* ThreadRegistry::Instance()
{
    static ThreadRegistry* inst = nullptr;
    if(inst == nullptr)
        inst = new ThreadRegistry();
    return inst;
}
