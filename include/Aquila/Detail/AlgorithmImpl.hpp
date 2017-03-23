#pragma once
#include "Aquila/Algorithm.h"
#include <queue>
#include <boost/thread/recursive_mutex.hpp>
namespace aq
{
    struct Algorithm::impl
    {
        size_t fn;
        boost::optional<mo::time_t> ts;
        boost::optional<mo::time_t> last_ts;
        mo::InputParameter* sync_input = nullptr;
        Algorithm::SyncMethod _sync_method;
        std::queue<mo::time_t> _ts_processing_queue;
        std::queue<size_t> _fn_processing_queue;
        boost::recursive_mutex _mtx;
    };
}
