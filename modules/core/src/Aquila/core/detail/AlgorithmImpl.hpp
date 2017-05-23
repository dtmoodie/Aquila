#pragma once
#include "Aquila/core/Algorithm.hpp"
#include <queue>
#include <boost/thread/recursive_mutex.hpp>
namespace aq
{
    struct Algorithm::impl
    {
        size_t fn = -1;
        boost::optional<mo::Time_t> ts;
        boost::optional<mo::Time_t> last_ts;
        mo::InputParam* sync_input = nullptr;
        Algorithm::SyncMethod _sync_method;
        std::queue<mo::Time_t> _ts_processing_queue;
        std::queue<size_t> _fn_processing_queue;
        boost::recursive_mutex _mtx;
    };
}
