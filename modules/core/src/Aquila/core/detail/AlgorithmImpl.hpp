#pragma once
#include "Aquila/core/Algorithm.hpp"
#include <queue>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/circular_buffer.hpp>
#include <type_traits>

namespace aq{
    struct Algorithm::impl{
        struct SyncData{
            SyncData(const boost::optional<mo::Time_t>& ts_, size_t fn_):
                ts(ts_), fn(fn_){}
            boost::optional<mo::Time_t> ts;
            size_t fn = std::numeric_limits<size_t>::max();
        };

        size_t fn = std::numeric_limits<size_t>::max();
        boost::optional<mo::Time_t> ts;
        boost::optional<mo::Time_t> last_ts;
        mo::InputParam* sync_input = nullptr;
        Algorithm::SyncMethod _sync_method;
        std::queue<mo::Time_t> _ts_processing_queue;
        std::queue<size_t> _fn_processing_queue;
        boost::recursive_mutex _mtx;
        std::map<mo::InputParam*, boost::circular_buffer<SyncData>> _buffer_timing_data;
    };
}
