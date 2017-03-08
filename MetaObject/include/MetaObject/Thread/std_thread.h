#pragma once
#include "Defs.h"
#include <thread>

namespace Signals
{
    size_t SIGNAL_EXPORTS get_thread_id(const std::thread::id& id);
}