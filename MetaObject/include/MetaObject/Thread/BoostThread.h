#pragma once
#include <cstddef>
#include "MetaObject/Detail/Export.hpp"

namespace boost
{
    class thread;
}
namespace mo
{
    size_t MO_EXPORTS GetThreadId(const boost::thread& thread);
}
