#include <cstddef>
#include "MetaObject/Thread/BoostThread.h"

#include "boost/thread.hpp"


size_t mo::GetThreadId(const boost::thread& thread)
{
    std::stringstream ss;
    ss << std::hex << thread.get_id();
    size_t output;
    ss >> output;
    return output;
}
