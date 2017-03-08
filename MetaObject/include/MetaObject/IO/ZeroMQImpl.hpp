#pragma once
#include "ZeroMQ.hpp"
using namespace mo;

#ifdef HAVE_ZEROMQ

#include "zmq.hpp"
struct ZeroMQContext::impl
{
    impl():
        ctx(1)
    {
    }
    zmq::context_t ctx;
};



#else



#endif