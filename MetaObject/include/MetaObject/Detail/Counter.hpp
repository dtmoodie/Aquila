#pragma once

namespace mo
{
    template<int N> struct _counter_
    {
        _counter_<N-1> operator--()
        {
            return _counter_<N-1>();
        }
        _counter_<N+1> operator++()
        {
            return _counter_<N+1>();
        }
    };
}