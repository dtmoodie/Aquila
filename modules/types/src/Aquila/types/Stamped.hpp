#pragma once
#include <MetaObject/core/detail/Time.hpp>
namespace aq {

template <typename T>
struct FN : public T {
    template <class... U>
    FN(int frame_number_ = 0)
        : frame_number(frame_number_)
    {
    }
    template <class... U>
    FN(int frame_number_, const U&... args)
        : frame_number(frame_number_)
        , T(args...)
    {
    }
    FN& operator=(const T& other)
    {
        T::operator=(other);
        return *this;
    }
    template <class A>
    void serialize(A& ar)
    {
        ar(frame_number);
        ar(*static_cast<T*>(this));
    }

    int frame_number;
};

template <typename T>
struct TS : public T {
    template <class... U>
    TS(U... args)
        : T(args...)
    {
        timestamp = 0.0 * mo::second;
        frame_number = 0;
    }
    template <class... U>
    TS(mo::Time_t ts, size_t fn, U... args)
        : T(args...)
    {
        timestamp = ts;
        this->frame_number = fn;
    }
    template <class A>
    void serialize(A& ar)
    {
        ar(frame_number);
        ar(timestamp);
        ar(*static_cast<T*>(this));
    }

    mo::Time_t timestamp;
    size_t frame_number;
};
}
