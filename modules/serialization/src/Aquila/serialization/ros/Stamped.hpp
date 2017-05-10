#pragma once
#include "std_msgs/Header.h"

namespace aq{

template<class T> struct Stamped{
    std_msgs::Header header;
    T data;
};

}

namespace ros{
namespace message_traits{

template<typename M>
struct TimeStamp<aq::Stamped<M>, void>
{
    static ros::Time* pointer(typename boost::remove_const<aq::Stamped<M>>::type &m) { return &m.header.stamp; }
    static ros::Time const* pointer(const aq::Stamped<M>& m) { return &m.header.stamp; }
    static ros::Time value(const aq::Stamped<M>& m) { return m.header.stamp; }
};

template<typename M> struct MD5Sum<aq::Stamped<M>>
{
  static const char* value() { return MD5Sum<M>::value(); }
  static const char* value(const aq::Stamped<M>&) { return value(); }

  static const uint64_t static_value1 = MD5Sum<M>::static_value1;
  static const uint64_t static_value2 = MD5Sum<M>::static_value2;
};

template<typename M> struct Definition<aq::Stamped<M>>
{
  static const char* value() { return Definition<M>::value(); }
  static const char* value(const aq::Stamped<M>&) { return value(); }
};
template<typename M> struct HasHeader<aq::Stamped<M>> : HasHeader<M> {};

}
}
