#pragma once
#include "Aquila/types/SyncedMemory.hpp"
#include "MetaObject/Context.hpp"
#include "ros/serialization.h"
#include <opencv2/core/cuda.hpp>
#include <cv_bridge/cv_bridge.h>

namespace ros{
namespace serialization{

template<>
struct Serializer<cv::Mat>{
    template<typename Stream>
    inline static void write(Stream& stream, const cv::Mat& mat){
        Serializer<cv_bridge::CvImage>::write(stream, cv_bridge::CvImage(std_msgs::Header(), "bgr8", mat));
    }

    template<typename Stream>
    inline static void read(Stream& stream, cv::Mat& mat){
        cv_bridge::CvImage cvmat;
        Serializer<cv_bridge::CvImage>::read(stream, cvmat); // TODO handle encoding
        mat = cvmat.image;
    }

    inline static uint32_t serializedLength(const cv::Mat& mat){
        return Serializer<cv_bridge::CvImage>::serializedLength(cv_bridge::CvImage(std_msgs::Header(), "bgr8", mat));
    }
};

template<>
struct Serializer<aq::SyncedMemory>{
    template<typename Stream>
    inline static void write(Stream& stream, const aq::SyncedMemory& mem){
        auto ctx = mem.GetContext();
        if(ctx){
            const cv::Mat& mat = mem.GetMat(ctx->GetStream());
            ctx->GetStream().waitForCompletion();
            Serializer<cv::Mat>::write(stream, mat);
        }else{
            const cv::Mat& mat = mem.GetMat(cv::cuda::Stream::Null());
            Serializer<cv::Mat>::write(stream, mat);
        }
    }

    template<typename Stream>
    inline static void read(Stream& stream, aq::SyncedMemory& mem){
        cv::Mat mat;
        Serializer<cv::Mat>::read(stream, mat);
        mem = aq::SyncedMemory(mat);
    }

    inline static uint32_t serializedLength(const aq::SyncedMemory& mem){
        auto ctx = mem.GetContext();
        if(ctx){
            const cv::Mat& mat = mem.GetMat(ctx->GetStream());
            return Serializer<cv::Mat>::serializedLength(mat);
        }
        const cv::Mat& mat = mem.GetMat(cv::cuda::Stream::Null()); // this could be made async if we add a GetMatNoSync call
        return Serializer<cv::Mat>::serializedLength(mat);
    }
};
} // namespace ros::serialization
namespace message_traits{
template<> struct MD5Sum<aq::SyncedMemory>
{
  static const char* value() { return MD5Sum<sensor_msgs::Image>::value(); }
  static const char* value(const aq::SyncedMemory&) { return value(); }

  static const uint64_t static_value1 = MD5Sum<sensor_msgs::Image>::static_value1;
  static const uint64_t static_value2 = MD5Sum<sensor_msgs::Image>::static_value2;

  // If the definition of sensor_msgs/Image changes, we'll get a compile error here.
  ROS_STATIC_ASSERT(MD5Sum<sensor_msgs::Image>::static_value1 == 0x060021388200f6f0ULL);
  ROS_STATIC_ASSERT(MD5Sum<sensor_msgs::Image>::static_value2 == 0xf447d0fcd9c64743ULL);
};

template<> struct DataType<aq::SyncedMemory>
{
  static const char* value() { return DataType<sensor_msgs::Image>::value(); }
  static const char* value(const aq::SyncedMemory&) { return value(); }
};

template<> struct Definition<aq::SyncedMemory>
{
  static const char* value() { return Definition<sensor_msgs::Image>::value(); }
  static const char* value(const aq::SyncedMemory&) { return value(); }
};
template<> struct HasHeader<aq::SyncedMemory> : TrueType {};

} // namespace ros::message_traits

namespace message_operations {

template<> struct Printer<aq::SyncedMemory>
{
  template<typename Stream>
  static void stream(Stream& s, const std::string& indent, const aq::SyncedMemory& m)
  {
    /// @todo Replicate printing for sensor_msgs::Image
    s << indent << m.GetSize() << " " << m.GetType();
  }
};
} // namespace ros::message_traits
} // namespace ros
