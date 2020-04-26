#pragma once
#include "Aquila/types/SyncedMemory.hpp"
#include "MetaObject/core/Context.hpp"
#include "Stamped.hpp"
#include "ros/serialization.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/cuda.hpp>

namespace ros
{
    namespace serialization
    {

        template <>
        struct Serializer<cv::Mat>
        {
            template <typename Stream>
            inline static void write(Stream& stream, const cv::Mat& mat)
            {
                // Serializer<cv_bridge::CvImage>::write(stream, cv_bridge::CvImage(std_msgs::Header(), "bgr8", mat));
                stream.next(std_msgs::Header());
                stream.next((uint32_t)mat.rows);
                stream.next((uint32_t)mat.cols);
                stream.next(std::string("bgr8"));
                uint8_t is_bigendian = 0;
                stream.next(is_bigendian);
                stream.next((uint32_t)mat.step);
                size_t data_size = mat.step * mat.rows;
                stream.next((uint32_t)data_size);
                if (data_size > 0)
                    memcpy(stream.advance(data_size), mat.data, data_size);
            }

            template <typename Stream>
            inline static void read(Stream& stream, cv::Mat& mat)
            {
                std_msgs::Header hdr;
                stream.next(hdr);
                uint32_t rows, cols;
                stream.next(rows);
                stream.next(cols);
                std::string encoding;
                stream.next(encoding);
                uint8_t is_bigendian;
                stream.next(is_bigendian);
                uint32_t step, data_size;
                stream.next(step);
                stream.next(data_size);
                int type = CV_8UC3; // currently only support BGR8
                cv::Mat tmp(rows, cols, type, stream.advance(data_size), size_t(step));
                tmp.copyTo(mat);
            }

            inline static uint32_t serializedLength(const cv::Mat& mat)
            {
                size_t data_size = mat.step * mat.rows;
                return serializationLength(std_msgs::Header()) + serializationLength(std::string("bgr8")) + 17 +
                       data_size;
            }
        };

        template <>
        struct Serializer<aq::SyncedMemory>
        {
            template <typename Stream>
            inline static void write(Stream& stream, const aq::SyncedMemory& mem)
            {
                auto ctx = mem.getStream();
                if (ctx)
                {
                    const cv::Mat& mat = mem.getMat(ctx->getStream());
                    ctx->getStream().waitForCompletion();
                    Serializer<cv::Mat>::write(stream, mat);
                }
                else
                {
                    const cv::Mat& mat = mem.getMat(cv::cuda::Stream::Null());
                    Serializer<cv::Mat>::write(stream, mat);
                }
            }

            template <typename Stream>
            inline static void read(Stream& stream, aq::SyncedMemory& mem)
            {
                cv::Mat mat;
                Serializer<cv::Mat>::read(stream, mat);
                mem = aq::SyncedMemory(mat);
            }

            inline static uint32_t serializedLength(const aq::SyncedMemory& mem)
            {
                auto ctx = mem.getStream();
                if (ctx)
                {
                    const cv::Mat& mat = mem.getMat(ctx->getStream());
                    return Serializer<cv::Mat>::serializedLength(mat);
                }
                const cv::Mat& mat =
                    mem.getMat(cv::cuda::Stream::Null()); // this could be made async if we add a getMatNoSync call
                return Serializer<cv::Mat>::serializedLength(mat);
            }
        };

        template <>
        struct Serializer<aq::Stamped<aq::SyncedMemory>>
        {
            template <typename Stream>
            inline static void write(Stream& stream, const aq::Stamped<aq::SyncedMemory>& mem)
            {
                auto ctx = mem.data.getStream();
                cv::Mat mat;
                if (ctx)
                {
                    mat = mem.data.getMat(ctx->getStream());
                    ctx->getStream().waitForCompletion();
                }
                else
                {
                    mat = mem.data.getMat(cv::cuda::Stream::Null());
                }
                stream.next(mem.header);
                stream.next((uint32_t)mat.rows); // height
                stream.next((uint32_t)mat.cols); // width
                stream.next(std::string("bgr8"));
                uint8_t is_bigendian = 0;
                stream.next(is_bigendian);
                stream.next((uint32_t)mat.step);
                size_t data_size = mat.step * mat.rows;
                stream.next((uint32_t)data_size);
                if (data_size > 0)
                    memcpy(stream.advance(data_size), mat.data, data_size);
            }

            template <typename Stream>
            inline static void read(Stream& stream, aq::Stamped<aq::SyncedMemory>& mem)
            {
                stream.next(mem.header);
                uint32_t height, width;
                std::string encoding;
                stream.next(height);
                stream.next(width);
                stream.next(encoding);
                uint8_t is_bigendian;
                stream.next(is_bigendian);
                uint32_t step, data_size;
                stream.next(step);
                stream.next(data_size);
                // for now type must be CV_8UC3
                CV_Assert(encoding == "bgr8");
                int type = CV_8UC3;
                cv::Mat tmp((int)height, (int)width, type, stream.advance(data_size), (size_t)step);
                mem.data = aq::SyncedMemory(tmp.clone());
            }

            inline static uint32_t serializedLength(const aq::Stamped<aq::SyncedMemory>& mem)
            {
                const cv::Mat& mat = mem.data.getMatNoSync();
                size_t data_size = mat.step * mat.rows;
                return serializationLength(mem.header) + serializationLength(std::string("bgr8")) + 17 + data_size;
            }
        };

    } // namespace ros::serialization
    namespace message_traits
    {

        template <>
        struct MD5Sum<aq::SyncedMemory>
        {
            static const char* value() { return MD5Sum<sensor_msgs::Image>::value(); }
            static const char* value(const aq::SyncedMemory&) { return value(); }

            static const uint64_t static_value1 = MD5Sum<sensor_msgs::Image>::static_value1;
            static const uint64_t static_value2 = MD5Sum<sensor_msgs::Image>::static_value2;

            // If the definition of sensor_msgs/Image changes, we'll get a compile error here.
            ROS_STATIC_ASSERT(MD5Sum<sensor_msgs::Image>::static_value1 == 0x060021388200f6f0ULL);
            ROS_STATIC_ASSERT(MD5Sum<sensor_msgs::Image>::static_value2 == 0xf447d0fcd9c64743ULL);
        };

        template <>
        struct MD5Sum<aq::Stamped<aq::SyncedMemory>>
        {
            static const char* value() { return MD5Sum<sensor_msgs::Image>::value(); }
            static const char* value(const aq::SyncedMemory&) { return value(); }

            static const uint64_t static_value1 = MD5Sum<sensor_msgs::Image>::static_value1;
            static const uint64_t static_value2 = MD5Sum<sensor_msgs::Image>::static_value2;

            // If the definition of sensor_msgs/Image changes, we'll get a compile error here.
            ROS_STATIC_ASSERT(MD5Sum<sensor_msgs::Image>::static_value1 == 0x060021388200f6f0ULL);
            ROS_STATIC_ASSERT(MD5Sum<sensor_msgs::Image>::static_value2 == 0xf447d0fcd9c64743ULL);
        };

        template <>
        struct TimeStamp<aq::Stamped<aq::SyncedMemory>, void>
        {
            static ros::Time* pointer(aq::Stamped<aq::SyncedMemory>& m) { return &m.header.stamp; }
            static ros::Time const* pointer(const aq::Stamped<aq::SyncedMemory>& m) { return &m.header.stamp; }
            static ros::Time value(const aq::Stamped<aq::SyncedMemory>& m) { return m.header.stamp; }
        };

        template <>
        struct DataType<aq::SyncedMemory>
        {
            static const char* value() { return DataType<sensor_msgs::Image>::value(); }
            static const char* value(const aq::SyncedMemory&) { return value(); }
        };

        template <>
        struct DataType<aq::Stamped<aq::SyncedMemory>>
        {
            static const char* value() { return DataType<sensor_msgs::Image>::value(); }
            static const char* value(const aq::Stamped<aq::SyncedMemory>&) { return value(); }
        };

        template <>
        struct Definition<aq::SyncedMemory>
        {
            static const char* value() { return Definition<sensor_msgs::Image>::value(); }
            static const char* value(const aq::SyncedMemory&) { return value(); }
        };

        template <>
        struct Definition<aq::Stamped<aq::SyncedMemory>>
        {
            static const char* value() { return Definition<sensor_msgs::Image>::value(); }
            static const char* value(const aq::Stamped<aq::SyncedMemory>&) { return value(); }
        };
        template <>
        struct HasHeader<aq::SyncedMemory> : TrueType
        {
        };
        template <>
        struct HasHeader<aq::Stamped<aq::SyncedMemory>> : TrueType
        {
        };

    } // namespace ros::message_traits

    namespace message_operations
    {

        template <>
        struct Printer<aq::SyncedMemory>
        {
            template <typename Stream>
            static void stream(Stream& s, const std::string& indent, const aq::SyncedMemory& m)
            {
                /// @todo Replicate printing for sensor_msgs::Image
                s << indent << m.getSize() << " " << m.getType();
            }
        };

        template <>
        struct Printer<aq::Stamped<aq::SyncedMemory>>
        {
            template <typename Stream>
            static void stream(Stream& s, const std::string& indent, const aq::Stamped<aq::SyncedMemory>& m)
            {
                /// @todo Replicate printing for sensor_msgs::Image
                s << indent << m.header << m.data.getSize() << " " << m.data.getType();
            }
        };

    } // namespace ros::message_traits
} // namespace ros
