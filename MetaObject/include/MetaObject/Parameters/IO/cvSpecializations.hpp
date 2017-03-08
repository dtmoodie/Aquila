#pragma once
#include <opencv2/core/types.hpp>
#include <cereal/cereal.hpp>
#include <ostream>
namespace cereal
{
    template<class AR, class T> void serialize(AR& ar, cv::Point_<T>& pt)
    {
        ar(make_nvp("x", pt.x), make_nvp("y", pt.y));
    }
    template<class AR, class T> void serialize(AR& ar, cv::Point3_<T>& pt)
    {
        ar(make_nvp("x", pt.x), make_nvp("y", pt.y), make_nvp("z", pt.z));
    }
    template<class AR, class T> void serialize(AR& ar, cv::Scalar_<T>& pt)
    {
        ar(make_nvp("0", pt[0]), make_nvp("1", pt[1]), make_nvp("2", pt[2]), make_nvp("3", pt[3]));
    }
    template<class AR, class T> void serialize(AR& ar, cv::Rect_<T>& rect)
    {
        ar(make_nvp("x", rect.x), make_nvp("y", rect.y), make_nvp("width", rect.width), make_nvp("height", rect.height));
    }

    template<class AR, class T, int N> void serialize(AR& ar, cv::Vec<T, N>& vec)
    {
        for (int i = 0; i < N; ++i)
        {
            ar(make_nvp(std::to_string(i), vec[i]));
        }
    }
}
template<typename T> std::ostream& operator<<(std::ostream& out, cv::Point_<T>& pt)
{
    out << pt.x << ", " << pt.y;
    return out;
}

template<typename T> std::istream& operator >> (std::istream& in, cv::Point_<T>& pt)
{
    char ch;
    in >> pt.x >> ch >> pt.y;
    return in;
}

template<typename T> std::ostream& operator<<(std::ostream& out, cv::Point3_<T>& pt)
{
    out << pt.x << ", " << pt.y << ", " << pt.z;
    return out;
}

template<typename T> std::istream& operator >> (std::istream& in, cv::Point3_<T>& pt)
{
    char ch;
    in >> pt.x >> ch >> pt.y >> ch >> pt.z;
    return in;
}
template<typename T> std::ostream& operator<<(std::ostream& out, cv::Rect_<T>& pt)
{
    out << pt.x << ", " << pt.y << ", " << pt.width << ", " << pt.height;
    return out;
}

template<typename T> std::istream& operator >> (std::istream& in, cv::Rect_<T>& pt)
{
    char ch;
    in >> pt.x >> ch >> pt.y >> ch >> pt.width >> ch >> pt.height;
    return in;
}