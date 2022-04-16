#ifndef AQUILA_TYPES_OBJECT_TRACKING_UTILS_HPP
#define AQUILA_TYPES_OBJECT_TRACKING_UTILS_HPP
#include "ObjectDetection.hpp"

namespace aq
{
    template <class T, class U>
    cv::Vec<T, 2> center(const cv::Rect_<U>& rect)
    {
        return cv::Vec<T, 2>(rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    template <class T, class U>
    float iou(const cv::Rect_<T>& r1, const cv::Rect_<U>& r2)
    {
        float intersection = (r1 & r2).area();
        float rect_union = (r1 | r2).area();
        return intersection / rect_union;
    }

    /*!
     * \brief extentDistance
     * \param meas1
     * \param meas2
     * \param dims
     * \return
     */
    template <int N>
    float extentDistance(const cv::Mat& measurement, const cv::Mat& state)
    {
        float value = 0.0f;
        // Position state, size state, position measurement, size measurement
        cv::Vec<float, N> Ps, Ss, Pm, Sm;
        for (int i = 0; i < N; ++i)
        {
            Ps[i] = state.at<float>(i);
            Ss[i] = state.at<float>(i + N * 2);
            Pm[i] = measurement.at<float>(i);
            Sm[i] = measurement.at<float>(i + N);
        }
        // Calculate a normalized metric such that the if the centroid of measurement
        // is within the bounds of state the score will be between 0 and 1
        //

           //             _______________
           //             |              |
           //             |              |
           //      _____________.Pm      | Score == 1
           //      |      |     |        |
           //      |      |______________|
           //      |   Ps .     |
           //      |            |
           //      |____________|_______________
           //                   |              |
           //                   |              |
           //                   |      Pm      | Score == 2
           //                   |              |
           //                   |_______________
           //  The score should be 0 if Ps and Ss lie ontop of each other
        for (int i = 0; i < N; ++i)
        {
            value += abs(Ps[i] - Pm[i]) / (Ss[i] / 2);
        }
        // Calculate a score based on the % change in size of the object
        for (int i = 0; i < N; ++i)
        {
            value += abs(Ss[i] - Sm[i]) / (Ss[i]);
        }
        // Normalize value by the number of dimensions
        value /= (float)N;
        return value;
    }
}

#endif // AQUILA_TYPES_OBJECT_TRACKING_UTILS_HPP
