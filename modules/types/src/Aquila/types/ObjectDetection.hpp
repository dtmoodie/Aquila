#pragma once
#include <Aquila/core/detail/Export.hpp>
#include "MetaObject/params/IParameter.hpp"
#include <opencv2/core/types.hpp>
#include <Eigen/Geometry>
#include <string>
#include <vector>

namespace aq
{
    struct AQUILA_EXPORTS Classification
    {
        Classification(const std::string& label_ = "", float confidence_ = 0, int classNumber_ = -1);
        std::string label;
        float confidence;
        int classNumber;

        template<class AR> void serialize(AR& ar);
    };

    struct AQUILA_EXPORTS DetectedObject2d
    {
        typedef std::vector<DetectedObject2d> DetectionList;
        enum {Dims = 2};
        Classification classification;
        cv::Rect2f boundingBox;
        boost::optional<mo::time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR> void serialize(AR& ar);
    };

    typedef DetectedObject2d DetectedObject;

    struct AQUILA_EXPORTS DetectedObject3d
    {
        enum {Dims = 3};
        Classification classification;
        /*!
         * \brief pose determines the pose to the center of the object
         */
        Eigen::Affine3d pose;
        /*!
         * \brief size is the centered size of the object
         */
        Eigen::Vector3d size;
        boost::optional<mo::time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR> void serialize(AR& ar);
    };

    void AQUILA_EXPORTS CreateColormap(cv::Mat& lut, int num_classes, int ignore_class = -1);
}
