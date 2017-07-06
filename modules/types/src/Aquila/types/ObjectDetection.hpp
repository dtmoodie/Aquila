#pragma once
#include <Aquila/core/detail/Export.hpp>
#include "MetaObject/params/IParam.hpp"
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
    template<int N>
    struct AQUILA_EXPORTS DetectedObject2d_
    {
        typedef std::vector<DetectedObject2d_<N>> DetectionList;
        enum {Dims = 2};
        Classification classification[N];
        cv::Rect2f boundingBox;
        boost::optional<mo::Time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR> void serialize(AR& ar);
        template<int N1> DetectedObject2d_& operator=(const DetectedObject2d_<N1>& other){
            this->boundingBox = other.boundingBox;
            this->timestamp = other.timestamp;
            this->framenumber = other.framenumber;
            this->id = other.id;
            return *this;
        }
    };
    
    template<>
    struct AQUILA_EXPORTS DetectedObject2d_<1>
    {
        typedef std::vector<DetectedObject2d_<1>> DetectionList;
        enum {Dims = 2};
        Classification classification;
        cv::Rect2f boundingBox;
        boost::optional<mo::Time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR> void serialize(AR& ar);
    };
    
    template<>
    struct AQUILA_EXPORTS DetectedObject2d_<-1>
    {
        typedef std::vector<DetectedObject2d_<-1>> DetectionList;
        enum {Dims = 2};
        std::vector<Classification> classification;
        cv::Rect2f boundingBox;
        boost::optional<mo::Time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR> void serialize(AR& ar);
        DetectedObject2d_& operator=(const DetectedObject2d_<1>& other){
            this->boundingBox = other.boundingBox;
            this->timestamp = other.timestamp;
            this->framenumber = other.framenumber;
            this->id = other.id;
                classification.push_back(other.classification);
            return *this;
        }
        template<int N1> DetectedObject2d_& operator=(const DetectedObject2d_<N1>& other){
            this->boundingBox = other.boundingBox;
            this->timestamp = other.timestamp;
            this->framenumber = other.framenumber;
            this->id = other.id;
            for(int i = 0; i < N1; ++i){
                classification.push_back(other.classification[i]);
            }
            return *this;
        }
    };

    typedef DetectedObject2d_<1> DetectedObject;
    typedef DetectedObject2d_<1> DetectedObject2d;

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
        boost::optional<mo::Time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR> void serialize(AR& ar);
    };
}
