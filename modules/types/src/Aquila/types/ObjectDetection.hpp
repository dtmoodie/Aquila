#pragma once
#include "MetaObject/params/IParam.hpp"
#include <Aquila/core/detail/Export.hpp>
#include <Eigen/Geometry>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

namespace aq {

struct AQUILA_EXPORTS Classification {
    Classification(const std::string& label_ = "", float confidence_ = 0, int classNumber_ = -1);
    std::string label;
    float       confidence;
    int         classNumber;

    template <class AR>
    void serialize(AR& ar);
};

template <int D, int N>
struct DetectedObject_ {
};

template <int         N>
struct AQUILA_EXPORTS DetectedObject_<2, N> {
    typedef std::vector<DetectedObject_<2, N> > DetectionList;
    enum { Dims = 2 };

    template <class AR>
    void serialize(AR& ar);

    template <int N1>
    DetectedObject_<2, N>& operator=(const DetectedObject_<2, N1>& other) {
        this->bounding_box = other.bounding_box;
        this->timestamp    = other.timestamp;
        this->framenumber  = other.framenumber;
        this->id           = other.id;
        return *this;
    }

    Classification              classification[N];
    cv::Rect2f                  bounding_box;
    boost::optional<mo::Time_t> timestamp;
    size_t                      framenumber = 0;
    int                         id          = 0;
};

template <>
struct AQUILA_EXPORTS DetectedObject_<2, 1> {
    typedef std::vector<DetectedObject_<2, 1> > DetectionList;
    enum { Dims = 2 };

    template <class AR>
    void serialize(AR& ar);

    Classification              classification;
    cv::Rect2f                  bounding_box;
    boost::optional<mo::Time_t> timestamp;
    size_t                      framenumber = 0;
    int                         id          = 0;
};

template <>
struct AQUILA_EXPORTS DetectedObject_<2, -1> {
    typedef std::vector<DetectedObject_<2, -1> > DetectionList;
    enum { Dims = 2 };

    DetectedObject_<2, -1>() {}

    DetectedObject_<2, -1>(const DetectedObject_<2, 1>& other) {
        this->bounding_box = other.bounding_box;
        this->timestamp    = other.timestamp;
        this->framenumber  = other.framenumber;
        this->id           = other.id;
        this->classification.push_back(other.classification);
    }

    template <int N1>
    DetectedObject_<2, -1>(const DetectedObject_<2, N1>& other) {
        this->bounding_box = other.bounding_box;
        this->timestamp    = other.timestamp;
        this->framenumber  = other.framenumber;
        this->id           = other.id;
        for (int i = 0; i < N1; ++i) {
            this->classification.push_back(other.classification[i]);
        }
    }

    template <class AR>
    void serialize(AR& ar);

    DetectedObject_<2, -1>& operator=(const DetectedObject_<2, 1>& other) {
        this->bounding_box = other.bounding_box;
        this->timestamp    = other.timestamp;
        this->framenumber  = other.framenumber;
        this->id           = other.id;
        this->classification.clear();
        this->classification.push_back(other.classification);
        return *this;
    }

    template <int N1>
    DetectedObject_<2, -1>& operator=(const DetectedObject_<2, N1>& other) {
        this->bounding_box = other.bounding_box;
        this->timestamp    = other.timestamp;
        this->framenumber  = other.framenumber;
        this->id           = other.id;
        this->classification.clear();
        for (int i = 0; i < N1; ++i) {
            this->classification.push_back(other.classification[i]);
        }
        return *this;
    }

    std::vector<Classification> classification;
    cv::Rect2f                  bounding_box;
    boost::optional<mo::Time_t> timestamp;
    size_t                      framenumber = 0;
    int                         id          = 0;
};

typedef DetectedObject_<2, 1>  DetectedObject;
typedef DetectedObject_<2, 1>  DetectedObject2d;
typedef DetectedObject_<2, -1> NClassDetectedObject;

template <int N>
using DetectedObject2d_ = DetectedObject_<2, N>;

template <int         N>
struct AQUILA_EXPORTS DetectedObject_<3, N> {
    typedef std::vector<DetectedObject_<3, N> > DetectionList;
    enum { Dims = 3 };

    template <class AR>
    void serialize(AR& ar);

    Classification              classification[N];
    Eigen::Affine3d             pose;
    Eigen::Vector3d             size;
    boost::optional<mo::Time_t> timestamp;
    size_t                      framenumber = 0;
    int                         id          = 0;
};

template <>
struct AQUILA_EXPORTS DetectedObject_<3, 1> {
    typedef std::vector<DetectedObject_<3, 1> > DetectionList;
    enum { Dims = 3 };

    template <class AR>
    void serialize(AR& ar);

    Classification              classification;
    Eigen::Affine3d             pose;
    Eigen::Vector3d             size;
    boost::optional<mo::Time_t> timestamp;
    size_t                      framenumber = 0;
    int                         id          = 0;
};

template <int N>
using DetectedObject3d_ = DetectedObject_<3, N>;
typedef DetectedObject3d_<1> DetectedObject3d;
}
