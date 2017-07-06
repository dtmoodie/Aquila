#pragma once
#include <Aquila/serialization/cereal/eigen.hpp>
#include "Aquila/types/ObjectDetection.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/boost/optional.hpp>
namespace mo{
namespace IO{
namespace Text{
namespace imp{
    size_t textSize(const aq::DetectedObject2d& obj);
    template <class T>
    size_t textSize(const cv::Rect_<T>& bb) {
        (void)bb;
        size_t out = 20;
        return out;
    }
}
}
}
}

namespace aq{
template<class AR>
void Classification::serialize(AR& ar){
    ar(CEREAL_NVP(label), CEREAL_NVP(confidence), CEREAL_NVP(classNumber));
}

template<int N> template<class AR>
void DetectedObject2d_<N>::serialize(AR& ar){
    ar(CEREAL_NVP(boundingBox), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
}

template<class AR>
void DetectedObject2d_<1>::serialize(AR& ar){
    ar(CEREAL_NVP(boundingBox), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
}

template<class AR>
void DetectedObject2d_<-1>::serialize(AR& ar){
    ar(CEREAL_NVP(boundingBox), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
}

template<class AR>
void DetectedObject3d::serialize(AR& ar){
    ar(CEREAL_NVP(pose), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id));
}
std::ostream& operator<<(std::ostream& os, const aq::DetectedObject& obj);
}
