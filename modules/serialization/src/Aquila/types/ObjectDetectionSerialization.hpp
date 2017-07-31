#pragma once
#include "Aquila/types/ObjectDetection.hpp"
#include <Aquila/serialization/cereal/eigen.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/boost/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
namespace mo {
namespace IO {
    namespace Text {
        namespace imp {
            size_t textSize(const aq::DetectedObject_<2, 1>& obj);
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

namespace aq {
template <class AR>
void Classification::serialize(AR& ar) {
    ar(CEREAL_NVP(label), CEREAL_NVP(confidence), CEREAL_NVP(classNumber));
}

template <int N>
template <class AR>
void DetectedObject_<2, N>::serialize(AR& ar) {
    ar(CEREAL_NVP(bounding_box), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
}

template <class AR>
void DetectedObject_<2, 1>::serialize(AR& ar) {
    ar(CEREAL_NVP(bounding_box), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
}

template <class AR>
void DetectedObject_<2, -1>::serialize(AR& ar) {
    ar(CEREAL_NVP(bounding_box), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
}

template <class AR>
void DetectedObject_<3, 1>::serialize(AR& ar) {
    ar(CEREAL_NVP(pose), CEREAL_NVP(classification), CEREAL_NVP(timestamp), CEREAL_NVP(id));
}
std::ostream& operator<<(std::ostream& os, const aq::DetectedObject& obj);
std::ostream& operator<<(std::ostream& os, const aq::NClassDetectedObject& obj);
}
