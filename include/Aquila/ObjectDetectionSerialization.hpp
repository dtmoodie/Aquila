#pragma once
#include "Aquila/IO/cereal_eigen.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/boost/optional.hpp>
#include "ObjectDetection.hpp"
template<class AR>
void aq::Classification::serialize(AR& ar)
{
    ar(CEREAL_NVP(label), CEREAL_NVP(confidence), CEREAL_NVP(classNumber));
}

template<class AR>
void aq::DetectedObject2d::serialize(AR& ar)
{
    ar(CEREAL_NVP(boundingBox), CEREAL_NVP(detections), CEREAL_NVP(timestamp), CEREAL_NVP(id));
}

template<class AR>
void aq::DetectedObject3d::serialize(AR& ar)
{
    ar(CEREAL_NVP(pose), CEREAL_NVP(detections), CEREAL_NVP(timestamp), CEREAL_NVP(id));
}
