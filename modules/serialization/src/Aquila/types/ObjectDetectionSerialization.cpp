#include "Aquila/types/ObjectDetection.hpp"
#include "ObjectDetectionSerialization.hpp"

#include <Aquila/rcc/external_includes/cv_imgproc.hpp>

#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/params/buffers/NNStreamBuffer.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"
#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && (defined Aquila_EXPORTS)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/params/detail/MetaParamImpl.hpp"

using namespace aq;

INSTANTIATE_META_PARAM(DetectedObject)
INSTANTIATE_META_PARAM(Classification)
INSTANTIATE_META_PARAM(std::vector<DetectedObject>)
INSTANTIATE_META_PARAM(std::vector<DetectedObject3d>)

template AQUILA_EXPORTS void DetectedObject::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive& ar);
template AQUILA_EXPORTS void DetectedObject::serialize<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& ar);

