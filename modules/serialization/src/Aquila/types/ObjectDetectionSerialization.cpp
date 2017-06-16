#include "Aquila/types/ObjectDetection.hpp"
#include "ObjectDetectionSerialization.hpp"

#include <Aquila/rcc/external_includes/cv_imgproc.hpp>

#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/params/buffers/NNStreamBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && (defined Aquila_EXPORTS)
#define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define MO_EXPORTS __attribute__((visibility("default")))
#else
#define MO_EXPORTS
#endif
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include <MetaObject/core/detail/Counter.hpp>
using namespace aq;
namespace mo {
namespace IO {
    namespace Text {
        namespace imp {

            bool Serialize_imp(std::ostream& os, DetectedObject const& obj, mo::_counter_<10> dummy) {
                (void)dummy;
                os << obj.classification.confidence << " ";
                os << obj.id << " ";
                if (obj.classification.label.size())
                    os << obj.classification.label;
                else
                    os << obj.classification.classNumber;
                os << obj.boundingBox;
                return true;
            }

            bool DeSerialize_imp(std::stringstream& ss, DetectedObject& obj, mo::_counter_<10> dummy) {
                (void)dummy;
                (void)ss;
                (void)obj;
                return false;
            }
            inline size_t textSize(int value);
            template <class T>
            size_t textSize(const cv::Rect_<T>& bb) {
                size_t out = 20;

                return out;
            }

            size_t textSize(const DetectedObject2d& obj) {
                size_t out = 8;

                if (obj.classification.label.size())
                    out += obj.classification.label.size();
                else
                    out += textSize(obj.classification.classNumber);
                out += textSize(obj.boundingBox);
                return out;
            }
        }
    }
}
}
#include "MetaObject/serialization/TextPolicy.hpp"
std::ostream& operator<<(std::ostream& os, const DetectedObject& obj) {
    os << std::setprecision(3) << obj.classification.confidence << " ";
    os << obj.id << " ";

    if (obj.classification.label.size())
        os << obj.classification.label;
    else
        os << obj.classification.classNumber;
    os << std::fixed << obj.boundingBox;
    return os;
}

INSTANTIATE_META_PARAM(DetectedObject);
INSTANTIATE_META_PARAM(Classification);
INSTANTIATE_META_PARAM(std::vector<DetectedObject>);
INSTANTIATE_META_PARAM(std::vector<DetectedObject3d>);

template AQUILA_EXPORTS void DetectedObject::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive& ar);
template AQUILA_EXPORTS void DetectedObject::serialize<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& ar);
