#include "ObjectDetectionSerialization.hpp"
#include "Aquila/types/ObjectDetection.hpp"
#include <Aquila/rcc/external_includes/cv_imgproc.hpp>

#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/params/buffers/NNStreamBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"

#include <iomanip>
#include <ostream>

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

            template<int N>
            size_t textSize(const aq::DetectedObject2d_<N>& obj) {
                size_t out = 8;

                if (obj.classification.label.size())
                    out += obj.classification.label.size();
                else
                    out += textSize(obj.classification.classNumber);
                out += textSize(obj.boundingBox);
                return out;
            }
            size_t textSize(const aq::DetectedObject2d& obj){
                size_t out = 8;

                if (obj.classification.label.size())
                    out += obj.classification.label.size();
                else
                    out += textSize(obj.classification.classNumber);
                out += textSize(obj.boundingBox);
                return out;
            }
        } // namespace mo::IO::Text::imp
    } // namespace mo::IO::Text
} // namespace mo::IO
} // namespace mo

#include "MetaObject/serialization/TextPolicy.hpp"
namespace aq{
    std::ostream& operator<<(std::ostream& os, const aq::DetectedObject& obj) {
        os << std::setprecision(3) << obj.classification.confidence << " ";
        os << obj.id << " ";
    
        if (obj.classification.label.size())
            os << obj.classification.label;
        else
            os << obj.classification.classNumber;
        os << std::fixed << obj.boundingBox;
        return os;
    }
}

INSTANTIATE_META_PARAM(DetectedObject);
INSTANTIATE_META_PARAM(Classification);
INSTANTIATE_META_PARAM(std::vector<DetectedObject>);
INSTANTIATE_META_PARAM(std::vector<DetectedObject3d>);
namespace aq{
template struct DetectedObject2d_<1>;
template struct DetectedObject2d_<-1>;

//template<> AQUILA_EXPORTS void DetectedObject2d::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive& ar);
//template<> AQUILA_EXPORTS void DetectedObject2d::serialize<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& ar);
}
