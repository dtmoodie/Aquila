#include "Aquila/ObjectDetection.hpp"
#include "Aquila/ObjectDetectionSerialization.hpp"

#include <Aquila/rcc/external_includes/cv_imgproc.hpp>

#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/Map.hpp"
#include "MetaObject/Parameters/Buffers/NNStreamBuffer.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
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
#include "MetaObject/Parameters/detail/MetaParametersDetail.hpp"

using namespace aq;

aq::Classification::Classification(const std::string& label_, float confidence_, int classNumber_) :
    label(label_), confidence(confidence_), classNumber(classNumber_)
{

}

void aq::CreateColormap(cv::Mat& lut, int num_classes, int ignore_class)
{
    lut.create(1, num_classes, CV_8UC3);
    for(int i = 0; i < num_classes; ++i)
        lut.at<cv::Vec3b>(i) = cv::Vec3b(i*180 / num_classes, 200, 255);
    cv::cvtColor(lut, lut, cv::COLOR_HSV2BGR);
    if(ignore_class != -1 && ignore_class < num_classes)
        lut.at<cv::Vec3b>(ignore_class) = cv::Vec3b(0,0,0);
}

INSTANTIATE_META_PARAMETER(DetectedObject)
INSTANTIATE_META_PARAMETER(Classification)
INSTANTIATE_META_PARAMETER(std::vector<DetectedObject>)
INSTANTIATE_META_PARAMETER(std::vector<DetectedObject3d>)

template AQUILA_EXPORTS void DetectedObject::serialize<cereal::JSONInputArchive>(cereal::JSONInputArchive& ar);
template AQUILA_EXPORTS void DetectedObject::serialize<cereal::JSONOutputArchive>(cereal::JSONOutputArchive& ar);

