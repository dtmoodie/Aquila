#include "Aquila/types/ObjectDetection.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
aq::Classification::Classification(const std::string& label_, float confidence_, int classNumber_) :
    label(label_), confidence(confidence_), classNumber(classNumber_)
{

}


