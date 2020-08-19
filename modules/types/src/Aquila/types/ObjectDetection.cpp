#include "Aquila/types/ObjectDetection.hpp"
#include <MetaObject/logging/logging.hpp>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace aq
{

    bool operator==(const DetectedObject& lhs, const DetectedObject& rhs)
    {
        return lhs.bounding_box == rhs.bounding_box && lhs.id == rhs.id && lhs.confidence == rhs.confidence;
    }

    void boundingBoxToPixels(cv::Rect2f& bb, cv::Size size)
    {
        if (bb.width <= 1.f && bb.height <= 1.f)
        {
            bb.x *= size.width;
            bb.y *= size.height;
            bb.width *= size.width;
            bb.height *= size.height;
        }
    }

    void normalizeBoundingBox(cv::Rect2f& bb, cv::Size size)
    {
        if (bb.width > 1)
        {
            bb.x /= size.width;
            bb.y /= size.height;
            bb.width /= size.width;
            bb.height /= size.height;
        }
    }

    void clipNormalizedBoundingBox(cv::Rect2f& bb)
    {
        bb.x = std::min(1.0f, std::max<float>(bb.x, 0.0f));
        bb.y = std::min(1.0f, std::max<float>(bb.y, 0.0f));
        bb.width = std::min(1.0f, std::max<float>(bb.width, 0.0f));
        bb.height = std::min(1.0f, std::max<float>(bb.height, 0.0f));
    }

    DetectedObject::DetectedObject(const cv::Rect2f& rect,
                                   const detection::Classifications& cls,
                                   unsigned int id_,
                                   float conf)
        : bounding_box(rect)
        , classifications(cls)
        , id(id_)
        , confidence(conf)
    {
    }

    void DetectedObject::classify(const detection::Classifications& cls)
    {
        const size_t min = std::min(classifications.size(), cls.size());
        for (size_t i = 0; i < min; ++i)
        {
            classifications[i] = cls[i];
        }
    }

    void DetectedObject::classify(const Classification& cls, uint32_t k)
    {
        classifications[k] = cls;
    }

} // namespace aq
