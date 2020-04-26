#pragma once
#include "detection/Category.hpp"
#include "detection/Classification.hpp"
#include <Aquila/core/detail/Export.hpp>

#include <MetaObject/types/small_vec.hpp>

#include <ct/reflect.hpp>

#include <Eigen/Geometry>
#include <opencv2/core/types.hpp>

#include <map>
#include <string>
#include <vector>

namespace aq
{

    AQUILA_EXPORTS void boundingBoxToPixels(cv::Rect2f& bb, cv::Size size);
    AQUILA_EXPORTS void normalizeBoundingBox(cv::Rect2f& bb, cv::Size size);
    AQUILA_EXPORTS void clipNormalizedBoundingBox(cv::Rect2f& bb);

    //////////////////////////////////////////
    /// DetectedObject
    //////////////////////////////////////////
    struct AQUILA_EXPORTS DetectedObject
    {
        DetectedObject(const cv::Rect2f& rect = cv::Rect2f(),
                       const mo::SmallVec<Classification, 5>& cls = mo::SmallVec<Classification, 5>(),
                       unsigned int id = 0,
                       float confidence = 0.0);

        void classify(Classification&& cls);

        cv::Rect2f bounding_box;
        mo::SmallVec<Classification, 5> classifications;
        unsigned int id = 0;
        float confidence = 0.0f;
    };

    bool operator==(const DetectedObject& lhs, const DetectedObject& rhs);

    using DetectedObject2d = DetectedObject;

    //////////////////////////////////////////
    /// DetectedObject3d
    //////////////////////////////////////////
    struct AQUILA_EXPORTS DetectedObject3d
    {
        Eigen::Vector3f size;
        Eigen::Affine3f pose;
        mo::SmallVec<Classification, 5> classifications;
        unsigned int id = 0;
        float confidence = 0.0f;
    };

    //////////////////////////////////////////
    /// TDetectedObjectSet
    //////////////////////////////////////////
    template <class DetType>
    struct AQUILA_EXPORTS TDetectedObjectSet : public std::vector<DetType>
    {
        template <class... Args>
        TDetectedObjectSet(const CategorySet::ConstPtr& cats = CategorySet::ConstPtr())
            : cat_set(cats)
        {
        }

        template <class... Args>
        TDetectedObjectSet(CategorySet::ConstPtr& cats, Args&&... args)
            : std::vector<DetType>(std::forward<Args>(args)...)
            , cat_set(cats)
        {
        }

        template <class... Args>
        TDetectedObjectSet(Args&&... args)
            : std::vector<DetType>(std::forward<Args>(args)...)
        {
        }

        void setCatSet(const CategorySet::ConstPtr& cats)
        {
            cat_set = cats;
            std::vector<DetType>::clear();
        }

        /*template <class AR>
        void serialize(AR& ar)
        {
            ar(cereal::make_nvp("detections", static_cast<std::vector<DetType>&>(*this)));
            ar(CEREAL_NVP(cat_set));
        }*/

        CategorySet::ConstPtr getCatSet() const
        {
            return cat_set;
        }

        CategorySet::ConstPtr cat_set;
    };
    using DetectedObjectSet = TDetectedObjectSet<DetectedObject>;
}

namespace ct
{

    REFLECT_BEGIN(aq::DetectedObject)
        PUBLIC_ACCESS(bounding_box)
        PUBLIC_ACCESS(classifications)
        PUBLIC_ACCESS(id)
        PUBLIC_ACCESS(confidence)
    REFLECT_END;

    REFLECT_BEGIN(aq::DetectedObject3d)
        PUBLIC_ACCESS(size)
        PUBLIC_ACCESS(pose)
        PUBLIC_ACCESS(classifications)
        PUBLIC_ACCESS(id)
        PUBLIC_ACCESS(confidence)
    REFLECT_END;
}
