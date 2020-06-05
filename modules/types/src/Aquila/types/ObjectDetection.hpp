#pragma once
#include "detection/Category.hpp"
#include "detection/Classification.hpp"

#include <Aquila/core/detail/Export.hpp>
#include <Aquila/types/EntityComponentSystem.hpp>

#include <MetaObject/types/small_vec.hpp>

#include <ct/reflect.hpp>

#include <Eigen/Geometry>
#include <opencv2/core/types.hpp>

#include <ct/extensions/DataTable.hpp>

#include <map>
#include <string>
#include <vector>

namespace aq
{

    AQUILA_EXPORTS void boundingBoxToPixels(cv::Rect2f& bb, cv::Size size);
    AQUILA_EXPORTS void normalizeBoundingBox(cv::Rect2f& bb, cv::Size size);
    AQUILA_EXPORTS void clipNormalizedBoundingBox(cv::Rect2f& bb);

    namespace detection
    {
        struct BoundingBox2d : ct::ext::Component, cv::Rect2f
        {
            template <class... ARGS>
            BoundingBox2d(ARGS&&... args)
                : cv::Rect2f(std::forward<ARGS>(args)...)
            {
            }
        };

        struct Confidence : ct::ext::Component
        {
            Confidence(float v = 0.0F)
                : value(v)
            {
            }

            float value;
        };

        struct Classification : ct::ext::Component, mo::SmallVec<aq::Classification, 5>
        {
            template <class... ARGS>
            Classification(ARGS&&... args)
                : mo::SmallVec<aq::Classification, 5>(std::forward<ARGS>(args)...)
            {
            }
        };

        struct Id : ct::ext::Component
        {
            Id(uint32_t v = 0)
                : value(v)
            {
            }

            uint32_t value;
        };

        struct Size3d : Eigen::Vector3f, ct::ext::Component
        {
            template <class... ARGS>
            Size3d(ARGS&&... args)
                : Eigen::Vector3f(std::forward<ARGS>(args)...)
            {
            }
        };

        struct Pose3d : Eigen::Affine3f, ct::ext::Component
        {
            template <class... ARGS>
            Pose3d(ARGS&&... args)
                : Eigen::Affine3f(std::forward<ARGS>(args)...)
            {
            }
        };

    } // namespace detection

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

        detection::BoundingBox2d bounding_box;
        detection::Classification classifications;
        detection::Id id;
        detection::Confidence confidence;
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
    struct AQUILA_EXPORTS TDetectedObjectSet : public TEntityComponentSystem<DetType>
    {
        template <class... Args>
        TDetectedObjectSet(const CategorySet::ConstPtr& cats = CategorySet::ConstPtr())
            : cat_set(cats)
        {
        }

        template <class... Args>
        TDetectedObjectSet(CategorySet::ConstPtr& cats, Args&&... args)
            : TEntityComponentSystem<DetType>(std::forward<Args>(args)...)
            , cat_set(cats)
        {
        }

        template <class... Args>
        TDetectedObjectSet(Args&&... args)
            : TEntityComponentSystem<DetType>(std::forward<Args>(args)...)
        {
        }

        void setCatSet(const CategorySet::ConstPtr& cats)
        {
            cat_set = cats;
            TEntityComponentSystem<DetType>::clear();
        }

        CategorySet::ConstPtr getCatSet() const
        {
            return cat_set;
        }

        CategorySet::ConstPtr cat_set;
    };
    using DetectedObjectSet = TDetectedObjectSet<DetectedObject>;
} // namespace aq

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

    REFLECT_TEMPLATED_DERIVED(aq::TDetectedObjectSet, aq::TEntityComponentSystem<Args...>)
        static void setCats(DataType & obj, aq::CategorySet::Ptr cats)
        {
            obj.setCatSet(std::move(cats));
        }
        PROPERTY(cats, &DataType::getCatSet, &setCats)
    REFLECT_END;

    REFLECT_DERIVED(aq::detection::BoundingBox2d, cv::Rect2f)
    REFLECT_END;

    REFLECT_BEGIN(aq::detection::Confidence)
        PUBLIC_ACCESS(value)
    REFLECT_END;

    REFLECT_DERIVED(aq::detection::Classification, mo::SmallVec<aq::Classification, 5>)
        MEMBER_FUNCTION(erase)
    REFLECT_END;

    REFLECT_BEGIN(aq::detection::Id)
        PUBLIC_ACCESS(value)
    REFLECT_END;

    REFLECT_DERIVED(aq::detection::Size3d, Eigen::Vector3f)
    REFLECT_END;

    REFLECT_DERIVED(aq::detection::Pose3d, Eigen::Affine3f)
    REFLECT_END;

} // namespace ct
