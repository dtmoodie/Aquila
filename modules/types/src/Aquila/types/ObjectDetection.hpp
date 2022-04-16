#pragma once
#include "Shape.hpp"
#include "detection/Category.hpp"
#include "detection/Classification.hpp"

#include <MetaObject/types/opencv.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <Aquila/detail/export.hpp>
#include <Aquila/types/EntityComponentSystem.hpp>

#include <ctext/datatable/SelectComponents.hpp>

#include <Eigen/Geometry>
#include <opencv2/core/types.hpp>

#include <map>
#include <string>
#include <vector>

namespace aq
{

    AQUILA_EXPORTS void boundingBoxToPixels(cv::Rect2f& bb, cv::Size size);
    AQUILA_EXPORTS void boundingBoxToPixels(cv::Rect2f&, aq::Shape<2> size);
    AQUILA_EXPORTS void normalizeBoundingBox(cv::Rect2f& bb, cv::Size size);
    AQUILA_EXPORTS void normalizeBoundingBox(cv::Rect2f& bb, aq::Shape<2> size);
    AQUILA_EXPORTS void clipNormalizedBoundingBox(cv::Rect2f& bb);

    namespace detection
    {

        // Various commonly used components

        struct Classifications_;
        using Classifications = aq::ArrayComponent<aq::Classification, Classifications_>;

        struct BoundingBox2d_;
        using BoundingBox2d = aq::ScalarComponent<cv::Rect2f, BoundingBox2d_>;

        struct Confidence_;
        using Confidence = aq::ScalarComponent<float, Confidence_>;

        struct Id_;
        using Id = aq::ScalarComponent<uint32_t, Id_>;

        struct Size3d_;
        using Size3d = aq::ScalarComponent<Eigen::Vector3f, Size3d_>;

        struct Pose3d_;
        using Pose3d = aq::ScalarComponent<Eigen::Affine3f, Pose3d_>;

    } // namespace detection

    //////////////////////////////////////////
    /// DetectedObject
    //////////////////////////////////////////
    struct AQUILA_EXPORTS DetectedObject
    {
        DetectedObject(const cv::Rect2f& rect = cv::Rect2f(),
                       const detection::Classifications& cls = detection::Classifications(),
                       unsigned int id = 0,
                       float confidence = 0.0);

        void classify(const detection::Classifications& cls);
        void classify(const Classification&, uint32_t k = 0);

        detection::BoundingBox2d bounding_box;
        detection::Classifications classifications;
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
    struct AQUILA_EXPORTS DetectedObjectSet : EntityComponentSystem
    {
        DetectedObjectSet(const DetectedObjectSet&) = default;
        DetectedObjectSet(DetectedObjectSet&&) = default;
        DetectedObjectSet& operator=(const DetectedObjectSet&) = default;
        DetectedObjectSet& operator=(DetectedObjectSet&&) = default;

        DetectedObjectSet(const CategorySet::ConstPtr& cats = CategorySet::ConstPtr());

        void setCatSet(const CategorySet::ConstPtr& cats);

        CategorySet::ConstPtr getCatSet() const;

      private:
        CategorySet::ConstPtr cat_set;
    };

    template <class DetType>
    struct AQUILA_EXPORTS TDetectedObjectSet : DetectedObjectSet
    {
        TDetectedObjectSet(const DetectedObjectSet& other)
            : DetectedObjectSet(other)
        {
            ensureContainsComponents(*this, static_cast<const DetType*>(nullptr));
        }

        TDetectedObjectSet(DetectedObjectSet&& other)
            : DetectedObjectSet(other)
        {
            ensureContainsComponents(*this, static_cast<const DetType*>(nullptr));
        }

        TDetectedObjectSet(const TDetectedObjectSet&) = default;
        TDetectedObjectSet(TDetectedObjectSet&&) = default;
        TDetectedObjectSet& operator=(const TDetectedObjectSet&) = default;
        TDetectedObjectSet& operator=(TDetectedObjectSet&&) = default;

        TDetectedObjectSet(const CategorySet::ConstPtr& cats = CategorySet::ConstPtr())
        {
            addComponents<DetType>(*this);
        }

        template <class... T>
        TDetectedObjectSet(const TDetectedObjectSet<T...>& other)
            : DetectedObjectSet(other)
        {
            ensureContainsComponents(*this, static_cast<const DetType*>(nullptr));
        }
    };
} // namespace aq

namespace mo
{
    template <>
    struct TSubscriber<aq::DetectedObjectSet> : public TSubscriberImpl<aq::DetectedObjectSet>
    {
        bool setInput(IPublisher* publisher) override
        {
            if (!acceptsPublisher(*publisher))
            {
                return false;
            }
            return TSubscriberImpl<aq::DetectedObjectSet>::setInput(publisher);
        }

        bool acceptsPublisher(const IPublisher& param) const override
        {
            if (TSubscriberImpl<aq::DetectedObjectSet>::acceptsPublisher(param))
            {
                return true;
            }
            return false;
        }
    };

    template <class... T>
    struct TSubscriber<aq::TDetectedObjectSet<T...>> : public TSubscriberImpl<aq::DetectedObjectSet>
    {
        bool setInput(IPublisher* publisher) override
        {
            if (!acceptsPublisher(*publisher))
            {
                return false;
            }
            return TSubscriberImpl<aq::DetectedObjectSet>::setInput(publisher);
        }

        bool acceptsPublisher(const IPublisher& param) const override
        {
            if (TSubscriberImpl<aq::DetectedObjectSet>::acceptsPublisher(param))
            {
                return true;
            }
            return false;
        }
    };

    template <class... T>
    struct TPublisher<aq::TDetectedObjectSet<T...>> : public TPublisher<aq::DetectedObjectSet>
    {
        bool providesOutput(const TypeInfo type) const
        {
            static const TypeInfo s_type = TypeInfo::create<aq::DetectedObjectSet>();
            return type == s_type;
        }
    };
} // namespace mo

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

    REFLECT_DERIVED(aq::DetectedObjectSet, aq::EntityComponentSystem)
        PROPERTY(cats, &DataType::getCatSet, &DataType::setCatSet)
    REFLECT_END;

    REFLECT_TEMPLATED_DERIVED(aq::TDetectedObjectSet, aq::DetectedObjectSet)

    REFLECT_END;
} // namespace ct
