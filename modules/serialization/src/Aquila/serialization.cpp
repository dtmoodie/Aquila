#include <ct/types/opencv.hpp>

#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/params/detail/MetaParamImpl.hpp>

#include <cereal/types/string.hpp>

namespace aq
{

    namespace serialization
    {
        template <class T>
        using vector = std::vector<T>;

        void initModule(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(SyncedMemory, table);
            INSTANTIATE_META_PARAM(vector<SyncedMemory>, table);
            INSTANTIATE_META_PARAM(cv::Mat, table);
            INSTANTIATE_META_PARAM(Category, table);
            INSTANTIATE_META_PARAM(Classification, table);
            INSTANTIATE_META_PARAM(DetectedObject, table);
            INSTANTIATE_META_PARAM(DetectedObjectSet, table);
            INSTANTIATE_META_PARAM(CategorySet, table);
        }
    } // namespace serialization
} // namespace aq
