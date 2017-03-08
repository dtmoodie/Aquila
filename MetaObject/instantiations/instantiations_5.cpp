#ifdef HAVE_OPENCV
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/cvSpecializations.hpp"
#include "cereal/types/vector.hpp"
#include <boost/lexical_cast.hpp>

namespace mo
{
namespace IO
{
namespace Text
{
    namespace imp
    {
        template<typename T>
        void Serialize_imp(std::ostream &os, const cv::Rect_<T>& obj, int)
        {
            os << obj.x << ", " << obj.y << ", " <<  obj.width << ", " << obj.height;
        }
        template<typename T>
        void DeSerialize_imp(std::istream &is, cv::Rect_<T>& obj, int)
        {
            char c;
            for(int i = 0; i < 4; ++i)
            {
                is >> obj[i];
                is >> c;
            }
        }
    }
}
}
}

INSTANTIATE_META_PARAMETER(cv::Rect);
INSTANTIATE_META_PARAMETER(cv::Rect2d);
INSTANTIATE_META_PARAMETER(cv::Rect2f);
INSTANTIATE_META_PARAMETER(std::vector<cv::Rect>);
INSTANTIATE_META_PARAMETER(std::vector<cv::Rect2f>);

#endif
