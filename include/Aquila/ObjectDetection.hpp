#pragma once
#include <string>
#include <opencv2/core/types.hpp>
#include <vector>
#include <Aquila/Detail/Export.hpp>
#include <cereal/cereal.hpp>
#include <Eigen/Geometry>
#include "MetaObject/Parameters/IParameter.hpp"
#include <cereal/types/boost/optional.hpp>

namespace cereal
{
    template<class AR, class T> inline
    void serialize(AR& ar, cv::Rect_<T>& rect)
    {
        ar(make_nvp("x", rect.x), make_nvp("y", rect.y), make_nvp("width", rect.width), make_nvp("height", rect.height));
    }

     // http://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m)
    {
      int32_t rows = m.rows();
      int32_t cols = m.cols();
      ar(rows);
      ar(cols);
      ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m)
    {
      int32_t rows;
      int32_t cols;
      ar(rows);
      ar(cols);

      m.resize(rows, cols);

      ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options> const& m)
    {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows, cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& m)
    {
        int32_t rows, cols;
        ar(rows, cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }
} // namespace cereal

namespace aq
{
    struct AQUILA_EXPORTS Classification
    {
        Classification(const std::string& label_ = "", float confidence_ = 0, int classNumber_ = -1);
        std::string label;
        float confidence;
        int classNumber;

        template<class AR>
        void serialize(AR& ar)
        {
            ar(CEREAL_NVP(label), CEREAL_NVP(confidence), CEREAL_NVP(classNumber));
        }
    };
    struct AQUILA_EXPORTS DetectedObject2d
    {
        enum {Dims = 2};
        std::vector<Classification> detections;
        cv::Rect2f boundingBox;
        boost::optional<mo::time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR>
        void serialize(AR& ar)
        {
            ar(CEREAL_NVP(boundingBox), CEREAL_NVP(detections), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
        }
    };

    typedef DetectedObject2d DetectedObject;

    struct AQUILA_EXPORTS DetectedObject3d
    {
        enum {Dims = 3};
        std::vector<Classification> detections;
        /*!
         * \brief pose determines the pose to the center of the object
         */
        Eigen::Affine3d pose;
        /*!
         * \brief size is the centered size of the object
         */
        Eigen::Vector3d size;
        boost::optional<mo::time_t> timestamp;
        size_t framenumber = 0;
        int id = 0;
        template<class AR>
        void serialize(AR& ar)
        {
            ar(CEREAL_NVP(pose), CEREAL_NVP(size), CEREAL_NVP(detections), CEREAL_NVP(timestamp), CEREAL_NVP(id), CEREAL_NVP(framenumber));
        }
    };

    void AQUILA_EXPORTS CreateColormap(cv::Mat& lut, int num_classes, int ignore_class = -1);
}
