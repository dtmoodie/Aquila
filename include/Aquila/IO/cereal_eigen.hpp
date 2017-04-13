#pragma once
#include <opencv2/core/types.hpp>
#include "cereal/cereal.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cereal
{
    template<class AR, class T> inline
    void serialize(AR& ar, cv::Rect_<T>& rect)
    {
        ar(make_nvp("x", rect.x), make_nvp("y", rect.y), make_nvp("width", rect.width), make_nvp("height", rect.height));
    }

     // https://github.com/patrikhuber/eos/blob/master/include/eos/morphablemodel/io/eigen_cerealisation.hpp
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
    {
        std::int32_t rows = matrix.rows();
        std::int32_t cols = matrix.cols();
        ar(rows);
        ar(cols);
        ar(binary_data(matrix.data(), rows * cols * sizeof(_Scalar)));
    };

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline typename std::enable_if<!traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
    {
        (void)ar;
        (void)matrix;
    };

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
    {
        std::int32_t rows;
        std::int32_t cols;
        ar(rows);
        ar(cols);

        matrix.resize(rows, cols);

        ar(binary_data(matrix.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    };

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline typename std::enable_if<!traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
    {
        (void)ar;
        (void)matrix;
    };

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, const Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& m)
    {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows, cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<!traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, const Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& m)
    {
        (void)ar;
        (void)m;
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& m)
    {
        int32_t rows, cols;
        ar(rows, cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

    template<class Archive, class _Scalar, int _Dim, int _Mode, int _Options>
    typename std::enable_if<!traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& m)
    {
        (void)ar;
        (void)m;
    }
} // namespace cereal
