#ifndef AQ_TYPES_DATA_TYPE_HPP
#define AQ_TYPES_DATA_TYPE_HPP
#include <MetaObject/logging/logging.hpp>
#include <ct/enum.hpp>

#include <cstdint>

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#endif

namespace aq
{
    ENUM_BEGIN(DataFlag, uint8_t)
        ENUM_VALUE(kDEFAULT, 0)
        ENUM_VALUE(kUINT8, kDEFAULT + 1)
        ENUM_VALUE(kINT8, kUINT8 + 1)
        ENUM_VALUE(kUINT16, kINT8 + 1)
        ENUM_VALUE(kINT16, kUINT16 + 1)
        ENUM_VALUE(kFLOAT16, kINT16 + 1)
        ENUM_VALUE(kHALF, kFLOAT16)
        ENUM_VALUE(kUINT32, kHALF + 1)
        ENUM_VALUE(kINT32, kUINT32 + 1)
        ENUM_VALUE(kFLOAT32, kINT32 + 1)
        ENUM_VALUE(kFLOAT, kFLOAT32)
        ENUM_VALUE(kUINT64, kFLOAT + 1)
        ENUM_VALUE(kINT64, kUINT64 + 1)
        ENUM_VALUE(kFLOAT64, kINT64 + 1)
        ENUM_VALUE(kDOUBLE, kFLOAT64)

        // Number of bits required to store DataFlag
        ENUM_VALUE(kREQUIRED_FLAG_BITS, 4)

        constexpr size_t elemSize() const;
    ENUM_END;

    template <class T>
    struct DataType;

    template <>
    struct DataType<uint8_t>
    {
        static constexpr const auto depth_flag = DataFlag::kUINT8;
    };

    template <>
    struct DataType<int8_t>
    {
        static constexpr const auto depth_flag = DataFlag::kINT8;
    };

    template <>
    struct DataType<uint16_t>
    {
        static constexpr const auto depth_flag = DataFlag::kUINT16;
    };

    template <>
    struct DataType<int16_t>
    {
        static constexpr const auto depth_flag = DataFlag::kINT16;
    };

    template <>
    struct DataType<int32_t>
    {
        static constexpr const auto depth_flag = DataFlag::kINT32;
    };

    template <>
    struct DataType<uint32_t>
    {
        static constexpr const auto depth_flag = DataFlag::kUINT32;
    };

    template <>
    struct DataType<float>
    {
        static constexpr const auto depth_flag = DataFlag::kFLOAT32;
    };

    template <>
    struct DataType<uint64_t>
    {
        static constexpr const auto depth_flag = DataFlag::kUINT64;
    };

    template <>
    struct DataType<int64_t>
    {
        static constexpr const auto depth_flag = DataFlag::kINT64;
    };

    template <>
    struct DataType<double>
    {
        static constexpr const auto depth_flag = DataFlag::kFLOAT64;
    };

    using AvailableDataTypes_t =
        ct::VariadicTypedef<uint8_t, int8_t, uint16_t, int16_t, int32_t, uint32_t, float, uint64_t, int64_t, double>;

    template <class T>
    constexpr size_t elemSizeHelper(DataFlag val, ct::VariadicTypedef<T>)
    {
        return val == DataType<T>::depth_flag ? sizeof(T) : 0;
    }

    template <class T, class... ARGS>
    constexpr size_t elemSizeHelper(DataFlag val, ct::VariadicTypedef<T, ARGS...>)
    {
        return val == DataType<T>::depth_flag ? sizeof(T) : elemSizeHelper(val, ct::VariadicTypedef<ARGS...>{});
    }

    constexpr size_t DataFlag::elemSize() const
    {
        return elemSizeHelper(*this, AvailableDataTypes_t{});
    }

#ifdef HAVE_OPENCV
    inline int toCvDepth(DataFlag depth)
    {
        switch (depth)
        {
        case DataFlag::kUINT8:
            return CV_8U;
        case DataFlag::kINT8:
            return CV_8S;
        case DataFlag::kUINT16:
            return CV_16U;
        case DataFlag::kINT16:
            return CV_16S;
        case DataFlag::kINT32:
            return CV_32S;
        case DataFlag::kFLOAT:
            return CV_32F;
        case DataFlag::kDOUBLE:
            return CV_64F;
        default:
            THROW(warn, "Invalid depth {}, not mapped to opencv", depth);
        }
        return 0;
    }

    inline DataFlag fromCvDepth(int depth)
    {
        switch (depth)
        {
        case CV_8U:
            return DataFlag::kUINT8;
        case CV_8S:
            return DataFlag::kINT8;
        case CV_16U:
            return DataFlag::kUINT16;
        case CV_16S:
            return DataFlag::kINT16;
        case CV_32S:
            return DataFlag::kINT32;
        case CV_32F:
            return DataFlag::kFLOAT32;
        case CV_64F:
            return DataFlag::kFLOAT64;
        default:
            THROW(warn, "Invalid cv depth {} passed in", depth);
        }
        return {};
    }
#endif
} // namespace aq

#endif // AQ_TYPES_DATA_TYPE_HPP
