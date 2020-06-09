#ifndef AQ_TYPES_SYNCED_IMAGE_HPP
#define AQ_TYPES_SYNCED_IMAGE_HPP
#include "CompressedImage.hpp"
#include "DataType.hpp"
#include "Shape.hpp"
#include "TSyncedMemory.hpp"
#include "pixels.hpp"

#if defined(HAVE_OPENCV) || defined(OPENCV_CORE_CUDA_HPP)
#include <opencv2/core/cuda.hpp>
#if !defined(HAVE_OPENCV)
#define HAVE_OPENCV
#endif
#endif

namespace aq
{
    struct SyncedImage;

    struct CompressedImage;

    struct PixelType
    {
        DataFlag data_type;
        PixelFormat pixel_format;
    };

    // SyncedImage inherits from enable_shared_from_this because the opencv wrapping api needs a shared_ptr
    // to enable proper ref counting
    struct AQUILA_EXPORTS SyncedImage : ce::HashedBase, std::enable_shared_from_this<SyncedImage>
    {
        using Ptr_t = std::shared_ptr<SyncedImage>;
        using ConstPtr_t = std::shared_ptr<const SyncedImage>;

        static constexpr const uint8_t MAX_CHANNELS = 255;

        template <class PIXEL>
        using Matrix = Eigen::Map<Eigen::Matrix<PIXEL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        template <class PIXEL>
        using ConstMatrix = Eigen::Map<const Eigen::Matrix<PIXEL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        SyncedImage(Shape<2> size = Shape<2>(0, 0),
                    PixelFormat fmt = PixelFormat::kRGB,
                    DataFlag type = DataFlag::kUINT8,
                    std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        // Copy on write
        SyncedImage(const SyncedImage&, std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());

        SyncedImage(SyncedImage&, std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        SyncedImage(SyncedImage&&, std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        SyncedImage& operator=(const SyncedImage&);
        SyncedImage& operator=(SyncedImage&);
        SyncedImage& operator=(SyncedImage&&);

        void create(Shape<2> size, PixelFormat fmt = PixelFormat::kUNCHANGED, DataFlag type = DataFlag::kUINT8);
        void create(uint32_t height,
                    uint32_t width,
                    PixelFormat fmt = PixelFormat::kUNCHANGED,
                    DataFlag type = DataFlag::kUINT8);

        std::weak_ptr<mo::IDeviceStream> stream() const;
        void setStream(std::shared_ptr<mo::IDeviceStream>);

        PixelType pixelType() const;
        PixelFormat pixelFormat() const;
        size_t pixelSize() const;
        size_t elemSize() const;
        DataFlag dataType() const;
        uint8_t channels() const;
        Shape<3> shape() const;
        void reshape(Shape<2>);
        uint32_t rows() const;
        uint32_t cols() const;
        size_t sizeBytes() const;

        template <class PIXEL>
        Matrix<PIXEL> mutableHost(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr);

        template <class PIXEL>
        Matrix<PIXEL> mutableDevice(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr);

        template <class PIXEL>
        ConstMatrix<PIXEL> host(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const;

        template <class PIXEL>
        ConstMatrix<PIXEL> device(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const;

        ce::shared_ptr<SyncedMemory> data();
        ce::shared_ptr<const SyncedMemory> data() const;
        void setData(ce::shared_ptr<SyncedMemory>);
        bool empty() const;

#ifdef HAVE_OPENCV
        inline SyncedImage(const cv::Mat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        inline SyncedImage(const cv::cuda::GpuMat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        inline SyncedImage(cv::Mat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        inline SyncedImage(cv::cuda::GpuMat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        inline cv::Mat mutableMat();
        inline operator cv::Mat();
        inline cv::cuda::GpuMat mutableGpuMat();
        inline operator cv::cuda::GpuMat();

        inline const cv::Mat mat() const;
        inline operator const cv::Mat() const;

        inline const cv::cuda::GpuMat gpuMat() const;
        inline operator const cv::cuda::GpuMat() const;
#endif

      private:
        void makeData();
        ce::shared_ptr<SyncedMemory> m_data;
        PixelType m_pixel_type;
        Shape<2> m_shape;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///    IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////////////////////////////

#ifdef HAVE_OPENCV
    SyncedImage::SyncedImage(const cv::Mat& mat, PixelFormat fmt, std::shared_ptr<mo::IDeviceStream> stream)
        : m_data(std::make_shared<SyncedMemory>(SyncedMemory::wrapHost(
              ct::TArrayView<const void>(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize()),
              mat.elemSize(),
              std::make_shared<cv::Mat>(mat),
              stream)))
    {
        if (mat.channels() == 1 && fmt == PixelFormat::kBGR)
        {
            fmt = PixelFormat::kGRAY;
        }
        m_pixel_type.data_type = fromCvDepth(mat.depth());
        m_pixel_type.pixel_format = fmt;
        m_shape(0) = static_cast<uint32_t>(mat.rows);
        m_shape(1) = static_cast<uint32_t>(mat.cols);
    }

    SyncedImage::SyncedImage(const cv::cuda::GpuMat& mat, PixelFormat fmt, std::shared_ptr<mo::IDeviceStream> stream)
        : m_data(std::make_shared<SyncedMemory>(SyncedMemory::wrapDevice(
              ct::TArrayView<const void>(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize()),
              mat.elemSize(),
              std::make_shared<cv::cuda::GpuMat>(mat),
              stream)))
    {
        if (mat.channels() == 1 && fmt == PixelFormat::kBGR)
        {
            fmt = PixelFormat::kGRAY;
        }
        m_pixel_type.data_type = fromCvDepth(mat.depth());
        m_pixel_type.pixel_format = fmt;
        m_shape(0) = static_cast<uint32_t>(mat.rows);
        m_shape(1) = static_cast<uint32_t>(mat.cols);
    }

    SyncedImage::SyncedImage(cv::Mat& mat, PixelFormat fmt, std::shared_ptr<mo::IDeviceStream> stream)
        : m_data(std::make_shared<SyncedMemory>(SyncedMemory::wrapHost(
              ct::TArrayView<void>(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize()),
              mat.elemSize(),
              std::make_shared<cv::Mat>(mat),
              stream)))
    {
        if (mat.channels() == 1 && fmt == PixelFormat::kBGR)
        {
            fmt = PixelFormat::kGRAY;
        }
        m_pixel_type.data_type = fromCvDepth(mat.depth());
        m_pixel_type.pixel_format = fmt;
        m_shape(0) = static_cast<uint32_t>(mat.rows);
        m_shape(1) = static_cast<uint32_t>(mat.cols);
    }

    SyncedImage::SyncedImage(cv::cuda::GpuMat& mat, PixelFormat fmt, std::shared_ptr<mo::IDeviceStream> stream)
        : m_data(std::make_shared<SyncedMemory>(SyncedMemory::wrapDevice(
              ct::TArrayView<void>(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize()),
              mat.elemSize(),
              std::make_shared<cv::cuda::GpuMat>(mat),
              stream)))
    {
        if (mat.channels() == 1 && fmt == PixelFormat::kBGR)
        {
            fmt = PixelFormat::kGRAY;
        }
        m_pixel_type.data_type = fromCvDepth(mat.depth());
        m_pixel_type.pixel_format = fmt;
        m_shape(0) = static_cast<uint32_t>(mat.rows);
        m_shape(1) = static_cast<uint32_t>(mat.cols);
    }

    cv::Mat SyncedImage::mutableMat()
    {
        return cv::Mat(static_cast<int>(rows()),
                       static_cast<int>(cols()),
                       CV_MAKETYPE(toCvDepth(dataType()), channels()),
                       m_data->mutableHost().data());
    }

    SyncedImage::operator cv::Mat()
    {
        return mutableMat();
    }

    cv::cuda::GpuMat SyncedImage::mutableGpuMat()
    {
        return cv::cuda::GpuMat(static_cast<int>(rows()),
                                static_cast<int>(cols()),
                                CV_MAKETYPE(toCvDepth(dataType()), channels()),
                                m_data->mutableDevice().data());
    }

    SyncedImage::operator cv::cuda::GpuMat()
    {
        return mutableGpuMat();
    }

    const cv::Mat SyncedImage::mat() const
    {
        return cv::Mat(static_cast<int>(rows()),
                       static_cast<int>(cols()),
                       CV_MAKETYPE(toCvDepth(dataType()), channels()),
                       const_cast<void*>(m_data->host().data()));
    }
    SyncedImage::operator const cv::Mat() const
    {
        return mat();
    }

    const cv::cuda::GpuMat SyncedImage::gpuMat() const
    {
        return cv::cuda::GpuMat(static_cast<int>(rows()),
                                static_cast<int>(cols()),
                                CV_MAKETYPE(toCvDepth(dataType()), channels()),
                                const_cast<void*>(m_data->device().data()));
    }

    SyncedImage::operator const cv::cuda::GpuMat() const
    {
        return gpuMat();
    }
#endif // HAVE_OPENCV

    template <class PIXEL>
    SyncedImage::Matrix<PIXEL> SyncedImage::mutableHost(mo::IDeviceStream* stream, bool* sync_required)
    {
        MO_ASSERT_EQ(m_pixel_type.pixel_format, ct::value(PIXEL::pixel_format));
        MO_ASSERT_EQ(m_pixel_type.data_type, ct::value(DataType<typename PIXEL::Scalar_t>::depth_flag));
        auto host_memory = m_data->mutableHostAs<PIXEL>(stream, sync_required);
        return Matrix<PIXEL>(host_memory.data(), rows(), cols());
    }

    template <class PIXEL>
    SyncedImage::Matrix<PIXEL> SyncedImage::mutableDevice(mo::IDeviceStream* stream, bool* sync_required)
    {
        MO_ASSERT_EQ(ct::value(PIXEL::pixel_format), m_pixel_type.pixel_format);
        MO_ASSERT_EQ(ct::value(DataType<typename PIXEL::Scalar_t>::flag), m_pixel_type.data_type);
        auto device_memory = m_data->mutableDeviceAs<PIXEL>(stream, sync_required);
        return Matrix<PIXEL>(device_memory.data(), rows(), cols());
    }

    template <class PIXEL>
    SyncedImage::ConstMatrix<PIXEL> SyncedImage::host(mo::IDeviceStream* stream, bool* sync_required) const
    {
        MO_ASSERT_EQ(m_pixel_type.pixel_format, ct::value(PIXEL::pixel_format));
        MO_ASSERT_EQ(m_pixel_type.data_type, ct::value(DataType<typename PIXEL::Scalar_t>::depth_flag));
        auto host_memory = m_data->hostAs<PIXEL>(stream, sync_required);
        return ConstMatrix<PIXEL>(host_memory.data(), rows(), cols());
    }

    template <class PIXEL>
    SyncedImage::ConstMatrix<PIXEL> SyncedImage::device(mo::IDeviceStream* stream, bool* sync_required) const
    {
        MO_ASSERT_EQ(ct::value(PIXEL::pixel_format), m_pixel_type.pixel_format);
        MO_ASSERT_EQ(ct::value(DataType<typename PIXEL::Scalar_t>::flag), m_pixel_type.data_type);
        auto device_memory = m_data->deviceAs<PIXEL>(stream, sync_required);
        return ConstMatrix<PIXEL>(device_memory.data(), rows(), cols());
    }
} // namespace aq

namespace ct
{
    AQUILA_EXPORTS aq::Shape<2> getImageShape(const aq::SyncedImage&);
    REFLECT_BEGIN(aq::SyncedImage)
        PROPERTY(
            data,
            static_cast<ce::shared_ptr<const aq::SyncedMemory> (aq::SyncedImage::*)() const>(&aq::SyncedImage::data),
            &aq::SyncedImage::setData)
        PROPERTY(shape, &getImageShape, &aq::SyncedImage::reshape)
        PROPERTY(size, &aq::SyncedImage::sizeBytes)
    REFLECT_END;

    TArrayView<void> makeArrayView(AccessToken<void (aq::SyncedImage::*)(ce::shared_ptr<aq::SyncedMemory>)>&&, size_t);
} // namespace ct

#endif // AQ_TYPES_SYNCED_IMAGE_HPP
