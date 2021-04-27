#ifndef AQ_TYPES_SYNCED_IMAGE_HPP
#define AQ_TYPES_SYNCED_IMAGE_HPP

#if defined(HAVE_OPENCV) || defined(OPENCV_CORE_CUDA_HPP) || defined(OPENCV_CORE_TYPES_HPP)
#include <ct/types/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#if !defined(HAVE_OPENCV)
#define HAVE_OPENCV
#endif
#endif

#include "CompressedImage.hpp"

#include "Shape.hpp"
#include "TSyncedMemory.hpp"
#include "pixels.hpp"

#include "DataType.hpp"

namespace aq
{
    struct SyncedImage;

    struct CompressedImage;

    struct PixelType
    {
        DataFlag data_type;
        PixelFormat pixel_format;

        inline size_t pixelSize() const
        {
            return data_type.elemSize() * pixel_format.numChannels();
        }
#ifdef HAVE_OPENCV
        inline int32_t toCvType() const
        {
            const int32_t depth = toCvDepth(data_type);
            return CV_MAKE_TYPE(depth, pixel_format.numChannels());
        }
#endif
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

        // default ctr
        SyncedImage(Shape<2> size = Shape<2>(0, 0),
                    PixelFormat fmt = PixelFormat::kRGB,
                    DataFlag type = DataFlag::kUINT8,
                    std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());

        template <class PIXEL>
        SyncedImage(const Shape<2>& shape,
                    PIXEL* data,
                    std::shared_ptr<const void> owning = std::shared_ptr<const void>{},
                    std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());

        SyncedImage(const Shape<2>& shape,
                    PixelType type,
                    void* data,
                    std::shared_ptr<const void> owning = std::shared_ptr<const void>{},
                    std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());

        SyncedImage(const Shape<2>& shape,
                    PixelType type,
                    const void* data,
                    std::shared_ptr<const void> owning = std::shared_ptr<const void>{},
                    std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());

        SyncedImage(Shape<2> size, PixelType type, std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());

        // Copy on write
        SyncedImage(const SyncedImage&, std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());

        SyncedImage(SyncedImage&, std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());
        SyncedImage(SyncedImage&&, std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());
        SyncedImage& operator=(const SyncedImage&);
        SyncedImage& operator=(SyncedImage&);
        SyncedImage& operator=(SyncedImage&&);

        void create(Shape<2> size, PixelFormat fmt = PixelFormat::kUNCHANGED, DataFlag type = DataFlag::kUINT8);
        void create(Shape<2> size, PixelType type = {DataFlag::kUINT8, PixelFormat::kUNCHANGED});
        void create(uint32_t height,
                    uint32_t width,
                    PixelFormat fmt = PixelFormat::kUNCHANGED,
                    DataFlag type = DataFlag::kUINT8);

        std::weak_ptr<mo::IAsyncStream> getStream() const;
        void setStream(std::shared_ptr<mo::IAsyncStream>);

        PixelType pixelType() const;
        PixelFormat pixelFormat() const;
        size_t pixelSize() const;
        size_t elemSize() const;
        DataFlag dataType() const;
        uint8_t channels() const;
        Shape<3> shape() const;
        Shape<2> size() const;
        void reshape(Shape<2>);
        uint32_t rows() const;
        uint32_t cols() const;
        size_t sizeBytes() const;

        template <class PIXEL>
        Matrix<PIXEL> mutableHost(mo::IAsyncStream* = nullptr, bool* sync_required = nullptr);

        template <class PIXEL>
        Matrix<PIXEL> mutableDevice(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr);

        template <class PIXEL>
        ConstMatrix<PIXEL> host(mo::IAsyncStream* stream = nullptr, bool* sync_required = nullptr) const;

        template <class PIXEL>
        ConstMatrix<PIXEL> device(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const;

        ce::shared_ptr<SyncedMemory> data();
        ce::shared_ptr<const SyncedMemory> data() const;
        void setData(ce::shared_ptr<SyncedMemory>);
        bool empty() const;

        SyncedMemory::SyncState state() const;

        void setOwning(std::shared_ptr<const void>);

#ifdef HAVE_OPENCV
        inline SyncedImage(const cv::Mat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());
        inline SyncedImage(const cv::cuda::GpuMat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());
        inline SyncedImage(cv::Mat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IAsyncStream> = mo::IAsyncStream::current());
        inline SyncedImage(cv::cuda::GpuMat& mat,
                           PixelFormat = PixelFormat::kBGR,
                           std::shared_ptr<mo::IDeviceStream> = mo::IDeviceStream::current());

        inline cv::Mat mutableMat(mo::IAsyncStream* stream = nullptr, bool* sync = nullptr);
        inline cv::cuda::GpuMat mutableGpuMat(mo::IDeviceStream* stream = nullptr, bool* sync = nullptr);
        inline const cv::Mat mat(mo::IAsyncStream* stream = nullptr, bool* sync = nullptr) const;
        inline const cv::cuda::GpuMat gpuMat(mo::IDeviceStream* stream = nullptr, bool* sync = nullptr) const;

        inline cv::Mat getMutableMat(mo::IAsyncStream* stream = nullptr, bool* sync = nullptr);
        inline cv::cuda::GpuMat getMutableGpuMat(mo::IDeviceStream* stream = nullptr, bool* sync = nullptr);
        inline const cv::Mat getMat(mo::IAsyncStream* stream = nullptr, bool* sync = nullptr) const;
        inline const cv::cuda::GpuMat getGpuMat(mo::IDeviceStream* stream = nullptr, bool* sync = nullptr) const;

        inline operator cv::Mat();
        inline operator cv::cuda::GpuMat();
        inline operator const cv::Mat() const;
        inline operator const cv::cuda::GpuMat() const;

        inline void copyTo(cv::cuda::GpuMat& mat, mo::IDeviceStream* stream = nullptr) const;
        inline void copyTo(cv::Mat& mat, mo::IAsyncStream* stream = nullptr) const;
#endif

      private:
        void makeData();
        ce::shared_ptr<SyncedMemory> m_data;
        PixelType m_pixel_type;
        Shape<2> m_shape;
    };

    template <class PIXEL>
    SyncedImage::SyncedImage(const Shape<2>& shape,
                             PIXEL* data,
                             std::shared_ptr<const void> owning,
                             std::shared_ptr<mo::IAsyncStream> stream)
    {
        ct::TArrayView<PIXEL> view(data, shape.numel());
        if (owning)
        {
            m_data = ce::make_shared<SyncedMemory>(SyncedMemory::wrapHost(view, owning, stream));
        }
        else
        {
            m_data = ce::make_shared<SyncedMemory>(SyncedMemory::copyHost(ct::TArrayView<const PIXEL>(view), stream));
        }
        m_shape = std::move(shape);
        m_pixel_type.pixel_format = PIXEL::pixel_format;
        m_pixel_type.data_type = DataType<typename PIXEL::Scalar_t>::depth_flag;
    }

///////////////////////////////////////////////////////////////////////////////////////////
///    IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////////////////

#ifdef HAVE_OPENCV
    SyncedImage::SyncedImage(const cv::Mat& mat, PixelFormat fmt, std::shared_ptr<mo::IAsyncStream> stream)
    {
        ct::TArrayView<const void> data_view(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize());

        SyncedMemory wrapped =
            SyncedMemory::wrapHost(data_view, mat.elemSize(), std::make_shared<cv::Mat>(mat), stream);

        m_data = std::make_shared<SyncedMemory>(std::move(wrapped));

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
    {
        ct::TArrayView<const void> data_view(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize());
        SyncedMemory wrapped =
            SyncedMemory::wrapDevice(data_view, mat.elemSize(), std::make_shared<cv::cuda::GpuMat>(mat), stream);

        m_data = std::make_shared<SyncedMemory>(std::move(wrapped));

        if (mat.channels() == 1 && fmt == PixelFormat::kBGR)
        {
            fmt = PixelFormat::kGRAY;
        }
        m_pixel_type.data_type = fromCvDepth(mat.depth());
        m_pixel_type.pixel_format = fmt;
        m_shape(0) = static_cast<uint32_t>(mat.rows);
        m_shape(1) = static_cast<uint32_t>(mat.cols);
    }

    SyncedImage::SyncedImage(cv::Mat& mat, PixelFormat fmt, std::shared_ptr<mo::IAsyncStream> stream)
    {
        ct::TArrayView<void> data_view(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize());
        SyncedMemory wrapped =
            SyncedMemory::wrapHost(data_view, mat.elemSize(), std::make_shared<cv::Mat>(mat), stream);

        m_data = std::make_shared<SyncedMemory>(std::move(wrapped));

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
    {
        ct::TArrayView<void> data_view(mat.data, static_cast<size_t>(mat.rows * mat.cols) * mat.elemSize());
        SyncedMemory wrapped =
            SyncedMemory::wrapDevice(data_view, mat.elemSize(), std::make_shared<cv::cuda::GpuMat>(mat), stream);
        m_data = std::make_shared<SyncedMemory>(std::move(wrapped));

        if (mat.channels() == 1 && fmt == PixelFormat::kBGR)
        {
            fmt = PixelFormat::kGRAY;
        }
        m_pixel_type.data_type = fromCvDepth(mat.depth());
        m_pixel_type.pixel_format = fmt;
        m_shape(0) = static_cast<uint32_t>(mat.rows);
        m_shape(1) = static_cast<uint32_t>(mat.cols);
    }

    cv::Mat SyncedImage::mutableMat(mo::IAsyncStream* stream, bool* sync)
    {
        const auto height = static_cast<int>(rows());
        const auto width = static_cast<int>(cols());
        auto dtype = dataType();
        auto c = channels();
        auto cvdepth = toCvDepth(dtype);
        const auto type = CV_MAKETYPE(cvdepth, c);
        auto data = m_data->mutableHost(stream, sync).data();
        return cv::Mat(height, width, type, data);
    }

    cv::cuda::GpuMat SyncedImage::mutableGpuMat(mo::IDeviceStream* stream, bool* sync)
    {
        const auto height = static_cast<int>(rows());
        const auto width = static_cast<int>(cols());
        auto dtype = dataType();
        auto c = channels();
        auto cvdepth = toCvDepth(dtype);
        const auto type = CV_MAKETYPE(cvdepth, c);
        auto data = m_data->mutableDevice(stream, sync).data();
        return cv::cuda::GpuMat(height, width, type, data);
    }

    const cv::Mat SyncedImage::mat(mo::IAsyncStream* stream, bool* sync) const
    {
        const auto height = static_cast<int>(rows());
        const auto width = static_cast<int>(cols());
        auto dtype = dataType();
        auto c = channels();
        auto cvdepth = toCvDepth(dtype);
        const auto type = CV_MAKETYPE(cvdepth, c);
        auto data = const_cast<void*>(m_data->host(stream, sync).data());
        return cv::Mat(height, width, type, data);
    }

    const cv::cuda::GpuMat SyncedImage::gpuMat(mo::IDeviceStream* stream, bool* sync) const
    {
        const auto height = static_cast<int>(rows());
        const auto width = static_cast<int>(cols());
        auto dtype = dataType();
        auto c = channels();
        auto cvdepth = toCvDepth(dtype);
        const auto type = CV_MAKETYPE(cvdepth, c);
        auto data = const_cast<void*>(m_data->device(stream, sync).data());
        return cv::cuda::GpuMat(height, width, type, data);
    }

    cv::Mat SyncedImage::getMutableMat(mo::IAsyncStream* stream, bool* sync)
    {
        return mutableMat(stream, sync);
    }

    cv::cuda::GpuMat SyncedImage::getMutableGpuMat(mo::IDeviceStream* stream, bool* sync)
    {
        return mutableGpuMat(stream, sync);
    }
    const cv::Mat SyncedImage::getMat(mo::IAsyncStream* stream, bool* sync) const
    {
        return mat(stream, sync);
    }

    const cv::cuda::GpuMat SyncedImage::getGpuMat(mo::IDeviceStream* stream, bool* sync) const
    {
        return gpuMat(stream, sync);
    }

    SyncedImage::operator cv::Mat()
    {
        return mutableMat();
    }

    SyncedImage::operator cv::cuda::GpuMat()
    {
        return mutableGpuMat();
    }

    SyncedImage::operator const cv::Mat() const
    {
        return mat();
    }

    SyncedImage::operator const cv::cuda::GpuMat() const
    {
        return gpuMat();
    }

    void SyncedImage::copyTo(cv::cuda::GpuMat& mat, mo::IDeviceStream* stream) const
    {
        const auto height = static_cast<int>(rows());
        const auto width = static_cast<int>(cols());
        const auto type = CV_MAKETYPE(toCvDepth(dataType()), channels());
        auto src_view = m_data->device(stream);
        mat.create(height, width, type);
        ct::TArrayView<void> dst_view(mat.datastart, mat.size().area() * mat.elemSize());
        MO_ASSERT(dst_view.size() == src_view.size());
        if (stream)
        {
            stream->deviceToDevice(dst_view, src_view);
        }
    }

    void SyncedImage::copyTo(cv::Mat& mat, mo::IAsyncStream* dst_stream) const
    {
        bool sync = false;
        cv::Mat tmp = this->mat(dst_stream, &sync);
        if (sync)
        {
            if (dst_stream)
            {
                dst_stream->synchronize();
            }
            else
            {
                auto src_stream = getStream().lock();
                src_stream->synchronize();
            }
        }

        tmp.copyTo(mat);
    }
#endif // HAVE_OPENCV

    template <class PIXEL>
    SyncedImage::Matrix<PIXEL> SyncedImage::mutableHost(mo::IAsyncStream* stream, bool* sync_required)
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
    SyncedImage::ConstMatrix<PIXEL> SyncedImage::host(mo::IAsyncStream* stream, bool* sync_required) const
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
