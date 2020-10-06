#ifndef AQUILA_TSYNCED_IMAGE_HPP
#define AQUILA_TSYNCED_IMAGE_HPP

#include "SyncedImage.hpp"

namespace aq
{
    template <class T>
    struct TSyncedImage : public SyncedImage
    {
        using Scalar_t = typename T::Scalar_t;
        using Matrix_t = SyncedImage::Matrix<T>;
        using ConstMatrix_t = SyncedImage::ConstMatrix<T>;

        TSyncedImage(Shape<2> size_ = {0, 0}, std::shared_ptr<mo::IDeviceStream> stream = mo::IDeviceStream::current())
            : SyncedImage(size_, T::pixel_format, DataType<Scalar_t>::depth_flag, stream)
        {
        }

        TSyncedImage(SyncedImage& other)
            : SyncedImage(other)
        {
            MO_ASSERT_EQ(dataType(), ct::value(DataType<Scalar_t>::depth_flag));
            MO_ASSERT_EQ(pixelFormat(), ct::value(T::pixel_format));
        }

        TSyncedImage(const SyncedImage& other)
            : SyncedImage(other)
        {
            MO_ASSERT_EQ(dataType(), ct::value(DataType<Scalar_t>::depth_flag));
            MO_ASSERT_EQ(pixelFormat(), ct::value(T::pixel_format));
        }

        TSyncedImage(SyncedImage&& other)
            : SyncedImage(std::move(other))
        {
            MO_ASSERT_EQ(dataType(), ct::value(DataType<Scalar_t>::depth_flag));
            MO_ASSERT_EQ(pixelFormat(), ct::value(T::pixel_format));
        }

        void create(Shape<2> size_)
        {
            SyncedImage::create(size_, T::num_channels, DataType<Scalar_t>::depth_flag, T::pixel_format);
        }

        void create(uint32_t height, uint32_t width)
        {
            create({height, width});
        }

        Matrix_t mutableHost(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr)
        {
            return SyncedImage::template mutableHost<T>(stream, sync_required);
        }

        ConstMatrix_t host(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const
        {
            return SyncedImage::template host<T>(stream, sync_required);
        }

        Matrix_t mutableDevice(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr)
        {
            return SyncedImage::template mutableDevice<T>(stream, sync_required);
        }

        ConstMatrix_t device(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const
        {
            return SyncedImage::template device<T>(stream, sync_required);
        }
    };

    template <class T>
    struct TSyncedImage<const T> : public SyncedImage
    {
        using Scalar_t = const typename T::Scalar_t;
        using ConstMatrix_t = SyncedImage::ConstMatrix<T>;

        TSyncedImage(Shape<2> size_ = {0, 0})
            : SyncedImage(size_, T::num_channels, DataType<Scalar_t>::depth_flag, T::pixel_format)
        {
        }

        TSyncedImage(const SyncedImage& other)
            : SyncedImage(other)
        {
            MO_ASSERT_EQ(dataType(), DataType<Scalar_t>::depth_flag);
            MO_ASSERT_EQ(pixelFormat(), ct::value(T::pixel_format));
        }

        TSyncedImage(SyncedImage&& other)
            : SyncedImage(std::move(other))
        {
            MO_ASSERT_EQ(dataType(), DataType<Scalar_t>::depth_flag);
            MO_ASSERT_EQ(pixelFormat(), ct::value(T::pixel_format));
        }

        ConstMatrix_t host(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const
        {
            return SyncedImage::template host<T>(stream, sync_required);
        }

        ConstMatrix_t device(mo::IDeviceStream* stream = nullptr, bool* sync_required = nullptr) const
        {
            return SyncedImage::template device<T>(stream, sync_required);
        }
    };
} // namespace aq
namespace ct
{
    REFLECT_TEMPLATED_DERIVED(aq::TSyncedImage, aq::SyncedImage)

    REFLECT_END;
} // namespace ct

#endif // AQUILA_TSYNCED_IMAGE_HPP
