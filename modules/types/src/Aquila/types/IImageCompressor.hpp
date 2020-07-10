#ifndef AQ_TYPES_IMAGE_COMPRESSOR_HPP
#define AQ_TYPES_IMAGE_COMPRESSOR_HPP

#include "CompressedImage.hpp"
#include <MetaObject/object/MetaObject.hpp>
#include <RuntimeObjectSystem/IObjectInfo.h>
namespace aq
{
    struct ImageDecompressorInfo;
    struct ImageCompressorInfo;
    struct AQUILA_EXPORTS IImageCompressor : virtual public TInterface<IImageCompressor, mo::MetaObject>
    {
        using InterfaceInfo = ImageCompressorInfo;
        using Ptr_t = rcc::shared_ptr<IImageCompressor>;

        MO_DERIVE(IImageCompressor, mo::MetaObject)

        MO_END;

        static Ptr_t create(ImageEncoding encoding);

        void compress(const SyncedImage&, CompressedImage& compressed, ImageEncoding = ImageEncoding::JPEG) const;

      protected:
        virtual void compressImpl(const SyncedImage&, CompressedImage& compressed, ImageEncoding) const = 0;
        friend struct IImageDecompressor;

      private:
        ImageEncoding m_encoding;
    };

    struct AQUILA_EXPORTS IImageDecompressor : virtual public TInterface<IImageDecompressor, mo::MetaObject>
    {
        using InterfaceInfo = ImageDecompressorInfo;
        using Ptr_t = rcc::shared_ptr<IImageDecompressor>;

        MO_DERIVE(IImageDecompressor, mo::MetaObject)
        MO_END;

        static Ptr_t create(ImageEncoding encoding);

        void decompress(const CompressedImage& compressed, SyncedImage& img) const;

      protected:
        friend struct IImageCompressor;
        virtual void decompressImpl(const CompressedImage& compressed, SyncedImage& img) const = 0;
    };

    struct AQUILA_EXPORTS ImageDecompressorInfo : mo::MetaObject::InterfaceInfo
    {
        virtual int32_t priority(ImageEncoding) const = 0;
    };

    struct AQUILA_EXPORTS ImageCompressorInfo : mo::MetaObject::InterfaceInfo
    {
        virtual int32_t priority(ImageEncoding) const = 0;
    };
} // namespace aq

namespace ce
{
    template <class T>
    ct::EnableIf<ct::IsMemberProperty<T, 0>::value || ct::IsMemberObject<T, 0>::value, size_t>
    hashField(const T& obj, size_t hash, ct::Indexer<0> idx)
    {
        const auto accessor = ct::Reflect<T>::getPtr(idx);
        hash = ce::combineHash(hash, ce::generateHash(accessor.get(obj)));
        return hash;
    }

    template <class T>
    ct::EnableIf<!ct::IsMemberProperty<T, 0>::value && !ct::IsMemberObject<T, 0>::value, size_t>
    hashField(const T&, size_t hash, ct::Indexer<0>)
    {
        return hash;
    }

    template <class T, ct::index_t I>
    ct::EnableIf<ct::IsMemberProperty<T, I>::value || ct::IsMemberObject<T, I>::value, size_t>
    hashField(const T& obj, size_t hash, ct::Indexer<I> idx)
    {
        const auto accessor = ct::Reflect<T>::getPtr(idx);
        hash = ce::combineHash(hash, ce::generateHash(accessor.get(obj)));
        return hashField(obj, hash, --idx);
    }

    template <class T, ct::index_t I>
    ct::EnableIf<!ct::IsMemberProperty<T, I>::value && !ct::IsMemberObject<T, I>::value, size_t>
    hashField(const T& obj, size_t hash, ct::Indexer<I> idx)
    {
        return hashField(obj, hash, --idx);
    }

    template <class T>
    ct::EnableIfReflected<T, size_t> getObjectHash(const T& obj)
    {
        return hashField(obj, 0, ct::Reflect<T>::end());
    }

    template <class T>
    struct HashSelector<T, ct::EnableIfIsEnum<T>, 1>
    {
        static size_t generateHash(const T& val)
        {
            std::hash<decltype(val.value)> hasher;
            return hasher(val);
        }
    };
} // namespace ce

namespace mo
{
    template <class T>
    class MetaObjectInfoImpl<T, aq::ImageCompressorInfo> : public aq::ImageCompressorInfo
    {
        virtual int32_t priority(aq::ImageEncoding enc) const
        {
            return T::priority(enc);
        }
    };

    template <class T>
    class MetaObjectInfoImpl<T, aq::ImageDecompressorInfo> : public aq::ImageDecompressorInfo
    {
        virtual int32_t priority(aq::ImageEncoding enc) const
        {
            return T::priority(enc);
        }
    };
} // namespace mo

#endif // AQ_TYPES_IMAGE_COMPRESSION_ENGINE_HPP
