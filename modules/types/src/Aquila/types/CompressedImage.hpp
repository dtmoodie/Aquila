#pragma once
#include <Aquila/types.hpp>

#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/types/TDynArray.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <ce/hash.hpp>
#include <ce/output.hpp>

#include <ct/enum.hpp>
#include <ct/reflect.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <cstdint>
#include <memory>

namespace aq
{
    struct SyncedImage;
    struct IImageCompressor;
    struct IImageDecompressor;

    ENUM_BEGIN(ImageEncoding, uint8_t)
        ENUM_VALUE(JPEG, 0)
        ENUM_VALUE(JPG, JPEG)
        ENUM_VALUE(PNG, JPEG + 1)
        ENUM_VALUE(TIFF, PNG + 1)
        ENUM_VALUE(BMP, TIFF + 1)
        ENUM_VALUE(RAW, BMP)
        ENUM_VALUE(DNG, RAW + 1)
    ENUM_END;

    struct AQUILA_EXPORTS CompressedImage : ce::HashedBase
    {
        static std::shared_ptr<const CompressedImage> load(const boost::filesystem::path& path);

        CompressedImage();
        CompressedImage(mo::TDynArray<void, mo::Allocator>&& data, ImageEncoding enc);
        CompressedImage(const CompressedImage&);
        CompressedImage(CompressedImage&&);
        CompressedImage(boost::filesystem::path file_path);

        CompressedImage& operator=(const std::vector<uint8_t>& data);
        CompressedImage& operator=(const CompressedImage&);
        CompressedImage& operator=(CompressedImage&&);

        SyncedImage image(rcc::shared_ptr<IImageDecompressor> = rcc::shared_ptr<IImageDecompressor>()) const;

        ct::TArrayView<const void> data() const;
        ct::TArrayView<void> mutableData();

        void copyData(ct::TArrayView<const void>);
        void setData(mo::TDynArray<void, mo::Allocator>&& data);
        ImageEncoding getEncoding() const;
        void setEncoding(ImageEncoding);
        bool toDisk(boost::filesystem::path path = boost::filesystem::path()) const;
        bool empty() const;

      private:
        mo::TDynArray<void, mo::Allocator> m_data;
        ImageEncoding m_enc;
        boost::filesystem::path m_path;
    };
} // namespace aq

namespace ct
{
    REFLECT_BEGIN(aq::CompressedImage)
        PROPERTY(data, &aq::CompressedImage::data, &aq::CompressedImage::setData)
        PROPERTY(encoding, &aq::CompressedImage::getEncoding, &aq::CompressedImage::setEncoding)
        MEMBER_FUNCTION(toDisk)
        MEMBER_FUNCTION(empty)
    REFLECT_END;
} // namespace ct

namespace ce
{
    template <>
    struct HashSelector<boost::filesystem::path, void, 1>
    {
        static size_t generateHash(const boost::filesystem::path& path)
        {
            const auto hash = ce::generateHash(path.string());
            const auto time = boost::filesystem::last_write_time(path);
            return combineHash(hash, ce::generateHash(time));
        }
    };
} // namespace ce
