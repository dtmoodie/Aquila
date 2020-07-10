#include "SyncedImage.hpp"

#include "CompressedImage.hpp"
#include "IImageCompressor.hpp"

#include <ce/execute.hpp>

#include <fstream>
namespace aq
{

    ct::Result<mo::TDynArray<uint8_t, mo::Allocator>> binaryReadFromDisk(std::shared_ptr<mo::Allocator> allocator,
                                                                         const boost::filesystem::path& path)
    {
        std::ifstream ifs(path.string(), std::ios::binary | std::ios::ate);
        if (!ifs.is_open())
        {
            MO_LOG(warn, "Unable to open {}", path);
            return ct::error("Unable to open file");
        }
        const auto size = boost::filesystem::file_size(path);
        if (size == 0)
        {
            MO_LOG(warn, "Read an empty image {}", path);
            return ct::error("Read an empty image");
        }

        ifs.seekg(0, std::ios::beg);

        mo::TDynArray<uint8_t, mo::Allocator> arr(allocator, size);
        ifs.read(ct::ptrCast<char>(arr.mutableView().begin()), static_cast<std::streamsize>(size));
        ifs.close();
        return std::move(arr);
    }

    std::shared_ptr<const CompressedImage> loadImpl(const boost::filesystem::path& path)
    {
        if (!boost::filesystem::exists(path))
        {
            MO_LOG(warn, "File {} doesn't exist", path);
            return {};
        }
        if (!boost::filesystem::is_regular_file(path))
        {
            MO_LOG(warn, "File {} is not a regular file", path);
            return {};
        }
        const auto extension = path.extension();
        // extension includes the '.', whereas our extensions do not
        const auto encoding = ct::fromString<ImageEncoding>(extension.c_str() + 1, false);
        if (!encoding.success())
        {
            MO_LOG(warn, "Unable to determine compression encoding from file path extension {}", path);
        }
        auto alloc = mo::Allocator::getDefault();
        if (alloc == nullptr)
        {
            MO_LOG(warn, "Unable to get default allocator");
            return {};
        }
        auto result = ce::exec(&binaryReadFromDisk, alloc, path);
        if (result.data.success())
        {
            auto ret = std::make_shared<CompressedImage>(std::move(result.data.m_value), encoding);
            ret->setHash(ce::combineHash(ce::generateHash(&loadImpl), ce::generateHash(path.string())));
            return ret;
        }
        return {};
    }

    CompressedImage::CompressedImage() = default;

    CompressedImage::CompressedImage(const CompressedImage&) = default;
    CompressedImage::CompressedImage(CompressedImage&&) = default;

    std::shared_ptr<const CompressedImage> CompressedImage::load(const boost::filesystem::path& path)
    {
        return ce::exec(&loadImpl, path);
    }

    CompressedImage::CompressedImage(mo::TDynArray<void, mo::Allocator>&& data, ImageEncoding enc)
        : m_data(std::move(data))
        , m_enc(std::move(enc))
    {
    }

    CompressedImage& CompressedImage::operator=(const std::vector<uint8_t>& data)
    {
        // m_data = data;
        m_data.resize(data.size());
        auto view = m_data.mutableView();
        std::memcpy(view.begin(), data.data(), data.size());
        return *this;
    }

    CompressedImage& CompressedImage::operator=(const CompressedImage&) = default;
    CompressedImage& CompressedImage::operator=(CompressedImage&&) = default;

    void CompressedImage::image(SyncedImage& img, rcc::shared_ptr<IImageDecompressor> decompressor) const
    {
        if (!decompressor)
        {
            decompressor = IImageDecompressor::create(m_enc);
        }
        MO_ASSERT(decompressor != nullptr);
        decompressor->decompress(*this, img);
    }

    ct::TArrayView<const void> CompressedImage::data() const
    {
        return m_data;
    }

    ct::TArrayView<void> CompressedImage::mutableData()
    {
        return m_data;
    }

    void CompressedImage::copyData(ct::TArrayView<const void> data)
    {
        m_data.resize(data.size());
        std::memcpy(m_data.mutableView().begin(), data.begin(), data.size());
    }

    ImageEncoding CompressedImage::getEncoding() const
    {
        return m_enc;
    }

    void CompressedImage::setEncoding(ImageEncoding enc)
    {
        m_enc = enc;
    }

    bool CompressedImage::toDisk(boost::filesystem::path path) const
    {
        if (path.empty())
        {
            path = m_path;
        }
        if (path.empty())
        {
            return false;
        }
        auto view = m_data.view();
        if (view.size() == 0)
        {
            return false;
        }
        std::ofstream ofs(path.string());
        if (!ofs.is_open())
        {
            return false;
        }
        ofs.write(ct::ptrCast<char>(view.data()), view.size());
        return true;
    }

    void CompressedImage::setData(mo::TDynArray<void, mo::Allocator>&& data)
    {
        m_data = std::move(data);
    }

    bool CompressedImage::empty() const
    {
        return m_data.empty();
    }
} // namespace aq
