#include "SyncedImage.hpp"

#include "CompressedImage.hpp"
#include "IImageCompressor.hpp"

#include <ce/execute.hpp>

#include <fstream>
namespace aq
{

    void binaryReadFromDisk(ce::shared_ptr<SyncedMemory>& out,
                            const boost::filesystem::path& path,
                            std::shared_ptr<mo::IAsyncStream> stream)
    {
        std::ifstream ifs(path.string(), std::ios::binary | std::ios::ate);
        if (!ifs.is_open())
        {
            MO_LOG(warn, "Unable to open {}", path);
            return;
            // return ct::error("Unable to open file");
        }
        const auto size = boost::filesystem::file_size(path);
        if (size == 0)
        {
            MO_LOG(warn, "Read an empty image {}", path);
            return;
            // return ct::error("Read an empty image");
        }

        ifs.seekg(0, std::ios::beg);

        if (!out)
        {
            out = ce::make_shared<SyncedMemory>(size, 1, stream);
        }
        else
        {
            out->resize(size, 1, stream);
        }

        bool sync_required = false;
        ct::TArrayView<void> data = out->mutableHost(stream.get(), &sync_required);

        ifs.read(ct::ptrCast<char>(data.begin()), static_cast<std::streamsize>(size));
        ifs.close();
    }

    void loadImpl(CompressedImage& out, const boost::filesystem::path& path, std::shared_ptr<mo::IAsyncStream> stream)
    {
        if (!boost::filesystem::exists(path))
        {
            MO_LOG(warn, "File {} doesn't exist", path);
            return;
        }
        if (!boost::filesystem::is_regular_file(path))
        {
            MO_LOG(warn, "File {} is not a regular file", path);
            return;
        }
        const auto extension = path.extension();
        // extension includes the '.', whereas our extensions do not
        const auto encoding = ct::fromString<ImageEncoding>(extension.c_str() + 1, false);
        if (!encoding.success())
        {
            MO_LOG(warn, "Unable to determine compression encoding from file path extension {}", path);
        }
        std::shared_ptr<mo::Allocator> allocator = stream->hostAllocator();
        if (allocator == nullptr)
        {
            allocator = mo::Allocator::getDefault();
        }
        if (allocator == nullptr)
        {
            MO_LOG(warn, "Unable to get default allocator");
            return;
        }
        ce::shared_ptr<SyncedMemory> memory;
        ce::exec(&binaryReadFromDisk, ce::makeOutput(memory), path, stream);
        if (memory)
        {
            out = CompressedImage(std::move(memory), encoding);
            const size_t fhash = ce::generateHash(&loadImpl);
            const size_t arghash = ce::generateHash(path.string());
            const size_t combined_hash = ce::combineHash(fhash, arghash);
            out.setHash(combined_hash);
        }
    }

    CompressedImage::CompressedImage() = default;

    CompressedImage::CompressedImage(const CompressedImage&) = default;
    CompressedImage::CompressedImage(CompressedImage&&) = default;

    void CompressedImage::load(CompressedImage& out,
                               const boost::filesystem::path& path,
                               std::shared_ptr<mo::IAsyncStream> stream)
    {
        return ce::exec(&loadImpl, ce::makeOutput(out), path, stream);
    }

    CompressedImage::CompressedImage(ce::shared_ptr<SyncedMemory>&& data, ImageEncoding enc)
        : m_data(std::move(data))
        , m_enc(std::move(enc))
    {
    }

    CompressedImage& CompressedImage::operator=(const std::vector<uint8_t>& data)
    {
        if (!m_data)
        {
            m_data = ce::make_shared<SyncedMemory>();
        }
        m_data->resize(data.size(), 1);
        ct::TArrayView<void> view = m_data->mutableHost();
        std::memcpy(view.begin(), data.data(), data.size());
        return *this;
    }

    CompressedImage& CompressedImage::operator=(const CompressedImage&) = default;
    CompressedImage& CompressedImage::operator=(CompressedImage&&) = default;

    /*void CompressedImage::image(SyncedImage& img, rcc::shared_ptr<IImageDecompressor> decompressor) const
    {
        if (!decompressor)
        {
            decompressor = IImageDecompressor::create(m_enc);
        }
        MO_ASSERT(decompressor != nullptr);
        decompressor->decompress(*this, img);
    }*/

    ct::TArrayView<const void> CompressedImage::host() const
    {
        ct::TArrayView<const void> out;
        if (m_data)
        {
            out = m_data->host();
        }
        return out;
    }

    ct::TArrayView<void> CompressedImage::mutableHost()
    {
        ct::TArrayView<void> out;
        if (m_data)
        {
            out = m_data->mutableHost();
        }
        return out;
    }

    void CompressedImage::copyData(ct::TArrayView<const void> data)
    {
        if (!m_data)
        {
            m_data = ce::make_shared<SyncedMemory>();
        }
        m_data->resize(data.size(), 1);
        ct::TArrayView<void> view = m_data->mutableHost();
        std::memcpy(view.data(), data.begin(), data.size());
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
        if (!m_data)
        {
            return false;
        }
        ct::TArrayView<const void> view = m_data->host();
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

    void CompressedImage::setData(ce::shared_ptr<SyncedMemory> data)
    {
        m_data = std::move(data);
    }

    ce::shared_ptr<const SyncedMemory> CompressedImage::getData() const
    {
        return m_data;
    }

    bool CompressedImage::empty() const
    {
        if (!m_data)
        {
            return true;
        }
        return m_data->empty();
    }
} // namespace aq
