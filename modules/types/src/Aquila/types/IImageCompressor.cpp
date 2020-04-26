#include "SyncedImage.hpp"

#include "IImageCompressor.hpp"

#include <ce/ICacheEngine.hpp>
#include <ct/static_asserts.hpp>
namespace aq
{

    auto IImageCompressor::create(ImageEncoding encoding) -> Ptr_t
    {
        auto factory = mo::MetaObjectFactory::instance();

        auto obj = factory->createBestObject<IImageCompressor, ImageCompressorInfo>(
            s_interfaceID, [encoding](const ImageCompressorInfo& info) { return info.priority(encoding); });

        return obj;
    }

    CompressedImage IImageCompressor::compress(const SyncedImage& img, ImageEncoding enc) const
    {
        auto eng = ce::ICacheEngine::instance();
        MO_ASSERT(eng != nullptr);
        auto ret = eng->exec(&IImageCompressor::compressImpl, ce::makeEmptyInput(*this), img, enc);

        size_t fhash, arghash;
        const auto compression_hit = eng->wasCacheUsedLast();
        auto result =
            eng->getCachedResult(fhash, arghash, &IImageDecompressor::decompressImpl, ce::makeEmptyInput(*this), ret);
        eng->setCacheWasUsed(compression_hit);
        if (result)
        {
            return ret;
        }
        using PackType = typename decltype(result)::element_type;
        // ct::StaticEquality<int, PackType::OUTPUT_COUNT, 1>{};
        result = std::make_shared<PackType>();
        const auto combined = ce::combineHash(fhash, arghash);
        auto copy = img;
        result->setHash(combined);
        result->saveOutputs(copy, ret);
        std::get<0>(result->values).setHash(img.hash());
        eng->pushCachedResult(std::move(result), fhash, arghash);
        return ret;
    }

    auto IImageDecompressor::create(ImageEncoding encoding) -> Ptr_t
    {
        auto factory = mo::MetaObjectFactory::instance();
        MO_ASSERT(factory != nullptr);
        return factory->createBestObject<IImageDecompressor, ImageDecompressorInfo>(
            s_interfaceID, [encoding](const ImageDecompressorInfo& info) { return info.priority(encoding); });
    }

    SyncedImage IImageDecompressor::decompress(const CompressedImage& compressed) const
    {
        auto eng = ce::ICacheEngine::instance();

        auto ret = eng->exec(&IImageDecompressor::decompressImpl, ce::makeEmptyInput(*this), compressed);

        size_t fhash, arghash;
        auto enc = compressed.getEncoding();
        auto result = eng->getCachedResult(fhash,
                                           arghash,
                                           &IImageCompressor::compressImpl,
                                           ce::makeEmptyInput(*this),
                                           static_cast<const SyncedImage&>(ret),
                                           enc);
        if (result)
        {
            return ret;
        }

        auto combined = ce::combineHash(fhash, arghash);
        result = std::make_shared<typename decltype(result)::element_type>();

        auto copy = compressed;
        result->setHash(combined);
        result->saveOutputs(copy, static_cast<const SyncedImage&>(ret), enc);
        std::get<0>(result->values).setHash(compressed.hash());
        eng->pushCachedResult(result, fhash, arghash);
        return ret;
    }
} // namespace aq
