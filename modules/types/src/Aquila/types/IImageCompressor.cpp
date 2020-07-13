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

    void IImageCompressor::compress(const SyncedImage& img, CompressedImage& ret, ImageEncoding enc) const
    {
        auto eng = ce::ICacheEngine::instance();
        MO_ASSERT(eng != nullptr);
        eng->exec(&IImageCompressor::compressImpl, ce::makeEmptyInput(*this), img, ce::makeOutput(ret), enc);

        size_t fhash, arghash;
        const auto compression_hit = eng->wasCacheUsedLast();
        SyncedImage decompressed = img;
        auto result = eng->getCachedResult(fhash,
                                           arghash,
                                           &IImageDecompressor::decompressImpl,
                                           ce::makeEmptyInput(*this),
                                           ce::wrapInput(ret),
                                           ce::makeOutput(decompressed));
        eng->setCacheWasUsed(compression_hit);

        using PackType = typename decltype(result)::element_type;
        const size_t original_hash = ret.hash();
        result = std::make_shared<PackType>();
        const auto combined = ce::combineHash(fhash, arghash);
        result->setHash(combined);
        result->saveOutputs(ce::wrapInput(ret), ce::makeOutput(decompressed));
        std::get<0>(result->values).setHash(img.hash());
        eng->pushCachedResult(std::move(result), fhash, arghash);
    }

    auto IImageDecompressor::create(ImageEncoding encoding) -> Ptr_t
    {
        auto factory = mo::MetaObjectFactory::instance();
        MO_ASSERT(factory != nullptr);
        return factory->createBestObject<IImageDecompressor, ImageDecompressorInfo>(
            s_interfaceID, [encoding](const ImageDecompressorInfo& info) { return info.priority(encoding); });
    }

    void IImageDecompressor::decompress(const CompressedImage& compressed, SyncedImage& img) const
    {
        auto eng = ce::ICacheEngine::instance();
        const size_t original_hash = compressed.hash();

        eng->exec(&IImageDecompressor::decompressImpl, ce::makeEmptyInput(*this), compressed, ce::makeOutput(img));

        size_t fhash = 0;
        size_t arghash = 0;
        CompressedImage copy = compressed;
        auto enc = compressed.getEncoding();
        eng->calcHash(fhash,
                      arghash,
                      &IImageCompressor::compressImpl,
                      ce::makeEmptyInput(*this),
                      static_cast<const SyncedImage&>(img),
                      ce::makeOutput(copy),
                      enc);

        auto result = eng->getCachedResult(fhash,
                                           arghash,
                                           &IImageCompressor::compressImpl,
                                           ce::makeEmptyInput(*this),
                                           static_cast<const SyncedImage&>(img),
                                           ce::makeOutput(copy),
                                           enc);
        if (result)
        {
            return;
        }

        auto combined = ce::combineHash(fhash, arghash);
        result = std::make_shared<typename decltype(result)::element_type>();

        /*using T = const aq::SyncedImage&;
        static_assert(ce::result_traits::DerivedFromHashedBase<T>::value, "");
        ct::StaticEqualTypes<ce::result_traits::RemoveRef_t<T>, const aq::SyncedImage>{};
        static_assert(ce::result_traits::IsConst<ce::result_traits::RemoveRef_t<T>>::value == true, "");
        static_assert(ce::result_traits::IsOutput<T, T>::value == false, "");*/

        result->setHash(combined);
        result->saveOutputs(static_cast<const SyncedImage&>(img), ce::makeOutput(copy), enc);
        std::get<0>(result->values).setHash(original_hash);
        eng->pushCachedResult(result, fhash, arghash);
    }
} // namespace aq
