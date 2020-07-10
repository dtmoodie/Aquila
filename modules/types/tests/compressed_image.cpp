#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/IImageCompressor.hpp>

#include <MetaObject/cuda/AsyncStream.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>

#include <ce/ICacheEngine.hpp>
#include <ct/enum.hpp>
#include <ct/reflect.hpp>
#include <ct/static_asserts.hpp>

#include "cv_compression/CvMatAllocatorWrapper.hpp"
#include <boost/filesystem.hpp>

#include "cv_compression/cv_compressor.hpp"
#include "cv_compression/cv_decompressor.hpp"

#include <gtest/gtest.h>

TEST(compressed_image, static_checks)
{
    static_assert(ct::IsReflected<aq::OpenCVDecompressor>::value, "asdf");
    static_assert(aq::OpenCVDecompressor::NUM_FIELDS == 0, "asdf");
    ct::StaticEqualTypes<
        typename ct::Reflect<aq::OpenCVDecompressor>::ImplementationFilter_t,
        ct::ImplementationFilter<aq::OpenCVDecompressor, aq::OpenCVDecompressor, ct::VariadicTypedef<>>>{};

    auto control_block =
        dynamic_cast<TObjectControlBlock<aq::OpenCVDecompressor>*>(static_cast<IObjectControlBlock*>(nullptr));
    (void)control_block;
    auto decompressor = aq::OpenCVDecompressor::create();
    auto typed_cb = decompressor.GetControlBlock();
    auto cb = typed_cb.get();
    EXPECT_NE(dynamic_cast<TObjectControlBlock<aq::IImageDecompressor>*>(cb), nullptr);
    EXPECT_NE(dynamic_cast<TObjectControlBlock<mo::MetaObject>*>(cb), nullptr);
    EXPECT_NE(dynamic_cast<TObjectControlBlock<IObject>*>(cb), nullptr);
    EXPECT_NE(decompressor, nullptr);
    rcc::shared_ptr<IObject> objptr(decompressor);
    EXPECT_NE(objptr, nullptr);
    rcc::shared_ptr<aq::IImageDecompressor> decptr(objptr);
    EXPECT_NE(decptr, nullptr);
    decompressor = rcc::shared_ptr<aq::OpenCVDecompressor>(objptr);
    EXPECT_NE(decompressor, nullptr);
}

TEST(compressed_image, png_load)
{
    auto eng = ce::ICacheEngine::instance();
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    EXPECT_NE(stream, nullptr) << "Unable to create a cuda stream";
    stream->setCurrent(stream);
    auto path = boost::filesystem::path("../data.png");
    if (!boost::filesystem::exists(path))
    {
        if (boost::filesystem::exists("build/data.png"))
        {
            path = boost::filesystem::canonical(boost::filesystem::path("build/data.png"));
        }
        if (boost::filesystem::exists("data.png"))
        {
            path = boost::filesystem::canonical("data.png");
        }
    }
    ASSERT_TRUE(boost::filesystem::exists(path)) << "Cannot find data.png file used for unit test " << path;
    path = boost::filesystem::canonical(path);
    auto compressed = aq::CompressedImage::load(path);
    EXPECT_TRUE(compressed != nullptr) << "Unable to load image data from disk";
    EXPECT_FALSE(eng->wasCacheUsedLast());
    {
        // additional loads of the same path should hit the cache
        auto cmp = aq::CompressedImage::load(path);
        EXPECT_TRUE(eng->wasCacheUsedLast());
        EXPECT_EQ(cmp.get(), compressed.get());
    }
    eng->printDebug(true);
    const auto initial_hash = compressed->hash();
    std::cout << "png -> raw" << std::endl;
    aq::SyncedImage image;
    compressed->image(image);
    EXPECT_EQ(compressed->hash(), initial_hash)
        << "Hash of the compressed image should not change from decompressing it";
    EXPECT_FALSE(eng->wasCacheUsedLast()) << "We don't expect to hit the cache on first load of an image";
    EXPECT_FALSE(image.empty()) << "Unable to decompress image";
    EXPECT_EQ(image.cols(), 1920);
    EXPECT_EQ(image.rows(), 1080);

    // Decompressing the same compressed image should hit the cache
    std::cout << "png -> raw" << std::endl;
    aq::SyncedImage img;
    compressed->image(img);
    EXPECT_TRUE(eng->wasCacheUsedLast());
    EXPECT_EQ(image.hash(), img.hash());

    auto compressor = aq::IImageCompressor::create(aq::ImageEncoding::PNG);
    EXPECT_NE(compressor, nullptr);

    // Since we originally decoded a png, we shouldn't need to actually compress this image
    std::cout << "raw -> png" << std::endl;
    aq::CompressedImage recompressed;
    compressor->compress(img, recompressed, aq::ImageEncoding::PNG);
    EXPECT_TRUE(eng->wasCacheUsedLast());
    EXPECT_EQ(recompressed.hash(), compressed->hash());

    eng->clearCache();
    std::cout << "raw -> png" << std::endl;
    compressor->compress(img, recompressed, aq::ImageEncoding::PNG);
    // Ok so the problem here is that since the original compressed image got its hash
    // from the file that it was loaded from, it has one hash
    // whereas the recompressed image got its hash from the raw image that it originated from
    // And in reality unless the decompress -> recompress process is lossless, these are going to be two different
    // compressed images... And in actuality these usually don't even have the same size so the compression is not the
    // same
    // EXPECT_EQ(recompressed.hash(), compressed->hash());
    EXPECT_TRUE(recompressed.toDisk("test.png"));
    recompressed.image(img);
    // For the same reason above, these images aren't going to be exactly the same so there is no reason to expect
    // The hash to be the same
    // EXPECT_EQ(image.hash(), img.hash());
    // BOOST_REQUIRE(eng->wasCacheUsedLast());

    compressor->compress(img, recompressed, aq::ImageEncoding::JPG);
    EXPECT_TRUE(recompressed.toDisk("test.jpg"));
}
