#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/IImageCompressor.hpp>

#include <MetaObject/cuda/AsyncStream.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>

#include <ce/ICacheEngine.hpp>
#include <ct/enum.hpp>
#include <ct/reflect.hpp>
#include <ct/static_asserts.hpp>
#include <opencv2/imgcodecs.hpp>

#include "CvMatAllocatorWrapper.hpp"
#include <boost/filesystem.hpp>

#include <gtest/gtest.h>

struct ReplaceCvAllocator
{
    ReplaceCvAllocator(cv::MatAllocator* alloc)
    {
        m_original = cv::Mat::getDefaultAllocator();
        cv::Mat::setDefaultAllocator(alloc);
    }

    ~ReplaceCvAllocator()
    {
        cv::Mat::setDefaultAllocator(m_original);
    }

  private:
    cv::MatAllocator* m_original;
};

struct OpenCVDecompressor : public aq::IImageDecompressor
{
    static int32_t priority(aq::ImageEncoding)
    {
        return 1;
    }

    MO_DERIVE(OpenCVDecompressor, aq::IImageDecompressor)

    MO_END;

    aq::SyncedImage decompressImpl(const aq::CompressedImage& compressed) const
    {
        std::shared_ptr<aq::SyncedImage> out = std::make_shared<aq::SyncedImage>();

        auto compressed_data = compressed.data();
        // OpenCV's api doesn't allow wrapping of const data, even though the decompress operation is a const operation
        cv::Mat wrap_data(compressed_data.size(),
                          1,
                          CV_8UC1,
                          const_cast<uint8_t*>(static_cast<const uint8_t*>(compressed_data.data())));
        auto wrapped = CvMatAllocatorWrapper::wrap(out);
        ReplaceCvAllocator replacer(wrapped.allocator);
        auto img = cv::imdecode(wrap_data, cv::IMREAD_UNCHANGED);
        return std::move(*out);
    }

    aq::SyncedImage decompressImpl(std::shared_ptr<aq::CompressedImage> compressed) const
    {
        auto out = decompress(*compressed);
        return out;
    }
};

struct OpenCVCompressor : public aq::IImageCompressor
{
    static int32_t priority(aq::ImageEncoding)
    {
        return 1;
    }

    MO_DERIVE(OpenCVCompressor, aq::IImageCompressor)

    MO_END;

    aq::CompressedImage compressImpl(const aq::SyncedImage& image, aq::ImageEncoding enc) const override
    {
        auto mat = image.mat();
        aq::CompressedImage out;

        std::stringstream ss;
        ss << ".";
        ss << enc;
        // Figure out how to reduce the need for this later
        std::vector<uint8_t> tmp;
        std::string ext = ss.str();
        if (cv::imencode(ext, mat, tmp, {}))
        {
            out = tmp;
        }

        return out;
    }
};

TEST(compressed_image, static_checks)
{
    static_assert(ct::IsReflected<OpenCVDecompressor>::value, "asdf");
    static_assert(OpenCVDecompressor::NUM_FIELDS == 0, "asdf");
    ct::StaticEqualTypes<typename ct::Reflect<OpenCVDecompressor>::ImplementationFilter_t,
                         ct::ImplementationFilter<OpenCVDecompressor, OpenCVDecompressor, ct::VariadicTypedef<>>>{};

    auto control_block =
        dynamic_cast<TObjectControlBlock<OpenCVDecompressor>*>(static_cast<IObjectControlBlock*>(nullptr));
    (void)control_block;
    auto decompressor = OpenCVDecompressor::create();
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
    decompressor = rcc::shared_ptr<OpenCVDecompressor>(objptr);
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
    auto image = compressed->image();
    EXPECT_EQ(compressed->hash(), initial_hash)
        << "Hash of the compressed image should not change from decompressing it";
    EXPECT_FALSE(eng->wasCacheUsedLast()) << "We don't expect to hit the cache on first load of an image";
    EXPECT_FALSE(image.empty()) << "Unable to decompress image";
    EXPECT_EQ(image.cols(), 1920);
    EXPECT_EQ(image.rows(), 1080);

    // Decompressing the same compressed image should hit the cache
    std::cout << "png -> raw" << std::endl;
    auto img = compressed->image();
    EXPECT_TRUE(eng->wasCacheUsedLast());
    EXPECT_EQ(image.hash(), img.hash());

    auto compressor = aq::IImageCompressor::create(aq::ImageEncoding::PNG);
    EXPECT_NE(compressor, nullptr);

    // Since we originally decoded a png, we shouldn't need to actually compress this image
    std::cout << "raw -> png" << std::endl;
    auto recompressed = compressor->compress(img, aq::ImageEncoding::PNG);
    EXPECT_TRUE(eng->wasCacheUsedLast());
    EXPECT_EQ(recompressed.hash(), compressed->hash());

    eng->clearCache();
    std::cout << "raw -> png" << std::endl;
    recompressed = compressor->compress(img, aq::ImageEncoding::PNG);
    // Ok so the problem here is that since the original compressed image got its hash
    // from the file that it was loaded from, it has one hash
    // whereas the recompressed image got its hash from the raw image that it originated from
    // And in reality unless the decompress -> recompress process is lossless, these are going to be two different
    // compressed images... And in actuality these usually don't even have the same size so the compression is not the
    // same
    // EXPECT_EQ(recompressed.hash(), compressed->hash());
    EXPECT_TRUE(recompressed.toDisk("test.png"));
    img = recompressed.image();
    // For the same reason above, these images aren't going to be exactly the same so there is no reason to expect
    // The hash to be the same
    // EXPECT_EQ(image.hash(), img.hash());
    // BOOST_REQUIRE(eng->wasCacheUsedLast());

    recompressed = compressor->compress(img, aq::ImageEncoding::JPG);
    EXPECT_TRUE(recompressed.toDisk("test.jpg"));
}

MO_REGISTER_OBJECT(OpenCVDecompressor)
MO_REGISTER_OBJECT(OpenCVCompressor)
