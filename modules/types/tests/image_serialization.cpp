#include <gtest/gtest.h>

#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <MetaObject/cuda/AsyncStream.hpp>

#include <MetaObject/runtime_reflection/StructTraits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>
#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>

#include <fstream>

static_assert(ct::IsReflected<aq::CompressedImage>::value, "");
static_assert(std::is_same<typename mo::TTraits<aq::CompressedImage>::base, mo::IStructTraits>::value, "");

std::shared_ptr<const aq::CompressedImage> loadImage()
{
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
    path = boost::filesystem::canonical(path);

    auto compressed = aq::CompressedImage::load(path);
    return compressed;
}

TEST(image_serialization, compressed_json)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    EXPECT_NE(stream, nullptr) << "Unable to create a cuda stream";
    stream->setCurrent(stream);
    auto compressed = loadImage();

    ASSERT_TRUE(compressed) << "Unable to load image, probably couldn't find the data.png file";

    {
        std::ofstream ofs("test.json");
        mo::JSONSaver saver(ofs);
        saver(&compressed, "compressed_ptr");
    }

    {
        std::ifstream ifs("test.json");
        mo::JSONLoader loader(ifs);
        std::shared_ptr<aq::CompressedImage> loaded;
        loader(&loaded, "compressed_ptr");
        ASSERT_TRUE(loaded);
        ASSERT_NE(loaded->data().begin(), compressed->data().begin());
        ct::TArrayView<const uint8_t> loaded_data = loaded->data();
        ct::TArrayView<const uint8_t> compressed_data = compressed->data();
        ASSERT_EQ(loaded_data.size(), compressed_data.size());
        for (size_t i = 0; i < loaded_data.size(); ++i)
        {
            ASSERT_EQ(loaded_data[i], compressed_data[i]) << i;
        }
        auto original_encoding = compressed->getEncoding();
        auto loaded_encoding = loaded->getEncoding();
        ASSERT_EQ(original_encoding, loaded_encoding);
    }
}

TEST(image_serialization, compressed_binary)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    EXPECT_NE(stream, nullptr) << "Unable to create a cuda stream";
    stream->setCurrent(stream);
    auto compressed = loadImage();
    ASSERT_TRUE(compressed) << "Unable to load image, probably couldn't find the data.png file";

    {
        std::ofstream ofs("test.bin");
        mo::BinarySaver saver(ofs);
        saver(&compressed, "compressed_ptr");
    }
    {
        std::ifstream ifs("test.bin");
        mo::BinaryLoader loader(ifs);
        std::shared_ptr<aq::CompressedImage> loaded;
        loader(&loaded, "compressed_ptr");
        ASSERT_TRUE(loaded);
        ASSERT_NE(loaded->data().begin(), compressed->data().begin());
        ct::TArrayView<const uint8_t> loaded_data = loaded->data();
        ct::TArrayView<const uint8_t> compressed_data = compressed->data();
        ASSERT_EQ(loaded_data.size(), compressed_data.size());
        for (size_t i = 0; i < loaded_data.size(); ++i)
        {
            ASSERT_EQ(loaded_data[i], compressed_data[i]) << i;
        }
        auto original_encoding = compressed->getEncoding();
        auto loaded_encoding = loaded->getEncoding();
        ASSERT_EQ(original_encoding, loaded_encoding);
    }
}