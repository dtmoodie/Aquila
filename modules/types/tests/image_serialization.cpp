#include <gtest/gtest.h>

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/types/CompressedImage.hpp>

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

    aq::CompressedImage img;
    aq::CompressedImage::load(img, path);
    return std::make_shared<aq::CompressedImage>(std::move(img));
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

        ce::shared_ptr<const aq::SyncedMemory> loaded_data = loaded->getData();
        ce::shared_ptr<const aq::SyncedMemory> compressed_data = compressed->getData();

        ASSERT_NE(loaded_data->host().data(), compressed_data->host().data());

        ct::TArrayView<const uint8_t> loaded_data_view = loaded_data->host();
        ct::TArrayView<const uint8_t> compressed_data_view = compressed_data->host();

        ASSERT_EQ(loaded_data_view.size(), compressed_data_view.size());
        for (size_t i = 0; i < loaded_data_view.size(); ++i)
        {
            ASSERT_EQ(loaded_data_view[i], compressed_data_view[i]) << i;
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

        auto loaded_data = loaded->getData();
        auto compressed_data = compressed->getData();

        ASSERT_NE(loaded_data->host().data(), compressed_data->host().data());

        ct::TArrayView<const uint8_t> loaded_data_view = loaded_data->host();
        ct::TArrayView<const uint8_t> compressed_data_view = compressed_data->host();
        ASSERT_EQ(loaded_data_view.size(), compressed_data_view.size());
        for (size_t i = 0; i < loaded_data_view.size(); ++i)
        {
            ASSERT_EQ(loaded_data_view[i], compressed_data_view[i]) << i;
        }
        auto original_encoding = compressed->getEncoding();
        auto loaded_encoding = loaded->getEncoding();
        ASSERT_EQ(original_encoding, loaded_encoding);
    }
}