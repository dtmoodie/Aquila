
#include <MetaObject/cuda/AsyncStream.hpp>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <gtest/gtest.h>

#include <MetaObject/logging/profiling.hpp>

#define private public
#include "cv_compression/CvMatAllocatorWrapper.hpp"
#include <Aquila/types/CvMatAllocator.hpp>
#include <Aquila/types/SyncedImage.hpp>

TEST(shape, shape)
{
    aq::Shape<2> shape2(0, 0);
    EXPECT_EQ(shape2(0), 0);
    EXPECT_EQ(shape2(1), 0);

    shape2 = aq::Shape<2>(1, 2);
    EXPECT_EQ(shape2(0), 1);
    EXPECT_EQ(shape2(1), 2);
}

TEST(synced_image, constructor)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    mo::IDeviceStream::setCurrent(stream);
    auto image = std::make_shared<aq::SyncedImage>();

    auto synced_mem = image->data();
    EXPECT_NE(synced_mem.get(), nullptr);
    EXPECT_EQ(synced_mem->size(), 0);
    EXPECT_EQ(synced_mem.get(), image->m_data.get());

    auto strm = synced_mem->getStream().lock();

    EXPECT_EQ(strm.get(), stream.get());

    synced_mem = image->data();
    EXPECT_EQ(synced_mem.get(), image->m_data.get());
    auto old = synced_mem;

    image->create(1024, 512, aq::PixelFormat::kRGB);
    synced_mem = image->data();
    EXPECT_NE(old.get(), synced_mem.get());
    EXPECT_NE(synced_mem.get(), nullptr);

    EXPECT_THROW(image->host<aq::RGB<uint16_t>>(), mo::TExceptionWithCallstack<std::runtime_error>);
    EXPECT_THROW(image->host<aq::BGR<uint8_t>>(), mo::TExceptionWithCallstack<std::runtime_error>);

    auto data = image->host<aq::RGB<uint8_t>>();
    EXPECT_EQ(data.rows(), 1024);
    EXPECT_EQ(data.cols(), 512);

    {
        const aq::SyncedImage cimage(*image);
        {
            EXPECT_EQ(cimage.m_data.m_data.get(), image->m_data.m_data.get());

            auto cdata = cimage.data();
            auto data = image->data();

            EXPECT_EQ(cdata.isConst(), true);
            EXPECT_EQ(cdata.m_data.get(), data.m_data.get());
            EXPECT_EQ(cdata.get(), data.get());
        }

        // Test copy on write stuffs
        {
            aq::SyncedImage rwimage(cimage);
            EXPECT_EQ(rwimage.m_data.m_data.get(), cimage.m_data.m_data.get());
            EXPECT_EQ(rwimage.m_data.m_data.get(), image->m_data.m_data.get());
            EXPECT_TRUE(rwimage.m_data.isConst());
            // When grabbing a const view of the data, we expect it to not copy and this should be fine and dandy
            rwimage.host<aq::RGB<uint8_t>>();
            EXPECT_EQ(rwimage.m_data.m_data.get(), cimage.m_data.m_data.get());
            EXPECT_EQ(rwimage.m_data.m_data.get(), image->m_data.m_data.get());

            // Now since we are grabbing a mutable view of the data, we want to copy it such that other references to
            // the data are happy
            rwimage.mutableHost<aq::RGB<uint8_t>>();
            EXPECT_NE(rwimage.m_data.m_data.get(), cimage.m_data.m_data.get());
            EXPECT_NE(rwimage.m_data.m_data.get(), image->m_data.m_data.get());
            EXPECT_FALSE(rwimage.m_data.isConst());
        }

        EXPECT_THROW(cimage.host<aq::RGB<uint16_t>>(), mo::TExceptionWithCallstack<std::runtime_error>);
        auto cdata = cimage.host<aq::RGB<uint8_t>>();

        EXPECT_EQ(data.data(), cdata.data());
    }
}

TEST(synced_image, reflect)
{
    auto ptr = ct::Reflect<aq::SyncedImage>::getPtr(ct::Indexer<0>{});

    aq::SyncedImage img;
    EXPECT_EQ(img.rows(), 0);
    EXPECT_EQ(img.cols(), 0);
    auto data = ptr.get(img);
    auto ref = ptr.set(img);
    ptr.set(img, data);
    ref = data;
}

#ifdef HAVE_OPENCV

TEST(synced_image_opencv, global_allocator)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    aq::CvMatAllocator alloc(stream);
    aq::ScopedOpenCVAllocator allocator_context_manager(&alloc);
    cv::Mat img;
    img.create(1024, 512, CV_32F);
    auto talloc = dynamic_cast<const aq::CvMatAllocator*>(img.u->currAllocator);
    EXPECT_EQ(talloc, &alloc);
    auto synced = alloc.getSyncedImage(img);
    EXPECT_NE(synced, nullptr);
    EXPECT_EQ(synced->rows(), 1024);
    EXPECT_EQ(synced->cols(), 512);
    EXPECT_EQ(synced->pixelType().data_type, ct::value(aq::DataFlag::kFLOAT32));
}

TEST(synced_image_opencv, sync_data)
{

    mo::initProfiling();
    PROFILE_FUNCTION
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    aq::SyncedImage image;
    // Count non zero only takes gray images
    image.create(1024, 1024, aq::PixelFormat::kGRAY);
    image.setStream(stream);

    auto gpu_mat = image.mutableGpuMat();
    EXPECT_EQ(gpu_mat.channels(), 1);
    gpu_mat.setTo(cv::Scalar::all(100));
    bool sync = false;
    const auto cpu_mat = image.mat(nullptr, &sync);
    EXPECT_TRUE(sync);
    stream->synchronize();

    EXPECT_EQ(cv::countNonZero(cpu_mat == 100), 1024 * 1024);
}

TEST(synced_image_opencv, wrapping_allocator)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    auto wrapped = cv::cuda::StreamAccessor::wrapStream(stream->getStream());
    auto img = std::make_shared<aq::SyncedImage>();
    img->setStream(stream);
    auto mat = aq::wrap(img);
    mat.create(1024, 512, CV_32F);
    mat.setTo(cv::Scalar::all(150));

    EXPECT_EQ(cv::countNonZero(mat == 150), 1024 * 512);

    auto shape = img->shape();
    EXPECT_EQ(shape(0), 1024);
    EXPECT_EQ(shape(1), 512);

    auto gpu_mat = img->mutableGpuMat();
    EXPECT_EQ(img->data()->device().data(), gpu_mat.data);
    EXPECT_EQ(gpu_mat.rows, 1024);
    EXPECT_EQ(gpu_mat.cols, 512);
    EXPECT_EQ(gpu_mat.depth(), CV_32F);
    auto val = img->dataType();
    EXPECT_EQ(val, ct::value(aq::DataFlag::kFLOAT32));
    cv::cuda::multiply(gpu_mat, cv::Scalar::all(2), gpu_mat, 1, -1, wrapped);
    img->host<aq::GRAY<float>>();
    EXPECT_EQ(cv::countNonZero(mat == 300), 1024 * 512);
}

TEST(synced_image_opencv, construct_from_mat)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    cv::Mat data(1024, 512, CV_32FC3);
    aq::SyncedImage img(data, aq::PixelFormat::kBGR, stream);
    EXPECT_EQ(img.rows(), 1024);
    EXPECT_EQ(img.cols(), 512);
    auto host_data = img.template mutableHost<aq::BGR<float>>();

    EXPECT_THROW(img.host<aq::BGR<uint8_t>>(), mo::TExceptionWithCallstack<std::runtime_error>);
    EXPECT_THROW(img.host<aq::RGB<float>>(), mo::TExceptionWithCallstack<std::runtime_error>);

    EXPECT_EQ(ct::ptrCast<uchar>(host_data.data()), data.data);
}

#endif

#include <Aquila/types/TSyncedImage.hpp>

template <class T>
void constructTImage(aq::SyncedImage& img)
{
    aq::TSyncedImage<T> tmp(img);
}

TEST(synced_image, tsynced_ctr)
{
    aq::TSyncedImage<aq::BGR<float>> timg(aq::Shape<2>(1024, 512));
    EXPECT_EQ(timg.rows(), 1024);
    EXPECT_EQ(timg.cols(), 512);
    EXPECT_EQ(timg.dataType(), ct::value(aq::DataFlag::kFLOAT));
    EXPECT_EQ(timg.pixelFormat(), ct::value(aq::PixelFormat::kBGR));
    EXPECT_EQ(timg.pixelSize(), sizeof(float) * 3);
    EXPECT_EQ(timg.channels(), 3);
    EXPECT_EQ(timg.elemSize(), sizeof(float));
    EXPECT_NE(timg.data().get(), nullptr);
}

TEST(synced_image, tsynced_from_image)
{
    aq::SyncedImage img({1024, 512}, aq::PixelFormat::kBGR, aq::DataFlag::kFLOAT);
    EXPECT_EQ(img.rows(), 1024);
    EXPECT_EQ(img.cols(), 512);
    EXPECT_EQ(img.dataType(), ct::value(aq::DataFlag::kFLOAT));
    EXPECT_EQ(img.pixelFormat(), ct::value(aq::PixelFormat::kBGR));
    EXPECT_EQ(img.pixelSize(), sizeof(float) * 3);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_EQ(img.elemSize(), sizeof(float));

    aq::TSyncedImage<aq::BGR<float>> timg(img);
    EXPECT_EQ(timg.rows(), 1024);
    EXPECT_EQ(timg.cols(), 512);
    EXPECT_EQ(timg.dataType(), ct::value(aq::DataFlag::kFLOAT));
    EXPECT_EQ(timg.pixelFormat(), ct::value(aq::PixelFormat::kBGR));
    EXPECT_EQ(timg.pixelSize(), sizeof(float) * 3);
    EXPECT_EQ(timg.channels(), 3);
    EXPECT_EQ(timg.elemSize(), sizeof(float));
    EXPECT_EQ(timg.data().get(), img.data().get());

    EXPECT_THROW(constructTImage<aq::RGB<float>>(img), mo::TExceptionWithCallstack<std::runtime_error>);
    EXPECT_THROW(constructTImage<aq::BGR<uint8_t>>(img), mo::TExceptionWithCallstack<std::runtime_error>);
}

TEST(synced_image, image_from_tsyncedimage)
{
    aq::TSyncedImage<aq::BGR<float>> timg(aq::Shape<2>(1024, 512));
    EXPECT_EQ(timg.rows(), 1024);
    EXPECT_EQ(timg.cols(), 512);
    EXPECT_EQ(timg.dataType(), ct::value(aq::DataFlag::kFLOAT));
    EXPECT_EQ(timg.pixelFormat(), ct::value(aq::PixelFormat::kBGR));
    EXPECT_EQ(timg.pixelSize(), sizeof(float) * 3);
    EXPECT_EQ(timg.channels(), 3);
    EXPECT_EQ(timg.elemSize(), sizeof(float));

    aq::SyncedImage img(timg);
    EXPECT_EQ(img.rows(), 1024);
    EXPECT_EQ(img.cols(), 512);
    EXPECT_EQ(img.dataType(), ct::value(aq::DataFlag::kFLOAT));
    EXPECT_EQ(img.pixelFormat(), ct::value(aq::PixelFormat::kBGR));
    EXPECT_EQ(img.pixelSize(), sizeof(float) * 3);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_EQ(img.elemSize(), sizeof(float));

    EXPECT_EQ(timg.data().get(), img.data().get());
}
