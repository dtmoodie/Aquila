#include <Aquila/types/TSyncedMemory.hpp>
#include <MetaObject/cuda/AsyncStream.hpp>
#include <ce/shared_ptr.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

TEST(synced_memory, host_allocation)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();

    aq::SyncedMemory memory(10 * sizeof(float), sizeof(float), stream);
    EXPECT_EQ(memory.size(), 10 * sizeof(float));

    bool sync = false;
    {
        auto host = memory.mutableHost(nullptr, &sync);
        EXPECT_FALSE(sync);
        EXPECT_EQ(host.size(), 10 * sizeof(float));
        ct::TArrayView<float> view(host);
        EXPECT_EQ(view.size(), 10);
        for (ssize_t i = 0; i < view.size(); ++i)
        {
            view[i] = i;
        }
    }

    {
        auto host = memory.host(nullptr, &sync);
        EXPECT_FALSE(sync);
        ct::TArrayView<const float> view(host);
        for (ssize_t i = 0; i < view.size(); ++i)
        {
            EXPECT_EQ(view[i], i);
        }
    }
}

TEST(synced_memory, wrap)
{
    std::shared_ptr<std::vector<float>> data = std::make_shared<std::vector<float>>(1000);

    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    auto wrapped = aq::SyncedMemory::wrapHost(ct::TArrayView<float>(data->data(), data->size()), data, stream);

    for (size_t i = 0; i < 1000; ++i)
    {
        (*data)[i] = i;
    }

    bool sync = false;
    auto view = wrapped.hostAs<float>(nullptr, &sync);

    EXPECT_FALSE(sync);
    EXPECT_EQ(view.size(), 1000);
    for (size_t i = 0; i < view.size(); ++i)
    {
        EXPECT_EQ(view[i], i);
    }
}

TEST(synced_memory, wrap_const)
{
    std::shared_ptr<std::vector<float>> data = std::make_shared<std::vector<float>>(1000);

    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    auto wrapped = aq::SyncedMemory::template wrapHost<float>(
        ct::TArrayView<const float>(data->data(), data->size()), data, stream);

    for (size_t i = 0; i < 1000; ++i)
    {
        (*data)[i] = i;
    }

    bool sync = false;
    auto view = wrapped.hostAs<float>(nullptr, &sync);
    EXPECT_EQ(view.data(), data->data());

    EXPECT_FALSE(sync);
    EXPECT_EQ(view.size(), 1000);
    for (size_t i = 0; i < view.size(); ++i)
    {
        EXPECT_EQ(view[i], i);
    }

    auto mutable_view = wrapped.mutableHostAs<float>(nullptr, &sync);
    EXPECT_FALSE(sync);
    EXPECT_NE(mutable_view.data(), data->data());
}

__global__ void setKernel(ct::TArrayView<float> data)
{
    const auto N = data.size();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        data[tid] = tid;
    }
}

TEST(synced_memory, device)
{
    std::vector<size_t> szs = {1000, 100000, 10000000};
    for (auto sz : szs)
    {
        auto stream = std::make_shared<mo::cuda::AsyncStream>();
        auto allocator = mo::DeviceAllocator::getDefault();
        EXPECT_NE(allocator, nullptr);

        aq::SyncedMemory memory(sz * sizeof(float), sizeof(float), stream);
        bool sync = false;
        auto view = memory.template mutableHostAs<float>(nullptr, &sync);
        EXPECT_FALSE(sync);
        for (auto& itr : view)
        {
            itr = 0;
        }

        auto device_view = memory.template mutableDeviceAs<float>(nullptr, &sync);
        EXPECT_TRUE(sync);
        auto blocks = sz / 256 + 1;
        setKernel<<<blocks, 256, 0, *stream>>>(device_view);

        auto host_view = memory.template hostAs<float>(nullptr, &sync);
        EXPECT_TRUE(sync);
        stream->synchronize();

        for (size_t i = 0; i < host_view.size(); ++i)
        {
            EXPECT_EQ(host_view[i], i);
        }
    }
}

TEST(synced_memory, typed_device)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    auto allocator = mo::DeviceAllocator::getDefault();
    EXPECT_NE(allocator, nullptr);

    aq::TSyncedMemory<float> memory(100, stream);
    bool sync = false;
    auto view = memory.mutableHost(nullptr, &sync);
    EXPECT_FALSE(sync);
    for (auto& itr : view)
    {
        itr = 0;
    }

    auto device_view = memory.mutableDevice(nullptr, &sync);
    EXPECT_TRUE(sync);

    setKernel<<<4, 256>>>(device_view);
    cudaDeviceSynchronize();

    auto host_view = memory.template hostAs<float>(nullptr, &sync);
    EXPECT_TRUE(sync);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < host_view.size(); ++i)
    {
        EXPECT_EQ(host_view[i], i);
    }
}

TEST(synced_memory, handle)
{
    auto stream = std::make_shared<mo::cuda::AsyncStream>();
    auto handle = ce::shared_ptr<aq::SyncedMemory>::create(100, 4, stream);
    bool sync = false;
    auto view = handle->mutableHost(nullptr, &sync);
    EXPECT_FALSE(sync);
    ce::shared_ptr<const aq::SyncedMemory> other(handle);
    auto other_view = other->host();
    EXPECT_EQ(other_view.data(), view.data());

    // Copy on write handle
    ce::shared_ptr<aq::SyncedMemory> copy_on_write_handle(other);
    view = copy_on_write_handle->mutableHost();
    EXPECT_NE(view.data(), other_view.data());
}