#include "Aquila/types/SyncedMemory.hpp"
#include <MetaObject/logging/logging.hpp>
#include <boost/lexical_cast.hpp>
#include <ct/reflect/print.hpp>

namespace aq
{
    SyncedMemory
    SyncedMemory::copyHost(ct::TArrayView<const void> data, size_t elem_size, std::shared_ptr<mo::IAsyncStream> stream)
    {

        SyncedMemory output(data.size(), elem_size, stream);
        auto dst = output.mutableHost(stream.get());
        std::memcpy(dst.data(), data.data(), data.size());
        return output;
    }

    SyncedMemory SyncedMemory::wrapHost(ct::TArrayView<void> data,
                                        size_t elem_size,
                                        std::shared_ptr<const void> owning,
                                        std::shared_ptr<mo::IAsyncStream> stream)
    {
        SyncedMemory output(data.size(), elem_size, stream);
        output.m_owning = owning;
        output.h_ptr = data.data();
        output.m_state = SyncState::HOST_UPDATED;
        output.h_flags = PointerFlags::UNOWNED;
        return output;
    }

    SyncedMemory SyncedMemory::wrapHost(ct::TArrayView<const void> data,
                                        size_t elem_size,
                                        std::shared_ptr<const void> owning,
                                        std::shared_ptr<mo::IAsyncStream> stream)
    {
        SyncedMemory output(data.size(), elem_size, stream);
        output.m_owning = owning;
        output.h_ptr = const_cast<void*>(data.data());
        output.m_state = SyncState::HOST_UPDATED;
        output.h_flags = PointerFlags::CONST | PointerFlags::UNOWNED;
        return output;
    }

    SyncedMemory SyncedMemory::copyDevice(ct::TArrayView<const void> data,
                                          size_t elem_size,
                                          std::shared_ptr<mo::IDeviceStream> stream)
    {

        SyncedMemory output(data.size(), elem_size, stream);
        auto dst = output.mutableDevice(stream.get());
        stream->deviceToDevice(dst, data);
        return output;
    }

    SyncedMemory SyncedMemory::wrapDevice(ct::TArrayView<void> data,
                                          size_t elem_size,
                                          std::shared_ptr<const void> owning,
                                          std::shared_ptr<mo::IDeviceStream> stream)
    {
        SyncedMemory output(data.size(), elem_size, stream);
        output.m_owning = owning;
        output.h_ptr = data.data();
        output.m_state = SyncState::DEVICE_UPDATED;
        output.d_flags = PointerFlags::UNOWNED;
        return output;
    }

    SyncedMemory SyncedMemory::wrapDevice(ct::TArrayView<const void> data,
                                          size_t elem_size,
                                          std::shared_ptr<const void> owning,
                                          std::shared_ptr<mo::IDeviceStream> stream)
    {
        SyncedMemory output(data.size(), elem_size, stream);
        output.m_owning = owning;
        output.d_ptr = const_cast<void*>(data.data());
        output.m_state = SyncState::DEVICE_UPDATED;
        output.d_flags = PointerFlags::CONST | PointerFlags::UNOWNED;
        return output;
    }

    SyncedMemory::SyncedMemory(size_t size, size_t elem_size, std::shared_ptr<mo::IAsyncStream> stream)
        : m_size(size)
        , m_elem_size(elem_size)
        , m_stream(stream)
    {
    }

    SyncedMemory::SyncedMemory(const SyncedMemory& other)
    {
        if (other.m_state == SyncState::HOST_UPDATED || other.d_ptr == nullptr)
        {
            h_ptr = other.h_ptr;
            h_flags = PointerFlags::CONST;
            m_state = SyncState::HOST_UPDATED;
        }
        else
        {
            d_ptr = other.d_ptr;
            d_flags = PointerFlags::CONST;
            m_state = SyncState::DEVICE_UPDATED;
        }
        m_elem_size = other.m_elem_size;
        m_size = other.m_size;
        m_stream = other.m_stream;
        m_owning = other.m_owning;
        m_capacity = other.m_capacity;
    }

    SyncedMemory::SyncedMemory(SyncedMemory&& other)
        : m_state(other.m_state)
        , d_ptr(other.d_ptr)
        , h_ptr(other.h_ptr)
        , h_flags(other.h_flags)
        , d_flags(other.d_flags)
        , m_size(other.m_size)
        , m_capacity(other.m_capacity)
        , m_elem_size(other.m_elem_size)
        , m_stream(std::move(other.m_stream))
        , m_owning(std::move(other.m_owning))
    {
        other.d_ptr = nullptr;
        other.h_ptr = nullptr;
    }

    SyncedMemory::~SyncedMemory()
    {
        auto stream = m_stream.lock();
        if (stream == nullptr)
        {
            return;
        }
        if (h_ptr && (h_flags & PointerFlags::OWNED))
        {
            auto alloc = stream->hostAllocator();
            if (alloc == nullptr)
            {
                MO_LOG(error, "Allocator has already been destroyed, cannot cleanup host memory");
            }
            else
            {
                alloc->deallocate(h_ptr, m_size);
            }
        }
        if (d_ptr && (d_flags & PointerFlags::OWNED))
        {
            mo::IDeviceStream* device_stream = stream->getDeviceStream();
            MO_ASSERT(device_stream != nullptr);
            auto alloc = device_stream->deviceAllocator();
            if (alloc == nullptr)
            {
                MO_LOG(error, "Allocator has already been destroyed, cannot cleanup device memory");
            }
            else
            {
                alloc->deallocate(d_ptr, m_size);
            }
        }
    }

    SyncedMemory& SyncedMemory::operator=(const SyncedMemory& other)
    {
        if (other.m_state == SyncState::HOST_UPDATED)
        {
            h_ptr = other.h_ptr;
            h_flags = PointerFlags::CONST;
            m_state = SyncState::HOST_UPDATED;
        }
        else
        {
            d_ptr = other.d_ptr;
            d_flags = PointerFlags::CONST;
            m_state = SyncState::DEVICE_UPDATED;
        }
        m_elem_size = other.m_elem_size;
        m_size = other.m_size;
        m_stream = other.m_stream;
        m_owning = other.m_owning;
        return *this;
    }

    SyncedMemory& SyncedMemory::operator=(SyncedMemory&& other)
    {
        m_state = other.m_state;

        d_ptr = other.d_ptr;
        h_ptr = other.h_ptr;

        h_flags = other.h_flags;
        d_flags = other.d_flags;

        m_size = other.m_size;
        m_capacity = other.m_capacity;
        m_elem_size = other.m_elem_size;

        m_stream = std::move(other.m_stream);
        m_owning = std::move(other.m_owning);

        other.d_ptr = nullptr;
        other.h_ptr = nullptr;
        return *this;
    }

    size_t SyncedMemory::size() const
    {
        return m_size;
    }

    size_t SyncedMemory::elemSize() const
    {
        return m_elem_size;
    }

    bool SyncedMemory::empty() const
    {
        return m_size == 0;
    }

    bool SyncedMemory::resize(size_t size_, size_t elem_size, std::shared_ptr<mo::IAsyncStream> dst_stream)
    {
        if (m_size == size_)
        {
            return true;
        }

        if (size_ < m_capacity)
        {
            m_size = size_;
            return true;
        }

        void* old_host = h_ptr;
        void* old_device = d_ptr;
        size_t old_size = m_size;
        m_elem_size = elem_size;

        m_size = size_;
        m_capacity = m_size;
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        MO_ASSERT(src_stream != nullptr);
        mo::IDeviceStream* src_device_stream = src_stream->getDeviceStream();

        // TODO use dst_stream

        if (old_host || old_device)
        {
            if (m_state == SyncState::HOST_UPDATED)
            {
                auto alloc = src_stream->hostAllocator();
                if (alloc == nullptr)
                {
                    return false;
                }
                h_ptr = alloc->allocate(m_size, m_elem_size);
                if (h_ptr == nullptr)
                {
                    // failed to allocate enough memory
                    return false;
                }
                // program error, state should not be HOST_UPDATED if we haven't even allocated data
                MO_ASSERT(old_host != nullptr);
                d_ptr = nullptr;
                const size_t copy_size = std::min(old_size, m_size);
                if (copy_size)
                {
                    src_stream->hostToHost({h_ptr, copy_size}, {old_host, copy_size});
                    src_stream->pushWork(
                        [old_host, old_size, alloc](mo::IAsyncStream*) { alloc->deallocate(ct::ptrCast<uint8_t>(old_host), old_size); });
                }

                if (old_device)
                {
                    mo::IDeviceStream* device_stream = src_stream->getDeviceStream();
                    MO_ASSERT(device_stream != nullptr);
                    auto alloc = device_stream->deviceAllocator();
                    MO_ASSERT(alloc);
                    alloc->deallocate(ct::ptrCast<uint8_t>(old_device), old_size);
                }
                return true;
            }
            else
            {
                if (SyncState::DEVICE_UPDATED == m_state || SyncState::SYNCED == m_state)
                {
                    mo::IDeviceStream* device_stream = src_stream->getDeviceStream();
                    MO_ASSERT(device_stream != nullptr);
                    auto alloc = device_stream->deviceAllocator();
                    MO_ASSERT(alloc);
                    d_ptr = alloc->allocate(m_size, m_elem_size);
                    MO_ASSERT(d_ptr != nullptr);
                    MO_ASSERT(old_device != nullptr);
                    const size_t copy_size = std::min(old_size, m_size);
                    if (copy_size)
                    {
                        src_device_stream->deviceToDevice({d_ptr, copy_size}, {old_device, copy_size});
                        src_device_stream->pushWork([old_device, old_size, alloc](mo::IAsyncStream*) {
                            alloc->deallocate(ct::ptrCast<uint8_t>(old_device), old_size);
                        });
                    }

                    if (old_host)
                    {
                        auto alloc = src_stream->hostAllocator();
                        MO_ASSERT(alloc);
                        alloc->deallocate(ct::ptrCast<uint8_t>(old_host), old_size);
                    }
                    return true;
                }
            }
        }
        return false;
    }

    ct::TArrayView<const void> SyncedMemory::host(mo::IAsyncStream* dst_stream, bool* sync_required) const
    {
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        if (!h_ptr)
        {
            MO_ASSERT(src_stream);
            auto host_allocator = src_stream->hostAllocator();
            MO_ASSERT(host_allocator);
            h_ptr = static_cast<void*>(host_allocator->allocate(m_size, m_elem_size));
            h_flags = PointerFlags::OWNED;
            if (d_ptr)
            {
                m_state = SyncState::DEVICE_UPDATED;
            }
        }

        if (SyncState::DEVICE_UPDATED == m_state)
        {
            mo::IDeviceStream* src_device_stream = src_stream->getDeviceStream();

            mo::IDeviceStream* dst_device_stream = src_device_stream;
            if (dst_stream)
            {
                dst_device_stream = dst_stream->getDeviceStream();
            }
            if (dst_device_stream && src_device_stream != dst_device_stream)
            {
                src_device_stream->synchronize(dst_device_stream);
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                dst_device_stream->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }
            else
            {
                MO_ASSERT_FMT(src_device_stream, "No default stream available");
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                src_device_stream->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }

            m_state = SyncState::SYNCED;
            if (sync_required)
            {
                *sync_required = true;
            }
            else
            {
                // If sync_required not passed in, do a synchronization here
                if (dst_stream && dst_stream != src_device_stream)
                {
                    dst_stream->synchronize();
                }
                else
                {
                    MO_ASSERT(src_device_stream);
                    src_device_stream->synchronize();
                }
            }
        }

        return {h_ptr, m_size};
    }

    ct::TArrayView<const void> SyncedMemory::device(mo::IDeviceStream* dst_stream, bool* sync_required) const
    {
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        MO_ASSERT(src_stream != nullptr);
        mo::IDeviceStream* src_device_stream = src_stream->getDeviceStream();
        MO_ASSERT(src_device_stream != nullptr);

        if (!d_ptr)
        {
            auto device_allocator = src_device_stream->deviceAllocator();
            MO_ASSERT(device_allocator);
            d_ptr = static_cast<void*>(device_allocator->allocate(m_size, m_elem_size));
            d_flags = PointerFlags::OWNED;
            if (h_ptr)
            {
                m_state = SyncState::HOST_UPDATED;
            }
        }

        if (SyncState::HOST_UPDATED == m_state)
        {
            if (dst_stream && dst_stream != src_device_stream)
            {
                dst_stream->synchronize(src_device_stream);
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                dst_stream->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }
            else
            {
                MO_ASSERT(src_device_stream != nullptr);
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                src_device_stream->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }

            m_state = SyncState::SYNCED;
            if (sync_required)
            {
                *sync_required = true;
            }
        }

        return {d_ptr, m_size};
    }

    ct::TArrayView<void> SyncedMemory::mutableHost(mo::IAsyncStream* dst_stream, bool* sync_required)
    {
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        if (!h_ptr)
        {
            MO_ASSERT(src_stream);
            std::shared_ptr<mo::Allocator> alloc = src_stream->hostAllocator();
            h_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            h_flags = PointerFlags::OWNED;
            if (d_ptr)
            {
                m_state = SyncState::DEVICE_UPDATED;
            }
        }

        // Copy on write semantics, we now allocate a new h_ptr and use that instead
        if (h_flags & PointerFlags::CONST)
        {
            const auto old_ptr = h_ptr;
            MO_ASSERT(src_stream);
            auto alloc = src_stream->hostAllocator();
            MO_ASSERT(alloc);
            h_ptr = alloc->allocate(m_size, m_elem_size);
            h_flags = PointerFlags::OWNED;
            if (SyncState::HOST_UPDATED & m_state)
            {
                MO_ASSERT(h_ptr != nullptr);
                MO_ASSERT(old_ptr != nullptr);
                src_stream->hostToHost({h_ptr, m_size}, {old_ptr, m_size});
            }
        }

        if (SyncState::DEVICE_UPDATED == m_state)
        {
            mo::IDeviceStream* src_device_stream = src_stream->getDeviceStream();
            MO_ASSERT(src_device_stream != nullptr);
            mo::IDeviceStream* dst_device_stream = dst_stream->getDeviceStream();
            MO_ASSERT(dst_device_stream != nullptr);
            if (dst_stream && dst_stream != src_stream.get())
            {
                dst_device_stream->synchronize(src_device_stream);
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                dst_device_stream->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }
            else
            {
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                src_device_stream->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }

            if (sync_required)
            {
                *sync_required = true;
            }
            else
            {
                // If sync_required not passed in, do a synchronization here
                if (src_stream && src_stream.get() != dst_stream)
                {
                    dst_stream->synchronize();
                }
                else
                {
                    src_stream->synchronize();
                }
            }
        }
        m_state = SyncState::HOST_UPDATED;
        return {h_ptr, m_size};
    }

    ct::TArrayView<void> SyncedMemory::mutableHostOnlyWrite(mo::IAsyncStream* dst_stream)
    {
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        if (!h_ptr)
        {
            MO_ASSERT(src_stream);
            std::shared_ptr<mo::Allocator> alloc = src_stream->hostAllocator();
            h_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            h_flags = PointerFlags::OWNED;
        }

        // Copy on write semantics, we now allocate a new h_ptr and use that instead
        if (h_flags & PointerFlags::CONST)
        {
            MO_ASSERT(src_stream);
            auto alloc = src_stream->hostAllocator();
            MO_ASSERT(alloc);
            h_ptr = alloc->allocate(m_size, m_elem_size);
            h_flags = PointerFlags::OWNED;
        }

        m_state = SyncState::HOST_UPDATED;
        return {h_ptr, m_size};
    }

    ct::TArrayView<void> SyncedMemory::mutableDevice(mo::IDeviceStream* stream, bool* sync_required)
    {
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        MO_ASSERT(src_stream != nullptr);
        mo::IDeviceStream* src_device_stream = src_stream->getDeviceStream();
        MO_ASSERT(src_device_stream != nullptr);
        if (!d_ptr)
        {
            auto alloc = src_device_stream->deviceAllocator();
            d_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            d_flags = PointerFlags::OWNED;
            if (h_ptr)
            {
                m_state = SyncState::HOST_UPDATED;
            }
        }

        // Copy on write semantics, we now allocate a new d_ptr and use that instead
        if (d_flags & PointerFlags::CONST)
        {
            const auto old_ptr = d_ptr;
            auto alloc = src_device_stream->deviceAllocator();
            MO_ASSERT(alloc);
            d_ptr = alloc->allocate(m_size, m_elem_size);
            d_flags = PointerFlags::OWNED;
            if (SyncState::HOST_UPDATED & m_state)
            {
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(old_ptr != nullptr);
                src_device_stream->deviceToDevice({d_ptr, m_size}, {old_ptr, m_size});
            }
        }

        if (SyncState::HOST_UPDATED == m_state)
        {

            if (stream && stream != src_device_stream)
            {
                // setup an event to sync between m_stream and stream
                stream->synchronize(src_device_stream);
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                stream->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }
            else
            {
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                src_device_stream->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }

            if (sync_required)
            {
                *sync_required = true;
            }
        }
        m_state = SyncState::DEVICE_UPDATED;
        return {d_ptr, m_size};
    }

    ct::TArrayView<void> SyncedMemory::mutableDeviceOnlyWrite(mo::IAsyncStream* dst_stream)
    {
        std::shared_ptr<mo::IAsyncStream> src_stream = m_stream.lock();
        MO_ASSERT(src_stream != nullptr);
        mo::IDeviceStream* src_device_stream = src_stream->getDeviceStream();
        MO_ASSERT(src_device_stream != nullptr);
        if (!d_ptr)
        {
            auto alloc = src_device_stream->deviceAllocator();
            d_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            d_flags = PointerFlags::OWNED;
            if (h_ptr)
            {
                m_state = SyncState::HOST_UPDATED;
            }
        }

        // Copy on write semantics, we now allocate a new d_ptr and use that instead
        if (d_flags & PointerFlags::CONST)
        {
            auto alloc = src_device_stream->deviceAllocator();
            MO_ASSERT(alloc);
            d_ptr = alloc->allocate(m_size, m_elem_size);
            d_flags = PointerFlags::OWNED;
        }
        m_state = SyncState::DEVICE_UPDATED;
        return {d_ptr, m_size};
    }

    SyncedMemory::SyncState SyncedMemory::state() const
    {
        return m_state;
    }

    std::weak_ptr<mo::IAsyncStream> SyncedMemory::getStream() const
    {
        return m_stream;
    }

    void SyncedMemory::setStream(std::shared_ptr<mo::IAsyncStream> stream)
    {
        auto strm = m_stream.lock();
        if (strm.get() == stream.get())
        {
            return;
        }
        // TODO copy?
        MO_ASSERT((h_ptr == nullptr) && (d_ptr == nullptr));
        m_stream = std::move(stream);
    }

    void SyncedMemory::setOwning(std::shared_ptr<const void> owning)
    {
        m_owning = std::move(owning);
    }

    bool SyncedMemory::operator==(const SyncedMemory& other) const
    {
        if ((m_size == other.m_size) && (m_elem_size == other.m_elem_size))
        {
            if ((h_ptr == other.h_ptr) && (d_ptr == other.d_ptr))
            {
                return true;
            }
        }
        return false;
    }

    std::ostream& operator<<(std::ostream& os, const SyncedMemory& memory)
    {
        os << memory.state() << " " << memory.size();
        return os;
    }

    SyncedView SyncedMemory::syncedView(mo::IDeviceStream*)
    {
        // TODO do the sync
        return SyncedView(h_ptr, d_ptr, m_size);
    }

    ConstSyncedView SyncedMemory::syncedView(mo::IDeviceStream*) const
    {
        // TODO do the sync
        return ConstSyncedView(h_ptr, d_ptr, m_size);
    }
} // namespace aq

namespace ct
{
    ct::TArrayView<const void> hostDefault(const aq::SyncedMemory& mem)
    {
        return mem.host();
    }

    ct::TArrayView<void> mutableHostDefault(aq::SyncedMemory& mem)
    {
        return mem.mutableHost();
    }

    TArrayView<const void> makeArrayView(ce::shared_ptr<const aq::SyncedMemory> mem, size_t sz)
    {
        if (mem)
        {
            MO_ASSERT_EQ(mem->size(), sz);
            return mem->host();
        }
        return {};
    }

    TArrayView<void> makeArrayView(ce::shared_ptr<aq::SyncedMemory> mem, size_t sz)
    {
        if (mem)
        {
            MO_ASSERT_EQ(mem->size(), sz);
            return mem->mutableHost();
        }
        return {};
    }
} // namespace ct
