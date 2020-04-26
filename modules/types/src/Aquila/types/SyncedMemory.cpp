#include "Aquila/types/SyncedMemory.hpp"
#include <MetaObject/logging/logging.hpp>
#include <boost/lexical_cast.hpp>
#include <ct/reflect/print.hpp>

namespace aq
{

    SyncedMemory SyncedMemory::wrapHost(ct::TArrayView<void> data,
                                        size_t elem_size,
                                        std::shared_ptr<void> owning,
                                        std::shared_ptr<mo::IDeviceStream> stream)
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
                                        std::shared_ptr<void> owning,
                                        std::shared_ptr<mo::IDeviceStream> stream)
    {
        SyncedMemory output(data.size(), elem_size, stream);
        output.m_owning = owning;
        output.h_ptr = const_cast<void*>(data.data());
        output.m_state = SyncState::HOST_UPDATED;
        output.h_flags = PointerFlags::CONST | PointerFlags::UNOWNED;
        return output;
    }

    SyncedMemory SyncedMemory::wrapDevice(ct::TArrayView<void> data,
                                          size_t elem_size,
                                          std::shared_ptr<void> owning,
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
                                          std::shared_ptr<void> owning,
                                          std::shared_ptr<mo::IDeviceStream> stream)
    {
        SyncedMemory output(data.size(), elem_size, stream);
        output.m_owning = owning;
        output.d_ptr = const_cast<void*>(data.data());
        output.m_state = SyncState::DEVICE_UPDATED;
        output.d_flags = PointerFlags::CONST | PointerFlags::UNOWNED;
        return output;
    }

    SyncedMemory::SyncedMemory(size_t size, size_t elem_size, std::shared_ptr<mo::IDeviceStream> stream)
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
    }

    SyncedMemory::SyncedMemory(SyncedMemory&& other)
        : m_state(other.m_state)
        , d_ptr(other.d_ptr)
        , h_ptr(other.h_ptr)
        , h_flags(other.h_flags)
        , d_flags(other.d_flags)
        , m_size(other.m_size)
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
            auto alloc = stream->deviceAllocator();
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

    bool SyncedMemory::resize(size_t size_, size_t elem_size, mo::IDeviceStream* stream)
    {
        if (m_size == size_)
            return false;

        void* old_host = h_ptr;
        void* old_device = d_ptr;
        size_t old_size = m_size;
        m_elem_size = elem_size;

        m_size = size_;
        auto stream_ = m_stream.lock();

        if (old_host || old_device)
        {
            if (m_state == SyncState::HOST_UPDATED)
            {
                auto alloc = stream_->hostAllocator();
                MO_ASSERT(alloc != nullptr);
                h_ptr = alloc->allocate(m_size, m_elem_size);
                MO_ASSERT(h_ptr != nullptr);
                MO_ASSERT(old_host != nullptr);
                d_ptr = nullptr;
                stream->hostToHost({h_ptr, m_size}, {old_host, old_size});
                stream->pushWork(
                    [old_host, old_size, alloc]() { alloc->deallocate(ct::ptrCast<uint8_t>(old_host), old_size); });
                if (old_device)
                {
                    auto alloc = stream_->deviceAllocator();
                    MO_ASSERT(alloc);
                    alloc->deallocate(ct::ptrCast<uint8_t>(old_device), old_size);
                }
                return true;
            }
            else
            {
                if (SyncState::DEVICE_UPDATED == m_state || SyncState::SYNCED == m_state)
                {
                    auto alloc = stream_->deviceAllocator();
                    MO_ASSERT(alloc);
                    d_ptr = alloc->allocate(m_size, m_elem_size);
                    MO_ASSERT(d_ptr != nullptr);
                    MO_ASSERT(old_device != nullptr);
                    stream->deviceToDevice({d_ptr, m_size}, {old_device, old_size});
                    stream->pushWork([old_device, old_size, alloc]() {
                        alloc->deallocate(ct::ptrCast<uint8_t>(old_device), old_size);
                    });
                    if (old_host)
                    {
                        auto alloc = stream_->hostAllocator();
                        MO_ASSERT(alloc);
                        alloc->deallocate(ct::ptrCast<uint8_t>(old_host), old_size);
                    }
                    return true;
                }
            }
        }
        return false;
    }

    ct::TArrayView<const void> SyncedMemory::host(mo::IDeviceStream* stream, bool* sync_required) const
    {
        auto stream_ = m_stream.lock();
        if (!h_ptr)
        {
            MO_ASSERT(stream_);
            auto alloc = stream_->hostAllocator();
            h_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            h_flags = PointerFlags::OWNED;
            if (d_ptr)
            {
                m_state = SyncState::DEVICE_UPDATED;
            }
        }

        if (SyncState::DEVICE_UPDATED == m_state)
        {
            if (stream && stream != stream_.get())
            {
                stream->synchronize(stream_.get());
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                stream->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }
            else
            {
                MO_ASSERT_FMT(stream_, "No default stream available");
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                stream_->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }

            m_state = SyncState::SYNCED;
            if (sync_required)
            {
                *sync_required = true;
            }
            else
            {
                // If sync_required not passed in, do a synchronization here
                if (stream && stream != stream_.get())
                {
                    stream->synchronize();
                }
                else
                {
                    MO_ASSERT(stream_);
                    stream_->synchronize();
                }
            }
        }

        return {h_ptr, m_size};
    }

    ct::TArrayView<const void> SyncedMemory::device(mo::IDeviceStream* stream, bool* sync_required) const
    {
        auto stream_ = m_stream.lock();
        if (!d_ptr)
        {
            MO_ASSERT(stream_);
            auto alloc = stream_->deviceAllocator();
            d_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            d_flags = PointerFlags::OWNED;
            if (h_ptr)
            {
                m_state = SyncState::HOST_UPDATED;
            }
        }

        if (SyncState::HOST_UPDATED == m_state)
        {
            auto strm = m_stream.lock();
            if (stream && stream != strm.get())
            {
                stream->synchronize(strm.get());
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                stream->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }
            else
            {
                MO_ASSERT(strm != nullptr);
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                strm->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }

            m_state = SyncState::SYNCED;
            if (sync_required)
            {
                *sync_required = true;
            }
        }

        return {d_ptr, m_size};
    }

    ct::TArrayView<void> SyncedMemory::mutableHost(mo::IDeviceStream* stream, bool* sync_required)
    {
        auto stream_ = m_stream.lock();
        if (!h_ptr)
        {
            MO_ASSERT(stream_);
            auto alloc = stream_->hostAllocator();
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
            MO_ASSERT(stream_);
            auto alloc = stream_->hostAllocator();
            MO_ASSERT(alloc);
            h_ptr = alloc->allocate(m_size, m_elem_size);
            h_flags = PointerFlags::OWNED;
            if (SyncState::HOST_UPDATED & m_state)
            {
                auto strm = m_stream.lock();
                MO_ASSERT(h_ptr != nullptr);
                MO_ASSERT(old_ptr != nullptr);
                strm->hostToHost({h_ptr, m_size}, {old_ptr, m_size});
            }
        }

        if (SyncState::DEVICE_UPDATED == m_state)
        {
            auto strm = m_stream.lock();
            if (stream && stream != strm.get())
            {
                stream->synchronize(strm.get());
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                stream->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }
            else
            {
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                strm->deviceToHost({h_ptr, m_size}, {d_ptr, m_size});
            }

            if (sync_required)
            {
                *sync_required = true;
            }
            else
            {
                // If sync_required not passed in, do a synchronization here
                if (stream && stream != strm.get())
                {
                    stream->synchronize();
                }
                else
                {
                    strm->synchronize();
                }
            }
        }
        m_state = SyncState::HOST_UPDATED;
        return {h_ptr, m_size};
    }

    ct::TArrayView<void> SyncedMemory::mutableDevice(mo::IDeviceStream* stream, bool* sync_required)
    {
        auto stream_ = m_stream.lock();
        if (!d_ptr)
        {
            MO_ASSERT(stream_);
            auto alloc = stream_->deviceAllocator();
            d_ptr = static_cast<void*>(alloc->allocate(m_size, m_elem_size));
            d_flags = PointerFlags::OWNED;
            if (h_ptr)
            {
                m_state = SyncState::HOST_UPDATED;
            }
        }

        // Copy on write semantics, we now allocate a new h_ptr and use that instead
        if (d_flags & PointerFlags::CONST)
        {
            const auto old_ptr = d_ptr;
            MO_ASSERT(stream_);
            auto alloc = stream_->deviceAllocator();
            MO_ASSERT(alloc);
            d_ptr = alloc->allocate(m_size, m_elem_size);
            d_flags = PointerFlags::OWNED;
            if (SyncState::HOST_UPDATED & m_state)
            {
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(old_ptr != nullptr);
                stream_->deviceToDevice({d_ptr, m_size}, {old_ptr, m_size});
            }
        }

        if (SyncState::HOST_UPDATED == m_state)
        {
            auto strm = m_stream.lock();
            MO_ASSERT(strm);
            if (stream && stream != strm.get())
            {
                // setup an event to sync between m_stream and stream
                stream->synchronize(strm.get());
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                stream->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }
            else
            {
                MO_ASSERT(d_ptr != nullptr);
                MO_ASSERT(h_ptr != nullptr);
                strm->hostToDevice({d_ptr, m_size}, {h_ptr, m_size});
            }

            if (sync_required)
            {
                *sync_required = true;
            }
        }
        m_state = SyncState::DEVICE_UPDATED;
        return {d_ptr, m_size};
    }

    SyncedMemory::SyncState SyncedMemory::state() const
    {
        return m_state;
    }

    std::weak_ptr<mo::IDeviceStream> SyncedMemory::stream() const
    {
        return m_stream;
    }

    void SyncedMemory::setStream(std::shared_ptr<mo::IDeviceStream> stream)
    {
        auto strm = m_stream.lock();
        if (strm.get() == stream.get())
        {
            return;
        }
        MO_ASSERT((h_ptr == nullptr) && (d_ptr == nullptr));
        m_stream = std::move(stream);
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
