#ifndef AQ_TYPES_SYNCED_MEMORY_HPP
#define AQ_TYPES_SYNCED_MEMORY_HPP
#include "SyncedView.hpp"
#include "export.hpp"
#include "shared_ptr.hpp"

#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>

#include <ct/enum.hpp>
#include <ct/types/TArrayView.hpp>

#include <cstdint>
#include <ostream>

namespace aq
{
    struct AQUILA_EXPORTS SyncedMemory
    {
        ENUM_BEGIN(SyncState, uint8_t)
            ENUM_VALUE(SYNCED, 0)
            ENUM_VALUE(HOST_UPDATED, 1)
            ENUM_VALUE(DEVICE_UPDATED, 2)
        ENUM_END;

        ENUM_BEGIN(PointerFlags, uint8_t)
            ENUM_VALUE(OWNED, 0)
            ENUM_VALUE(UNOWNED, 1)
            ENUM_VALUE(CONST, 2)
        ENUM_END;

        static SyncedMemory wrapHost(ct::TArrayView<void>,
                                     size_t elem_size,
                                     std::shared_ptr<void> owning,
                                     std::shared_ptr<mo::IAsyncStream> stream = mo::IAsyncStream::current());
        static SyncedMemory wrapHost(ct::TArrayView<const void>,
                                     size_t elem_size,
                                     std::shared_ptr<void> owning,
                                     std::shared_ptr<mo::IAsyncStream> stream = mo::IAsyncStream::current());

        static SyncedMemory wrapDevice(ct::TArrayView<void>,
                                       size_t elem_size,
                                       std::shared_ptr<void> owning,
                                       std::shared_ptr<mo::IDeviceStream> stream = mo::IDeviceStream::current());
        static SyncedMemory wrapDevice(ct::TArrayView<const void>,
                                       size_t elem_size,
                                       std::shared_ptr<void> owning,
                                       std::shared_ptr<mo::IDeviceStream> stream = mo::IDeviceStream::current());

        template <class T>
        static SyncedMemory wrapHost(ct::TArrayView<T>,
                                     std::shared_ptr<void> owning,
                                     std::shared_ptr<mo::IAsyncStream> stream = mo::IAsyncStream::current());
        template <class T>
        static SyncedMemory wrapHost(ct::TArrayView<const T>,
                                     std::shared_ptr<void> owning,
                                     std::shared_ptr<mo::IAsyncStream> stream = mo::IAsyncStream::current());

        template <class T>
        static SyncedMemory wrapDevice(ct::TArrayView<T>,
                                       std::shared_ptr<void> owning,
                                       std::shared_ptr<mo::IDeviceStream> stream = mo::IDeviceStream::current());
        template <class T>
        static SyncedMemory wrapDevice(ct::TArrayView<const T>,
                                       std::shared_ptr<void> owning,
                                       std::shared_ptr<mo::IDeviceStream> stream = mo::IDeviceStream::current());

        SyncedMemory(size_t size = 0,
                     size_t elem_size = 1,
                     std::shared_ptr<mo::IAsyncStream> stream = mo::IDeviceStream::current());

        SyncedMemory(const SyncedMemory& other);
        SyncedMemory(SyncedMemory&& other);
        ~SyncedMemory();

        SyncedMemory& operator=(const SyncedMemory& other);
        SyncedMemory& operator=(SyncedMemory&& other);

        size_t size() const;
        bool resize(size_t size, size_t elem_size = 1, std::shared_ptr<mo::IAsyncStream> = {});

        ct::TArrayView<const void> host(mo::IAsyncStream* = nullptr, bool* sync_required = nullptr) const;
        ct::TArrayView<const void> device(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr) const;

        ct::TArrayView<void> mutableHost(mo::IAsyncStream* = nullptr, bool* sync_required = nullptr);
        ct::TArrayView<void> mutableDevice(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr);

        SyncState state() const;

        bool empty() const;

        template <class T>
        ct::TArrayView<const T> hostAs(mo::IAsyncStream* = nullptr, bool* sync_required = nullptr) const;
        template <class T>
        ct::TArrayView<const T> deviceAs(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr) const;

        template <class T>
        ct::TArrayView<T> mutableHostAs(mo::IAsyncStream* = nullptr, bool* sync_required = nullptr);
        template <class T>
        ct::TArrayView<T> mutableDeviceAs(mo::IDeviceStream* = nullptr, bool* sync_required = nullptr);

        std::weak_ptr<mo::IAsyncStream> getStream() const;
        void setStream(std::shared_ptr<mo::IAsyncStream>);

        bool operator==(const SyncedMemory& other) const;

        // SynchronizesData and returns a view that is updated on both the CPU and GPU
        SyncedView syncedView(mo::IDeviceStream* = nullptr);
        ConstSyncedView syncedView(mo::IDeviceStream* = nullptr) const;

      private:
        mutable SyncState m_state;
        mutable void* d_ptr = nullptr;
        mutable void* h_ptr = nullptr;
        mutable PointerFlags h_flags;
        mutable PointerFlags d_flags;
        // number of elements
        size_t m_size;
        // element size in bytes
        size_t m_elem_size = 0;
        // max number of elements
        size_t m_capacity;
        std::weak_ptr<mo::IAsyncStream> m_stream;
        std::shared_ptr<void> m_owning;
    }; // namespace aq

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///    IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////////////////////////////
    template <class T>
    ct::TArrayView<const T> SyncedMemory::hostAs(mo::IAsyncStream* stream, bool* sync_required) const
    {
        MO_ASSERT_EQ(m_elem_size, sizeof(T));
        return host(stream, sync_required);
    }

    template <class T>
    ct::TArrayView<const T> SyncedMemory::deviceAs(mo::IDeviceStream* stream, bool* sync_required) const
    {
        MO_ASSERT_EQ(m_elem_size, sizeof(T));
        return device(stream, sync_required);
    }

    template <class T>
    ct::TArrayView<T> SyncedMemory::mutableHostAs(mo::IAsyncStream* stream, bool* sync_required)
    {
        MO_ASSERT_EQ(m_elem_size, sizeof(T));
        return mutableHost(stream, sync_required);
    }

    template <class T>
    ct::TArrayView<T> SyncedMemory::mutableDeviceAs(mo::IDeviceStream* stream, bool* sync_required)
    {
        MO_ASSERT_EQ(m_elem_size, sizeof(T));
        return mutableDevice(stream, sync_required);
    }

    template <class T>
    SyncedMemory SyncedMemory::wrapHost(ct::TArrayView<T> data,
                                        std::shared_ptr<void> owning,
                                        std::shared_ptr<mo::IAsyncStream> stream)
    {
        return wrapHost(ct::TArrayView<void>(std::move(data)), sizeof(T), owning, stream);
    }

    template <class T>
    SyncedMemory SyncedMemory::wrapHost(ct::TArrayView<const T> data,
                                        std::shared_ptr<void> owning,
                                        std::shared_ptr<mo::IAsyncStream> stream)
    {
        return wrapHost(ct::TArrayView<const void>(std::move(data)), sizeof(T), owning, stream);
    }

    template <class T>
    SyncedMemory SyncedMemory::wrapDevice(ct::TArrayView<T> data,
                                          std::shared_ptr<void> owning,
                                          std::shared_ptr<mo::IDeviceStream> stream)
    {
        return wrapDevice(data, sizeof(T), owning, stream);
    }

    template <class T>
    SyncedMemory SyncedMemory::wrapDevice(ct::TArrayView<const T> data,
                                          std::shared_ptr<void> owning,
                                          std::shared_ptr<mo::IDeviceStream> stream)
    {
        return wrapDevice(data, sizeof(T), owning, stream);
    }

    std::ostream& operator<<(std::ostream& os, const SyncedMemory& memory);
} // namespace aq

#ifdef CT_REFLECT_CEREALIZE_HPP
#error "Must include this file before ct/reflect/cerealize.hpp since this adds additional overloads for makeArrayView"
#endif

namespace ct
{
    AQUILA_EXPORTS TArrayView<const void> hostDefault(const aq::SyncedMemory&);
    AQUILA_EXPORTS TArrayView<void> mutableHostDefault(aq::SyncedMemory&);

    REFLECT_BEGIN(aq::SyncedMemory)
        static void setSize(aq::SyncedMemory & obj, size_t sz)
        {
            obj.resize(sz);
        }
        PROPERTY(size, &aq::SyncedMemory::size, &ct::ReflectImpl<DataType>::setSize)
        PROPERTY(host, &hostDefault, &mutableHostDefault)
    REFLECT_END;

    AQUILA_EXPORTS TArrayView<const void> makeArrayView(ce::shared_ptr<const aq::SyncedMemory>, size_t);
    AQUILA_EXPORTS TArrayView<void> makeArrayView(ce::shared_ptr<aq::SyncedMemory>, size_t);
} // namespace ct

#endif
