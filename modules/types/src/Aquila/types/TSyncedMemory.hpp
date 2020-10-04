#ifndef AQ_TSYNCED_MEMORY_HPP
#define AQ_TSYNCED_MEMORY_HPP
#include "SyncedMemory.hpp"
#include "TSyncedView.hpp"

namespace aq
{
    template <class T>
    struct TSyncedMemory : virtual SyncedMemory
    {
        template <uint8_t D>
        static TSyncedMemory copyHost(mt::Tensor<const T, D> tensor,
                                      std::shared_ptr<mo::IAsyncStream> stream = mo::IAsyncStream::current())
        {
            return TSyncedMemory(SyncedMemory::copyHost(std::move(tensor), std::move(stream)));
        }

        template <uint8_t D>
        static TSyncedMemory copyDevice(mt::Tensor<const T, D> tensor,
                                        std::shared_ptr<mo::IAsyncStream> stream = mo::IAsyncStream::current())
        {
            return TSyncedMemory(SyncedMemory::copyDevice(std::move(tensor), std::move(stream)));
        }

        TSyncedMemory(size_t elements = 0, std::shared_ptr<mo::IDeviceStream> stream = mo::IDeviceStream::current())
            : SyncedMemory(elements * sizeof(T), sizeof(T), stream)
        {
        }

        template <class... ARGS>
        TSyncedMemory(ARGS&&... args)
            : SyncedMemory(std::forward<ARGS>(args)...)
        {
        }

        size_t size() const
        {
            return SyncedMemory::size() / sizeof(T);
        }

        ct::TArrayView<const T> host(mo::IAsyncStream* stream, bool* sync_required = nullptr) const
        {
            return SyncedMemory::hostAs<T>(stream, sync_required);
        }

        ct::TArrayView<const T> device(mo::IDeviceStream* stream, bool* sync_required = nullptr) const
        {
            return SyncedMemory::deviceAs<T>(stream, sync_required);
        }

        ct::TArrayView<T> mutableHost(mo::IAsyncStream* stream, bool* sync_required = nullptr)
        {
            return SyncedMemory::mutableHostAs<T>(stream, sync_required);
        }

        ct::TArrayView<T> mutableDevice(mo::IDeviceStream* stream, bool* sync_required = nullptr)
        {
            return SyncedMemory::mutableDeviceAs<T>(stream, sync_required);
        }

        TSyncedView<T> syncedView(mo::IDeviceStream* stream)
        {
            return TSyncedView<T>(SyncedMemory::syncedView(stream));
        }

        TSyncedView<const T> syncedView(mo::IDeviceStream* stream) const
        {
            return TSyncedView<const T>(SyncedMemory::syncedView(stream));
        }
    };
} // namespace aq

#endif // AQ_TSYNCED_MEMORY_HPP
