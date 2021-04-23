#ifndef AQ_TSYNCED_MEMORY_HPP
#define AQ_TSYNCED_MEMORY_HPP
#include "SyncedMemory.hpp"
#include "TSyncedView.hpp"

namespace aq
{
    template <class T>
    struct TSyncedMemory : virtual SyncedMemory
    {
        using DType = T;
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

        TSyncedMemory(SyncedMemory&& other)
            : SyncedMemory(std::move(other))
        {
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
        }

        TSyncedMemory(TSyncedMemory&& other)
            : SyncedMemory(std::move(other))
        {
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
        }

        TSyncedMemory(const TSyncedMemory& other)
            : SyncedMemory(other)
        {
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
        }

        TSyncedMemory(const SyncedMemory& other)
            : SyncedMemory(other)
        {
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
        }

        TSyncedMemory& operator=(SyncedMemory&& other)

        {
            SyncedMemory::operator=(std::move(other));
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
            return *this;
        }

        TSyncedMemory& operator=(const SyncedMemory& other)
        {
            SyncedMemory::operator=(other);
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
            return *this;
        }

        TSyncedMemory& operator=(TSyncedMemory&& other)

        {
            SyncedMemory::operator=(std::move(other));
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
            return *this;
        }

        TSyncedMemory& operator=(const TSyncedMemory& other)
        {
            SyncedMemory::operator=(other);
            MO_ASSERT_EQ(this->elemSize(), sizeof(T));
            return *this;
        }

        // size in number of elements
        bool resize(size_t size, std::shared_ptr<mo::IAsyncStream> stream = {})
        {
            return SyncedMemory::resize(size * sizeof(T), sizeof(T), std::move(stream));
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

namespace ct
{
    REFLECT_TEMPLATED_DERIVED(aq::TSyncedMemory, aq::SyncedMemory)
        using DType = typename DataType::DType;
        static TArrayView<const DType> hostDefault(const DataType& obj)
        {
            return obj.host();
        }
        static TArrayView<DType> mutableHostDefault(DataType & obj)
        {
            return obj.mutableHost();
        }
        // This just exists to strip off the default arg
        static void setSize(DataType & obj, size_t sz)
        {
            obj.resize(sz);
        }
        PROPERTY(size, &DataType::size, &ct::ReflectImpl<DataType>::setSize)
        PROPERTY(host, &ct::ReflectImpl<DataType>::hostDefault, &ct::ReflectImpl<DataType>::mutableHostDefault)
    REFLECT_END;
} // namespace ct
#endif // AQ_TSYNCED_MEMORY_HPP
