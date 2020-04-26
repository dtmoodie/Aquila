#ifndef AQUILA_SYNCED_VIEW_HPP
#define AQUILA_SYNCED_VIEW_HPP
#include <Aquila/detail/export.hpp>

#include <MetaObject/detail/defines.hpp>

#include <ct/reflect.hpp>
#include <ct/types/TArrayView.hpp>

#include <cstdint>

namespace aq
{
    struct ConstSyncedView;
    // This class represents two views into data, one host side and one device side.
    // This should only be used in cases where the user is aware of the synchronization
    // needed
    // This is used in data types that are expected to be used in a ct::DataTable on both the CPU and GPU
    // the SyncedView should be used as a member of that data type in a similar fashion as a ct::TArrayView
    // Can be used in that datatype, or even better a TSyncedView.
    struct AQUILA_EXPORTS SyncedView
    {
        SyncedView(void* host_ptr, void* device_ptr, size_t size);

        MO_XINLINE ct::TArrayView<const void> host() const
        {
            return ct::TArrayView<const void>(h_ptr, m_size);
        }

        MO_XINLINE ct::TArrayView<void> mutableHost() const
        {
            return ct::TArrayView<void>(h_ptr, m_size);
        }

        MO_XINLINE ct::TArrayView<const void> device() const
        {
            return ct::TArrayView<const void>(d_ptr, m_size);
        }

        MO_XINLINE ct::TArrayView<void> mutableDevice() const
        {
            return ct::TArrayView<void>(d_ptr, m_size);
        }

        MO_XINLINE size_t size() const
        {
            return m_size;
        }

      private:
        friend struct ConstSyncedView;
        void* h_ptr = nullptr;
        void* d_ptr = nullptr;
        size_t m_size = 0;
    };

    // Const synced view is safer than synced view since once this is returned from
    // a synced memory, we can be sure that no mutation of the data will occur
    // and thus we can be sure any additional synchronization is not necessary after usage
    struct AQUILA_EXPORTS ConstSyncedView
    {
        ConstSyncedView(const void* host_ptr, const void* device_ptr, size_t size);
        MO_XINLINE ConstSyncedView(const SyncedView& other)
            : h_ptr(other.h_ptr)
            , d_ptr(other.d_ptr)
            , m_size(other.m_size)
        {
        }

        MO_XINLINE ct::TArrayView<const void> host() const
        {
            return ct::TArrayView<const void>(h_ptr, m_size);
        }

        MO_XINLINE ct::TArrayView<const void> device() const
        {
            return ct::TArrayView<const void>(d_ptr, m_size);
        }

        MO_XINLINE size_t size() const
        {
            return m_size;
        }

      private:
        const void* h_ptr = nullptr;
        const void* d_ptr = nullptr;
        size_t m_size = 0;
    };
}

namespace ct
{
    REFLECT_BEGIN(aq::SyncedView)
        PROPERTY(data, &aq::SyncedView::host, &aq::SyncedView::mutableHost)
        PROPERTY(size, &aq::SyncedView::size, nullptr)
    REFLECT_END;

    REFLECT_BEGIN(aq::ConstSyncedView)
        PROPERTY(data, &aq::ConstSyncedView::host, nullptr)
        PROPERTY(size, &aq::ConstSyncedView::size, nullptr)
    REFLECT_END;
}

#endif // AQUILA_SYNCED_VIEW_HPP
