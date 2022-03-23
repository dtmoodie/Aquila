#include "SyncedView.hpp"

namespace aq
{
    SyncedView::SyncedView(void* host_ptr, void* device_ptr, size_t size)
        : h_ptr(host_ptr)
        , d_ptr(device_ptr)
        , m_size(size)
    {
    }

    ConstSyncedView::ConstSyncedView(const void* host_ptr, const void* device_ptr, size_t size)
        : h_ptr(host_ptr)
        , d_ptr(device_ptr)
        , m_size(size)
    {
    }
} // namespace aq
