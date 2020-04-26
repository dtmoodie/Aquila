#ifndef AQUILA_TSYNCED_VIEW_HPP
#define AQUILA_TSYNCED_VIEW_HPP
#include "SyncedView.hpp"

namespace aq
{
    template <class T>
    struct TSyncedView : public SyncedView
    {
        MO_XINLINE T& operator[](size_t idx)
        {
#ifdef __CUDA_ARCH__
            return mutableDevice()[idx];
#else
            return mutableHost()[idx];
#endif
        }

        MO_XINLINE const T& operator[](size_t idx) const
        {
#ifdef __CUDA_ARCH__
            return device()[idx];
#else
            return host()[idx];
#endif
        }

        MO_XINLINE ct::TArrayView<const T> host() const
        {
            return SyncedView::host();
        }

        MO_XINLINE ct::TArrayView<T> mutableHost() const
        {
            return SyncedView::host();
        }

        MO_XINLINE ct::TArrayView<const T> device() const
        {
            return SyncedView::device();
        }

        MO_XINLINE ct::TArrayView<T> mutableDevice() const
        {
            return SyncedView::mutableDevice();
        }
    };

    template <class T>
    struct TSyncedView<const T> : public ConstSyncedView
    {
        MO_XINLINE const T& operator[](size_t idx) const
        {
#ifdef __CUDA_ARCH__
            return device()[idx];
#else
            return host()[idx];
#endif
        }

        MO_XINLINE ct::TArrayView<const T> host() const
        {
            return ConstSyncedView::host();
        }

        MO_XINLINE ct::TArrayView<const T> device() const
        {
            return ConstSyncedView::device();
        }
    };
} // namespace aq

namespace ct
{
    // clang-format on
    template <class T>
    struct ReflectImpl<aq::TSyncedView<T>>
    {
        using DataType = aq::TSyncedView<T>;
        using TemplateParameters = VariadicTypedef<T>;
        static constexpr const bool SPECIALIZED = true;
        REFLECT_STUB
            PROPERTY(data, &DataType::host, &DataType::mutableHost)
        REFLECT_INTERNAL_END;
    };

    template <class T>
    struct ReflectImpl<aq::TSyncedView<const T>>
    {
        using DataType = aq::TSyncedView<const T>;
        using TemplateParameters = VariadicTypedef<const T>;
        static constexpr const bool SPECIALIZED = true;
        REFLECT_STUB
            PROPERTY(data, &DataType::host, nullptr)
        REFLECT_INTERNAL_END;
    };
    // clang-format off
}

#endif // AQUILA_TSYNCED_VIEW_HPP
