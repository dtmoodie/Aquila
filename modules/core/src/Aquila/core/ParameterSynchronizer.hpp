#ifndef AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
#define AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
#include <Aquila/core/export.hpp>

#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/params/Header.hpp>
#include <MetaObject/signals/TSlot.hpp>
#include <MetaObject/types/small_vec.hpp>

#include <boost/circular_buffer.hpp>
#include <functional>

namespace mo
{
    struct ISubscriber;
    struct IParam;
} // namespace mo

namespace aq
{
    class ParameterSynchronizer
    {
      public:
        using SubscriberVec_t = mo::SmallVec<mo::ISubscriber*, 20>;
        using Callback_s = void(const mo::Time*, const mo::FrameNumber*, const SubscriberVec_t);
        ParameterSynchronizer(std::chrono::nanoseconds slop = std::chrono::nanoseconds(0));
        virtual ~ParameterSynchronizer();

        virtual void setInputs(std::vector<mo::ISubscriber*>);
        virtual void setCallback(std::function<Callback_s>);

        mo::OptionalTime findEarliestCommonTimestamp() const;
        void removeTimestamp(const mo::Time& time);

      private:
        bool closeEnough(const mo::Time& reference_time, const mo::Time& other_time) const;
        void onNewData();
        void onParamUpdate(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&);
        bool dedoup(const mo::Time&);

        std::function<Callback_s> m_callback;
        mo::TSlot<mo::Update_s> m_slot;

        std::vector<mo::ISubscriber*> m_publishers;

        std::unordered_map<const mo::IParam*, boost::circular_buffer<mo::Header>> m_headers;

        boost::circular_buffer<mo::Time> m_previous_timestamps;
        std::chrono::nanoseconds m_slop;
    };
}

#endif // AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
