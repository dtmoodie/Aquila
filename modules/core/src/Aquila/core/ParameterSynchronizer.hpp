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
    class IPublisher;
    class IParam;
} // namespace mo

namespace aq
{
    class ParameterSynchronizer
    {
      public:
        using PublisherVec_t = mo::SmallVec<mo::IPublisher*, 20>;
        using Callback_s = void(const mo::Time*, const mo::FrameNumber*, const PublisherVec_t);
        ParameterSynchronizer();
        virtual ~ParameterSynchronizer();

        virtual void setInputs(std::vector<mo::IPublisher*>);
        virtual void setCallback(std::function<Callback_s>);

        mo::OptionalTime findEarliestCommonTimestamp() const;
        void removeTimestamp(const mo::Time& time);

      private:
        void onNewData();
        void onParamUpdate(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&);
        bool dedoup(const mo::Time&);

        std::function<Callback_s> m_callback;
        mo::TSlot<mo::Update_s> m_slot;

        std::vector<mo::IPublisher*> m_publishers;

        std::unordered_map<const mo::IParam*, boost::circular_buffer<mo::Header>> m_headers;

        boost::circular_buffer<mo::Time> m_previous_timestamps;
    };
}

#endif // AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
