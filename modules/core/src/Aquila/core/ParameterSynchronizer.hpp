#ifndef AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
#define AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
#include <Aquila/core/export.hpp>

#include <MetaObject/params/TDataContainer.hpp>
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

namespace variadic_args
{
    namespace detail
    {
        template<class Functor, ct::index_t I, class Arg>
        void applyToArgsHelper(Functor& functor, ct::Indexer<I> idx, Arg& arg)
        {
            functor(I, arg);
        }

        template<class Functor, ct::index_t I, class Arg, class ... Args>
        void applyToArgsHelper(Functor& functor, ct::Indexer<I> idx, Arg& arg, Args&... args)
        {
            functor(I, arg);
            applyToArgsHelper(functor, ++idx, args...);

        }
    }


    template<class Functor, class ... Args>
    void applyToArgs(Functor& functor, Args&... args)
    {
        detail::applyToArgsHelper(functor, ct::Indexer<0>{}, args...);
    }
}

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

        void setSlop(std::chrono::nanoseconds slop);

        mo::OptionalTime findEarliestCommonTimestamp() const;
        void removeTimestamp(const mo::Time& time);

        /**
         * @brief getNextSample get the header for the next sample of data that should be processed
         * @return the header of the next sample of data that we should process
         */
        boost::optional<mo::Header> getNextSample();
        /**
         * @brief getNextSample get the next sample of data that we should process
         * @param data is a vector of the resulting data that we should process
         * @param stream async stream for synchronization purposes
         */
        bool getNextSample(ct::TArrayView<mo::IDataContainerConstPtr_t>& data, mo::IAsyncStream* stream = nullptr);

        template<class ... Types>
        bool getNextSample(mo::TDataContainerConstPtr_t<Types>&... data, mo::IAsyncStream* stream = nullptr);

        template<class ... Types>
        bool getNextSample(Types&... data, mo::IAsyncStream* stream = nullptr);

        template<class ... Types>
        bool getNextSample(std::tuple<mo::TDataContainerConstPtr_t<Types>...>& data, mo::IAsyncStream* stream = nullptr);

        template<class ... Types>
        bool getNextSample(std::tuple<Types...>& data, mo::IAsyncStream* stream = nullptr);

      private:
        bool closeEnough(const mo::Time& reference_time, const mo::Time& other_time) const;
        void onNewData();
        void onParamUpdate(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&);
        bool dedoup(const mo::Time&);

        std::function<Callback_s> m_callback;
        mo::TSlot<mo::Update_s> m_slot;

        std::vector<mo::ISubscriber*> m_subscribers;

        std::unordered_map<const mo::IParam*, boost::circular_buffer<mo::Header>> m_headers;

        boost::circular_buffer<mo::Time> m_previous_timestamps;
        std::chrono::nanoseconds m_slop;

        template<class ... Types, size_t ... I>
        bool unpackTuple(std::tuple<Types...>& data, ct::IndexSequence<I...>, mo::IAsyncStream* stream)
        {
            return this->template getNextSample<Types...>(std::get<I>(data)..., stream);
        }
    };

    struct ScatterDataHelper
    {
        ScatterDataHelper(ct::TArrayView<mo::IDataContainerConstPtr_t>& vec): m_vec(vec){}

        template<class T>
        void operator()(const uint32_t i, mo::TDataContainerConstPtr_t<T>& data)
        {
            data = std::dynamic_pointer_cast<const mo::TDataContainer<T>>(m_vec[i]);
            // It is a programming error to not cast to the correct type
            MO_ASSERT(data);
        }

        template<class T>
        void operator()(const uint32_t i, T& data)
        {
            using Container_t = mo::TDataContainerBase<typename mo::ContainerTraits<T>::type, mo::ContainerTraits<T>::CONST>;
            auto tmp = std::dynamic_pointer_cast<const Container_t>(m_vec[i]);
            // It is a programming error to not cast to the correct type
            MO_ASSERT(tmp);
            data = tmp->data;
        }
    private:
        ct::TArrayView<mo::IDataContainerConstPtr_t>& m_vec;
    };

    template<class ... Types>
    bool ParameterSynchronizer::getNextSample(mo::TDataContainerConstPtr_t<Types>&... data, mo::IAsyncStream* stream)
    {
        mo::IDataContainerConstPtr_t vec[sizeof...(data)];
        ct::TArrayView<mo::IDataContainerConstPtr_t> view(vec, sizeof...(data));
        if(this->getNextSample(view, stream))
        {
            ScatterDataHelper helper(view);
            variadic_args::applyToArgs(helper, data...);
            return true;
        }
        return false;
    }

    template<class ... Types>
    bool ParameterSynchronizer::getNextSample(Types&... data, mo::IAsyncStream* stream)
    {
        mo::IDataContainerConstPtr_t vec[sizeof...(data)];
        ct::TArrayView<mo::IDataContainerConstPtr_t> view(vec, sizeof...(data));
        if(this->getNextSample(view, stream))
        {
            ScatterDataHelper helper(view);
            variadic_args::applyToArgs(helper, data...);
            return true;
        }
        return false;
    }

    template<class ... Types>
    bool ParameterSynchronizer::getNextSample(std::tuple<mo::TDataContainerConstPtr_t<Types>...>& data, mo::IAsyncStream* stream)
    {
        return this->unpackTuple(data, ct::makeIndexSequence<sizeof...(Types)>{}, stream);
    }

    template<class ... Types>
    bool ParameterSynchronizer::getNextSample(std::tuple<Types...>& data, mo::IAsyncStream* stream)
    {
        return this->unpackTuple(data, ct::makeIndexSequence<sizeof...(Types)>{}, stream);
    }
}

#endif // AQ_CORE_PARAMETER_SYNCHRONIZER_HPP
