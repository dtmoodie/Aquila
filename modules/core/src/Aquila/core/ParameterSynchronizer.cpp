#include "ParameterSynchronizer.hpp"
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <chrono>

namespace aq
{
    ParameterSynchronizer::ParameterSynchronizer(spdlog::logger& logger, std::chrono::nanoseconds slop)
        : m_slop(std::move(slop))
        , m_logger(&logger)
    {
        m_slot.bind(&ParameterSynchronizer::onParamUpdate, this);
        m_previous_timestamps.set_capacity(20);
        m_previous_frame_numbers.set_capacity(20);
    }

    ParameterSynchronizer::~ParameterSynchronizer() = default;

    void ParameterSynchronizer::setLogger(spdlog::logger& logger)
    {
        m_logger = &logger;
    }

    void ParameterSynchronizer::setInputs(std::vector<mo::ISubscriber*> inputs)
    {
        for (auto input : inputs)
        {
            if (m_headers.find(input) == m_headers.end())
            {
                m_headers[input] = boost::circular_buffer<mo::Header>(100);
            }
            input->registerUpdateNotifier(m_slot);
        }
        m_subscribers = std::move(inputs);
    }

    void ParameterSynchronizer::setCallback(std::function<Callback_s> cb)
    {
        m_callback = std::move(cb);
    }

    void ParameterSynchronizer::setSlop(std::chrono::nanoseconds slop)
    {
        m_slop = std::move(slop);
    }

    bool ParameterSynchronizer::closeEnough(const mo::Time& reference_time, const mo::Time& other_time) const
    {
        const std::chrono::nanoseconds delta = reference_time.time_since_epoch() - other_time.time_since_epoch();
        return std::abs(delta.count()) <= m_slop.count();
    }

    template <class T, class F>
    T ParameterSynchronizer::findDirect(F&& predicate) const
    {
        uint32_t valid_count = 0;
        uint32_t direct_inputs = 0;
        T output;
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            const mo::IPublisher* publisher = itr->first->getPublisher();
            // If the input isn't set, can't do anything unless it is an optional input
            if (publisher == nullptr)
            {
                if (!itr->first->checkFlags(mo::ParamFlags::kOPTIONAL))
                {
                    m_logger->warn("No input set for '{}'", itr->first->getName());
                }
            }
            else
            {
                // If this is not a buffered connection, then we need to use the direct timestamp since there is no
                // buffer history, we can only operate on current data
                if (!publisher->checkFlags(mo::ParamFlags::kBUFFER))
                {
                    ++direct_inputs;
                    if (itr->first->hasNewData())
                    {
                        auto hdr = itr->first->getNewestHeader();
                        if (hdr)
                        {
                            if (predicate(output, *hdr))
                            {
                                ++valid_count;
                            }
                        }
                    }
                }
            }
        }
        if (valid_count != direct_inputs)
        {
            output = T();
        }
        return output;
    }

    struct TimestampPredicate
    {
        TimestampPredicate(const std::chrono::nanoseconds& slop)
            : m_slop(slop)
        {
        }

        bool equal(const mo::Time& t0, const mo::Time& t1)
        {
            const std::chrono::nanoseconds delta = t0.time_since_epoch() - t1.time_since_epoch();
            return std::abs(delta.count()) <= m_slop.count();
        }

        bool less(const mo::Time& t0, const mo::Time& t1)
        {
            return t0 < t1;
        }

        template <class T>
        bool check(mo::OptionalTime& output, const mo::Header& hdr, T pred)
        {
            if (boost::none != hdr.timestamp)
            {
                if (boost::none == output)
                {
                    output = hdr.timestamp;
                    return true;
                }
                else
                {
                    if (pred(*output, *hdr.timestamp))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        bool checkExact(mo::OptionalTime& output, const mo::Header& hdr)
        {
            return check(output, hdr, [this](const mo::Time& t0, const mo::Time& t1) { return this->equal(t0, t1); });
        }

        bool checkLessThan(mo::OptionalTime& output, const mo::Header& hdr)
        {
            if (check(output, hdr, [this](const mo::Time& t0, const mo::Time& t1) { return this->less(t0, t1); }))
            {
                return true;
            }
            // We found a new smaller timestamp and thus we update output
            output = hdr.timestamp;
            return false;
        }

      private:
        std::chrono::nanoseconds m_slop;
    };

    mo::OptionalTime ParameterSynchronizer::findDirectTimestamp() const
    {
        TimestampPredicate pred(m_slop);
        return findDirect<mo::OptionalTime>(
            [&pred](mo::OptionalTime& output, const mo::Header& hdr) { return pred.checkExact(output, hdr); });
    }

    template <class T, class F>
    T ParameterSynchronizer::findEarliest(F&& predicate) const
    {
        T output;
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            // The assumption that itr->second is sorted is not necessarily true since data playback could be rewound
            for (const mo::Header& hdr : itr->second)
            {
                predicate(output, hdr);
            }
        }
        return output;
    }

    mo::OptionalTime ParameterSynchronizer::findEarliestTimestamp() const
    {
        TimestampPredicate pred(m_slop);
        return findEarliest<mo::OptionalTime>(
            [&pred](mo::OptionalTime& output, const mo::Header& hdr) { return pred.checkLessThan(output, hdr); });
    }

    mo::OptionalTime ParameterSynchronizer::findEarliestCommonTimestamp() const
    {
        // Todo earliest guarantee?
        mo::OptionalTime output = findDirectTimestamp();
        if (boost::none != output)
        {
            return output;
        }
        output = findEarliestTimestamp();
        // Can't continue since we don't have any data to start from
        if (boost::none == output)
        {
            return output;
        }
        uint32_t valid_count = 0;
        uint32_t required_count = 0;

        const auto check = [this, &output](const boost::circular_buffer<mo::Header>& headers) -> bool {
            for (const mo::Header& hdr : headers)
            {
                if (boost::none != hdr.timestamp)
                {
                    if (closeEnough(*hdr.timestamp, *output))
                    {

                        return true;
                    }
                }
            }
            return false;
        };
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            if (!itr->first->checkFlags(mo::ParamFlags::kOPTIONAL))
            {
                ++required_count;
            }
            if (!itr->second.empty())
            {
                bool found = check(itr->second);

                // The selected timestamp was not found for this input,
                if (found)
                {
                    ++valid_count;
                }
                else
                {
                    if (!itr->second.empty())
                    {
                        const mo::Header& hdr = *itr->second.begin();
                        if (boost::none != hdr.timestamp)
                        {
                            if (*hdr.timestamp > *output)
                            {
                                output = hdr.timestamp;
                                valid_count = 0;
                                itr = m_headers.begin();
                                found = check(itr->second);
                                if (found)
                                {
                                    ++valid_count;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (valid_count >= required_count)
        {
            return output;
        }

        return {};
    }

    struct FrameNumberPredicate
    {

        static bool equal(const mo::FrameNumber& t0, const mo::FrameNumber& t1)
        {
            return t0 == t1;
        }

        static bool less(const mo::FrameNumber& t0, const mo::FrameNumber& t1)
        {
            return t0 < t1;
        }

        template <class T>
        static bool check(mo::FrameNumber& output, const mo::Header& hdr, T pred)
        {
            if (boost::none == hdr.timestamp)
            {
                if (!output.valid())
                {
                    output = hdr.frame_number;
                    return true;
                }
                else
                {
                    if (pred(output, hdr.frame_number))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        static bool checkExact(mo::FrameNumber& output, const mo::Header& hdr)
        {
            return check(output, hdr, &FrameNumberPredicate::equal);
        }

        static bool checkLessThan(mo::FrameNumber& output, const mo::Header& hdr)
        {
            if (!check(output, hdr, &FrameNumberPredicate::less))
            {
                output = hdr.frame_number;
                return false;
            }
            return true;
        }
    };

    mo::FrameNumber ParameterSynchronizer::findDirectFrameNumber() const
    {
        return findDirect<mo::FrameNumber>(&FrameNumberPredicate::checkExact);
    }

    mo::FrameNumber ParameterSynchronizer::findEarliestFrameNumber() const
    {
        return findEarliest<mo::FrameNumber>(&FrameNumberPredicate::checkLessThan);
    }

    mo::FrameNumber ParameterSynchronizer::findEarliestCommonFrameNumber() const
    {
        mo::FrameNumber output = findDirectFrameNumber();
        if (output.valid())
        {
            return output;
        }
        // Todo earliest guarantee?
        uint32_t valid_count = 0;
        uint32_t required_count = 0;
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            if (!itr->first->checkFlags(mo::ParamFlags::kOPTIONAL))
            {
                ++required_count;
            }
            if (!itr->second.empty())
            {
                if (!output.valid())
                {
                    const mo::Header& hdr = itr->second.front();
                    if (hdr.frame_number.valid())
                    {
                        output = hdr.frame_number;
                        ++valid_count;
                    }
                }
                else
                {
                    for (const mo::Header& hdr : itr->second)
                    {
                        if (hdr.frame_number.valid())
                        {
                            // We do an exact comparison because if we want to synchronize across different sources, we
                            // should be synchronizing based on time stamp
                            if (hdr.frame_number == output)
                            {
                                if (hdr.frame_number < output)
                                {
                                    output = hdr.frame_number;
                                }
                                ++valid_count;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if (valid_count >= required_count)
        {
            return output;
        }

        return {};
    }

    bool ParameterSynchronizer::dedoup(const mo::Time& time)
    {
        auto itr = std::find_if(m_previous_timestamps.begin(),
                                m_previous_timestamps.end(),
                                [time, this](const mo::Time& other) { return closeEnough(other, time); });
        if (itr != m_previous_timestamps.end())
        {
            // Duplicate timestamp detected
            return false;
        }
        m_previous_timestamps.push_back(time);
        return true;
    }

    bool ParameterSynchronizer::dedoup(const mo::FrameNumber& fn)
    {
        auto itr = std::find_if(m_previous_frame_numbers.begin(),
                                m_previous_frame_numbers.end(),
                                [fn](const mo::FrameNumber& other) { return other == fn; });
        if (itr != m_previous_frame_numbers.end())
        {
            // Duplicate timestamp detected
            return false;
        }
        m_previous_frame_numbers.push_back(fn);
        return true;
    }

    void ParameterSynchronizer::removeTimestamp(const mo::Time& time)
    {
        // Remove any timestamp instances that are less than the new time
        // We don't go back and look at stale data
        for (auto& itr : m_headers)
        {
            auto find_itr = itr.second.begin();
            auto pred = [&time, this](const mo::Header& hdr) {
                if (hdr.timestamp)
                {
                    return *hdr.timestamp <= (time + m_slop);
                }
                return false;
            };
            find_itr = std::find_if(find_itr, itr.second.end(), pred);
            while (find_itr != itr.second.end())
            {
                find_itr = itr.second.erase(find_itr);
                find_itr = std::find_if(find_itr, itr.second.end(), pred);
            }
        }
    }

    void ParameterSynchronizer::removeFrameNumber(const mo::FrameNumber& fn)
    {
        // Remove any timestamp instances that are less than the new time
        // We don't go back and look at stale data
        for (auto& itr : m_headers)
        {
            auto find_itr = itr.second.begin();
            auto pred = [&fn](const mo::Header& hdr) {
                if (hdr.frame_number.valid())
                {
                    return hdr.frame_number == fn;
                }
                return false;
            };
            find_itr = std::find_if(find_itr, itr.second.end(), pred);
            while (find_itr != itr.second.end())
            {
                find_itr = itr.second.erase(find_itr);
                find_itr = std::find_if(find_itr, itr.second.end(), pred);
            }
        }
    }

    boost::optional<mo::Header> ParameterSynchronizer::getNextSample()
    {
        mo::OptionalTime time = findEarliestCommonTimestamp();

        if (time)
        {
            const bool dedoup_success = dedoup(*time);
            removeTimestamp(*time);
            return dedoup_success ? boost::optional<mo::Header>(mo::Header(*time)) : boost::optional<mo::Header>();
        }
        mo::FrameNumber fn = findEarliestCommonFrameNumber();
        if (fn.valid())
        {
            const bool dedoup_success = dedoup(fn);
            removeFrameNumber(fn);
            return dedoup_success ? boost::optional<mo::Header>(mo::Header(fn)) : boost::optional<mo::Header>();
        }
        return {};
    }

    bool ParameterSynchronizer::getNextSample(ct::TArrayView<mo::IDataContainerConstPtr_t>& data,
                                              mo::IAsyncStream* stream)
    {
        boost::optional<mo::Header> header = this->getNextSample();
        if (header)
        {
            MO_ASSERT_GE(data.size(), this->m_subscribers.size());
            for (size_t i = 0; i < this->m_subscribers.size(); ++i)
            {
                data[i] = m_subscribers[i]->getData(header.get_ptr(), stream);
            }
            return true;
        }
        return false;
    }

    void ParameterSynchronizer::onNewData()
    {
        boost::optional<mo::Header> hdr = getNextSample();
        if (hdr)
        {
            if (m_callback)
            {
                if (hdr->timestamp)
                {
                    m_callback(hdr->timestamp.get_ptr(), nullptr, m_subscribers);
                }
                else
                {
                    m_callback(nullptr, &hdr->frame_number, m_subscribers);
                }
            }
        }
    }

    void ParameterSynchronizer::onParamUpdate(const mo::IParam& param,
                                              mo::Header header,
                                              mo::UpdateFlags flags,
                                              mo::IAsyncStream*)
    {
        auto itr = m_headers.find(dynamic_cast<const mo::ISubscriber*>(&param));
        MO_ASSERT(itr != m_headers.end());
        if (itr->second.size())
        {
            if (header.timestamp != boost::none && itr->second.back().timestamp != boost::none)
            {
                itr->second.push_back(std::move(header));
            }
            else
            {
                itr->second.push_back(std::move(header));
            }
        }
        else
        {
            if (flags == mo::UpdateFlags::kINPUT_UPDATED || flags == mo::UpdateFlags::kBUFFER_UPDATED)
            {
                itr->second.push_back(std::move(header));
            }
        }
        if (m_callback)
        {
            onNewData();
        }
    }

} // namespace aq
