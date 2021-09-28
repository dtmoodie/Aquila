#include "ParameterSynchronizer.hpp"
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <chrono>

namespace aq
{
    ParameterSynchronizer::ParameterSynchronizer(spdlog::logger& logger, std::chrono::nanoseconds slop):
        m_slop(std::move(slop)),
        m_logger(&logger)
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

    template<class T, class F>
    T ParameterSynchronizer::findDirect(F&& predicate) const
    {
        uint32_t valid_count = 0;
        uint32_t direct_inputs = 0;
        T output;
        for(auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            const mo::IPublisher* publisher = itr->first->getPublisher();
            // If the input isn't set, can't do anything unless it is an optional input
            if(publisher == nullptr)
            {
                if(!itr->first->checkFlags(mo::ParamFlags::kOPTIONAL))
                {
                    m_logger->warn("No input set for '{}'", itr->first->getName());
                }
            }else
            {
                // If this is not a buffered connection, then we need to use the direct timestamp since there is no buffer history, we can only operate on current data
                if(!publisher->checkFlags(mo::ParamFlags::kBUFFER))
                {
                    ++direct_inputs;
                    if(itr->first->hasNewData())
                    {
                        auto hdr = itr->first->getNewestHeader();
                        if(hdr)
                        {
                            output = predicate(std::move(output), *hdr, valid_count);
                        }
                    }
                }
            }
        }
        if(valid_count != direct_inputs)
        {
            output = T();
        }
        return output;
    }

    mo::OptionalTime ParameterSynchronizer::findDirectTimestamp() const
    {
        auto predicate = [](mo::OptionalTime&& output, const mo::Header& hdr, uint32_t& valid_count) -> mo::OptionalTime
        {
            if(boost::none == output && boost::none != hdr.timestamp)
            {
                output = hdr.timestamp;
                ++valid_count;
            }else
            {
                if(boost::none != hdr.timestamp)
                {
                    if(output == hdr.timestamp)
                    {
                        ++valid_count;
                    }
                }
            }
            return std::move(output);
        };

        return findDirect<mo::OptionalTime>(std::move(predicate));
    }

    template<class T, class F>
    T ParameterSynchronizer::findEarliest(F&& predicate) const
    {
        T output;
        for(auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            // The assumption that itr->second is sorted is not necessarily true since data playback could be rewound
            for(const mo::Header& hdr : itr->second)
            {
                output = predicate(std::move(output), hdr);
            }
        }
        return output;
    }

    mo::OptionalTime ParameterSynchronizer::findEarliestTimestamp() const
    {
        auto predicate = [](mo::OptionalTime&& output, const mo::Header& hdr) -> mo::OptionalTime
        {
            if(boost::none != hdr.timestamp)
            {
                if(boost::none == output)
                {
                    output = hdr.timestamp;
                }else
                {
                    if(*hdr.timestamp < *output)
                    {
                        output = hdr.timestamp;
                    }
                }
            }
            return std::move(output);
        };
        return findEarliest<mo::OptionalTime>(std::move(predicate));
    }

    mo::OptionalTime ParameterSynchronizer::findEarliestCommonTimestamp() const
    {
        // Todo earliest guarantee?
        mo::OptionalTime output = findDirectTimestamp();
        if(boost::none != output)
        {
            return output;
        }
        output = findEarliestTimestamp();
        // Can't continue since we don't have any data to start from
        if(boost::none == output)
        {
            return output;
        }
        uint32_t valid_count = 0;

        const auto check = [this, &output](const boost::circular_buffer<mo::Header>& headers) -> bool
        {
            for(const mo::Header& hdr : headers)
            {
                if(boost::none != hdr.timestamp)
                {
                    if(closeEnough(*hdr.timestamp, *output))
                    {

                        return true;
                    }
                }
            }
            return false;
        };
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            if (!itr->second.empty())
            {
                bool found = check(itr->second);

                // The selected timestamp was not found for this input,
                if(found)
                {
                    ++valid_count;
                }
                else
                {
                    if(!itr->second.empty())
                    {
                        const mo::Header& hdr = *itr->second.begin();
                        if(boost::none != hdr.timestamp)
                        {
                            if(*hdr.timestamp > *output)
                            {
                                output = hdr.timestamp;
                                valid_count = 0;
                                itr = m_headers.begin();
                                found = check(itr->second);
                                if(found)
                                {
                                    ++valid_count;
                                }
                            }
                        }
                    }
                }
            }
        }

        if(valid_count != m_headers.size())
        {
            return {};
        }

        return output;
    }

    mo::FrameNumber ParameterSynchronizer::findDirectFrameNumber() const
    {
        auto predicate = [](mo::FrameNumber&& output, const mo::Header& hdr, uint32_t& valid_count) -> mo::FrameNumber
        {
            if(!output.valid())
            {
                output = hdr.frame_number;
                ++valid_count;
            }else
            {
                if(output == hdr.frame_number)
                {
                    ++valid_count;
                }
            }
            return std::move(output);
        };
        return findDirect<mo::FrameNumber>(std::move(predicate));
    }

    mo::FrameNumber ParameterSynchronizer::findEarliestFrameNumber() const
    {
        auto predicate = [](mo::FrameNumber&& output, const mo::Header& hdr) -> mo::FrameNumber
        {
            if(boost::none == hdr.timestamp)
            {
                if(!output.valid())
                {
                    output = hdr.frame_number;
                }else
                {
                    if(hdr.frame_number < output)
                    {
                        output = hdr.frame_number;
                    }
                }
            }
            return std::move(output);
        };
        return findEarliest<mo::FrameNumber>(std::move(predicate));
    }

    mo::FrameNumber ParameterSynchronizer::findEarliestCommonFrameNumber() const
    {
        // Todo earliest guarantee?
        mo::FrameNumber output;
        uint32_t valid_count = 0;
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            if (!itr->second.empty())
            {
                if(!output.valid())
                {
                    const mo::Header& hdr = itr->second.front();
                    if (hdr.frame_number.valid())
                    {
                        output = hdr.frame_number;
                        ++valid_count;
                    }
                }else
                {
                    for(const mo::Header& hdr : itr->second)
                    {
                        if(hdr.frame_number.valid())
                        {
                            // We do an exact comparison because if we want to synchronize across different sources, we should be synchronizing based on time stamp
                            if(hdr.frame_number == output)
                            {
                                if(hdr.frame_number < output)
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
        if(valid_count != m_headers.size())
        {
            return {};
        }

        return output;
    }

    bool ParameterSynchronizer::dedoup(const mo::Time& time)
    {
        auto itr = std::find_if(m_previous_timestamps.begin(), m_previous_timestamps.end(), [time, this](const mo::Time& other)
        {
            return closeEnough(other, time);
        });
        if(itr != m_previous_timestamps.end())
        {
            // Duplicate timestamp detected
            return false;
        }
        m_previous_timestamps.push_back(time);
        return true;
    }

    bool ParameterSynchronizer::dedoup(const mo::FrameNumber& fn)
    {
        auto itr = std::find_if(m_previous_frame_numbers.begin(), m_previous_frame_numbers.end(), [fn](const mo::FrameNumber& other)
        {
            return other == fn;
        });
        if(itr != m_previous_frame_numbers.end())
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
        for( auto& itr : m_headers)
        {
            auto find_itr = itr.second.begin();
            auto pred = [&time, this](const mo::Header& hdr)
            {
                if(hdr.timestamp)
                {
                    return *hdr.timestamp <= (time + m_slop);
                }
                return false;
            };
            find_itr = std::find_if(find_itr, itr.second.end(), pred);
            while(find_itr != itr.second.end())
            {
                find_itr = itr.second.erase(find_itr);
                find_itr = std::find_if(find_itr, itr.second.end(), pred);
            }
        }
    }

    void ParameterSynchronizer::removeFrameNumber(const mo::FrameNumber &fn)
    {
        // Remove any timestamp instances that are less than the new time
        // We don't go back and look at stale data
        for( auto& itr : m_headers)
        {
            auto find_itr = itr.second.begin();
            auto pred = [&fn](const mo::Header& hdr)
            {
                if(hdr.frame_number.valid())
                {
                    return hdr.frame_number == fn;
                }
                return false;
            };
            find_itr = std::find_if(find_itr, itr.second.end(), pred);
            while(find_itr != itr.second.end())
            {
                find_itr = itr.second.erase(find_itr);
                find_itr = std::find_if(find_itr, itr.second.end(), pred);
            }
        }
    }

    boost::optional<mo::Header> ParameterSynchronizer::getNextSample()
    {
        mo::OptionalTime time = findEarliestCommonTimestamp();

        if(time)
        {
            const bool dedoup_success = dedoup(*time);
            removeTimestamp(*time);
            return dedoup_success ? boost::optional<mo::Header>(mo::Header(*time)) : boost::optional<mo::Header>();
        }
        mo::FrameNumber fn = findEarliestCommonFrameNumber();
        if(fn.valid())
        {
            const bool dedoup_success = dedoup(fn);
            removeFrameNumber(fn);
            return dedoup_success ? boost::optional<mo::Header>(mo::Header(fn)) : boost::optional<mo::Header>();
        }
        return {};
    }

    bool ParameterSynchronizer::getNextSample(ct::TArrayView<mo::IDataContainerConstPtr_t>& data, mo::IAsyncStream* stream)
    {
        boost::optional<mo::Header> header = this->getNextSample();
        if(header)
        {
            MO_ASSERT_GE(data.size(), this->m_subscribers.size());
            for(size_t i = 0; i < this->m_subscribers.size(); ++i)
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
        if(hdr)
        {
            if(m_callback)
            {
                if(hdr->timestamp)
                {
                    m_callback(hdr->timestamp.get_ptr(), nullptr, m_subscribers);
                }else
                {
                    m_callback(nullptr, &hdr->frame_number, m_subscribers);
                }
            }
        }
    }

    void
    ParameterSynchronizer::onParamUpdate(const mo::IParam& param, mo::Header header, mo::UpdateFlags flags, mo::IAsyncStream&)
    {
        auto itr = m_headers.find(dynamic_cast<const mo::ISubscriber*>(&param));
        MO_ASSERT(itr != m_headers.end());
        if(itr->second.size())
        {
            if(header.timestamp != boost::none && itr->second.back().timestamp != boost::none)
            {
                // Only insert data if it is newer than the most recent sample.
                // This prevents time from moving backward, and causes complications with re-compute.
                if(header.timestamp > itr->second.back().timestamp)
                {
                    itr->second.push_back(std::move(header));
                }
            }else
            {
                if(header.frame_number.valid() && (header.frame_number > itr->second.back().frame_number))
                {
                    itr->second.push_back(std::move(header));
                }
            }
        }else
        {
            if(flags == mo::UpdateFlags::kINPUT_UPDATED || flags == mo::UpdateFlags::kBUFFER_UPDATED)
            {
                itr->second.push_back(std::move(header));
            }
        }
        if(m_callback)
        {
            onNewData();
        }
    }

} // namespace aq
