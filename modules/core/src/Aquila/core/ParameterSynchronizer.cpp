#include "ParameterSynchronizer.hpp"
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <chrono>

namespace aq
{
    ParameterSynchronizer::ParameterSynchronizer(std::chrono::nanoseconds slop):
        m_slop(std::move(slop))
    {
        m_slot.bind(&ParameterSynchronizer::onParamUpdate, this);
        m_previous_timestamps.set_capacity(20);
        m_previous_frame_numbers.set_capacity(20);
    }

    ParameterSynchronizer::~ParameterSynchronizer() = default;

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

    mo::OptionalTime ParameterSynchronizer::findEarliestCommonTimestamp() const
    {
        // Todo earliest guarantee?
        mo::OptionalTime output;
        uint32_t valid_count = 0;
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            if (!itr->second.empty())
            {
                if(boost::none == output)
                {
                    const mo::Header& hdr = itr->second.front();
                    if (boost::none != hdr.timestamp)
                    {
                        output = hdr.timestamp;
                        ++valid_count;
                    }
                }else
                {
                    for(const mo::Header& hdr : itr->second)
                    {
                        if(boost::none != hdr.timestamp)
                        {
                            if(closeEnough(*hdr.timestamp, *output))
                            {
                                if(*hdr.timestamp < *output)
                                {
                                    output = hdr.timestamp;
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
        auto itr = m_headers.find(&param);
        MO_ASSERT(itr != m_headers.end());
        if(itr->second.size())
        {
            if(header.timestamp != boost::none && itr->second.back().timestamp != boost::none)
            {
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
