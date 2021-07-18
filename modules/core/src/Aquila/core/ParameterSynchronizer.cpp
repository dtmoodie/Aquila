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
    }

    ParameterSynchronizer::~ParameterSynchronizer()
    {
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
        m_publishers = std::move(inputs);
    }

    void ParameterSynchronizer::setCallback(std::function<Callback_s> cb)
    {
        m_callback = std::move(cb);
    }

    bool ParameterSynchronizer::closeEnough(const mo::Time& reference_time, const mo::Time& other_time) const
    {
        return std::abs(std::chrono::nanoseconds(reference_time.time_since_epoch()).count() - std::chrono::nanoseconds(other_time.time_since_epoch()).count()) <= m_slop.count();
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

    void ParameterSynchronizer::onNewData()
    {
        mo::OptionalTime time = findEarliestCommonTimestamp();

        if(time)
        {

            if(dedoup(*time))
            {
                if(m_callback)
                {
                    m_callback(time.get_ptr(), nullptr, m_publishers);
                }
            }
            removeTimestamp(*time);
        }
    }

    void
    ParameterSynchronizer::onParamUpdate(const mo::IParam& param, mo::Header header, mo::UpdateFlags, mo::IAsyncStream&)
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
                if(header.frame_number > itr->second.back().frame_number)
                {
                    itr->second.push_back(std::move(header));
                }
            }
        }else
        {
            itr->second.push_back(std::move(header));
        }
        onNewData();
    }

} // namespace aq
