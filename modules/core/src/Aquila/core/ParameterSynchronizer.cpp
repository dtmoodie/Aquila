#include "ParameterSynchronizer.hpp"
#include <MetaObject/params/IPublisher.hpp>

namespace aq
{
    ParameterSynchronizer::ParameterSynchronizer()
    {
        m_slot.bind(&ParameterSynchronizer::onParamUpdate, this);
        m_previous_timestamps.set_capacity(20);
    }

    ParameterSynchronizer::~ParameterSynchronizer()
    {
    }

    void ParameterSynchronizer::setInputs(std::vector<mo::IPublisher*> inputs)
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

    mo::OptionalTime ParameterSynchronizer::findEarliestCommonTimestamp() const
    {
        mo::OptionalTime output;
        uint32_t valid_count = 0;
        for (auto itr = m_headers.begin(); itr != m_headers.end(); ++itr)
        {
            if (!itr->second.empty())
            {
                const mo::Header& hdr = itr->second.front();
                if (boost::none != hdr.timestamp)
                {
                    if (boost::none == output)
                    {
                        output = hdr.timestamp;
                        ++valid_count;
                        continue;
                    }
                    else
                    {
                        if(output == hdr.timestamp)
                        {
                            ++valid_count;
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
        auto itr = std::find(m_previous_timestamps.begin(), m_previous_timestamps.end(), time);
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
        for( auto& itr : m_headers)
        {
            auto find_itr = std::find_if(itr.second.begin(), itr.second.end(), [&time](const mo::Header& hdr)
            {
                if(hdr.timestamp)
                {
                    return time == hdr.timestamp;
                }
                return false;
            });
            if(find_itr != itr.second.end())
            {
                itr.second.erase(find_itr);
                // We don't expect duplicate timestamps in the same circular buffer, perhaps this should be a loop until we don't find the timestamp?
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
        itr->second.push_back(header);
        onNewData();
    }

} // namespace aq
