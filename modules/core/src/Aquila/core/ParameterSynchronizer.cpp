#include "ParameterSynchronizer.hpp"
#include <MetaObject/params/IPublisher.hpp>

namespace aq
{
    ParameterSynchronizer::ParameterSynchronizer()
    {
        m_slot.bind(&ParameterSynchronizer::onParamUpdate, this);
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
        }
    }

    void ParameterSynchronizer::setCallback(std::function<Callback_s>)
    {
    }

    boost::optional<mo::Time> ParameterSynchronizer::findEarliestCommonTimestamp() const
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
                    ++valid_count;
                    if (boost::none != output)
                    {
                        output = hdr.timestamp;
                        continue;
                    }
                    else
                    {
                    }
                }
            }
        }

        return output;
    }

    void
    ParameterSynchronizer::onParamUpdate(const mo::IParam& param, mo::Header header, mo::UpdateFlags, mo::IAsyncStream&)
    {
        auto itr = m_headers.find(&param);
        MO_ASSERT(itr != m_headers.end());
        itr->second.push_back(header);
    }

} // namespace aq
