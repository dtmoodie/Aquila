#ifndef AQUILA_NODES_STREAM_SWITCH_HPP
#define AQUILA_NODES_STREAM_SWITCH_HPP

#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/params/TypeSelector.hpp>

namespace aq
{
    namespace nodes
    {
        template <class N, class... Args>
        void nodeStreamSwitch(N* node, mo::IAsyncStream& stream, Args&&... args)
        {
            if (stream.isDeviceStream())
            {
                auto device_stream = stream.getDeviceStream();
                node->processImpl(*device_stream, std::forward<Args>(args)...);
            }
            else
            {
                node->processImpl(stream, std::forward<Args>(args)...);
            }
        }

    } // namespace nodes
} // namespace aq

#endif // AQUILA_NODES_STREAM_SWITCH_HPP