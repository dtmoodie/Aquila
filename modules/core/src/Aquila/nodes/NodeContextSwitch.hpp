#ifndef AQUILA_NODES_STREAM_SWITCH_HPP
#define AQUILA_NODES_STREAM_SWITCH_HPP

#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/params/TypeSelector.hpp>

namespace aq
{
    namespace nodes
    {
        template <class N, class... Args>
        auto nodeStreamSwitch(N* node, mo::IAsyncStream& stream, Args&&... args)
            -> decltype(node->processImpl(stream, std::forward<Args>(args)...))
        {
            if (stream.isDeviceStream())
            {
                mo::IDeviceStream* device_stream = stream.getDeviceStream();
                return node->processImpl(*device_stream, std::forward<Args>(args)...);
            }
            else
            {
                return node->processImpl(stream, std::forward<Args>(args)...);
            }
        }

    } // namespace nodes
} // namespace aq

#endif // AQUILA_NODES_STREAM_SWITCH_HPP