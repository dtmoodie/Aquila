#pragma once
#include "Aquila/detail/export.hpp"
#include "nodes.hpp"

struct IObjectConstructor;

namespace aq
{
    namespace python
    {
        AQUILA_EXPORTS void setupFrameGrabberInterface();
        AQUILA_EXPORTS void setupFrameGrabberObjects(std::vector<IObjectConstructor*>& ctrs);

    } // namespace python
} // namespace aq
