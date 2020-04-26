#pragma once
#include "nodes.hpp"
#include "Aquila/core/detail/Export.hpp"

struct IObjectConstructor;

namespace aq
{
namespace python
{
AQUILA_EXPORTS void setupFrameGrabberInterface();
AQUILA_EXPORTS void setupFrameGrabberObjects(std::vector<IObjectConstructor*>& ctrs);

}
}
