#pragma once
#include "nodes.hpp"
#include "Aquila/detail/export.hpp"

struct IObjectConstructor;

namespace aq
{
namespace python
{
AQUILA_EXPORTS void setupFrameGrabberInterface();
AQUILA_EXPORTS void setupFrameGrabberObjects(std::vector<IObjectConstructor*>& ctrs);

}
}
