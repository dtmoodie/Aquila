#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <vector>

struct IObjectConstructor;
namespace aq
{
namespace python
{
AQUILA_EXPORTS void setupGraphInterface();
AQUILA_EXPORTS void setupGraphObjects(std::vector<IObjectConstructor*>& ctrs);
}
}
