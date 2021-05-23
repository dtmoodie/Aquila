#pragma once
#include "Aquila/detail/export.hpp"
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
