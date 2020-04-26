#pragma once
#include <vector>
struct IObjectConstructor;

namespace aq
{
namespace python
{

void setupAlgorithmInterface();
void setupAlgorithmObjects(std::vector<IObjectConstructor*>& ctrs);
}
}
