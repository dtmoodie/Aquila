#ifdef HAVE_BOOST_PYTHON
#include "MetaObject/Python/Python.hpp"

#include <map>

using namespace mo;

static std::map<std::string, std::function<void(void)>> functions;

void PythonClassRegistry::SetupPythonModule()
{
    for(auto& func : functions)
    {
        func.second();
    }
}

void PythonClassRegistry::RegisterPythonSetupFunction(const char* name, std::function<void(void)> f)
{
    functions[name] = f;
}
#endif