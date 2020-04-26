#pragma once
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "parameters/Parameter.hpp"
#ifdef _WIN32
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("parametersd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("parameters.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lparameters")
#endif
