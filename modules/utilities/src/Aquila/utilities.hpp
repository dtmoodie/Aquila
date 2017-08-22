#pragma once

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef WIN32
#ifdef _DEBUG
  RUNTIME_COMPILER_LINKLIBRARY("aquila_utilitiesd.lib")
#else
  RUNTIME_COMPILER_LINKLIBRARY("aquila_utilities.lib")
#endif
#else // Unix
#ifdef NDEBUG
  RUNTIME_COMPILER_LINKLIBRARY("-laquila_utilities")
#else
  RUNTIME_COMPILER_LINKLIBRARY("-laquila_utilitiesd")
#endif
#endif