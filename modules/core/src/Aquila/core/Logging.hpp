#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <string>

namespace aq
{
    void AQUILA_EXPORTS SetupLogging(const std::string& log_dir = "");
    void AQUILA_EXPORTS ShutdownLogging();
}
