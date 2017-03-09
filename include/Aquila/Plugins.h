#include "Nodes/Node.h"

namespace aq
{
    bool AQUILA_EXPORTS loadPlugin(const std::string& fullPluginPath);
    std::vector<std::string> AQUILA_EXPORTS ListLoadedPlugins();
}
