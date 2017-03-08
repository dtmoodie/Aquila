#include "MetaObject/Signals/SlotInfo.hpp"
#include <sstream>
std::string mo::SlotInfo::Print()
{
    std::stringstream ss;
    ss << "- " << name << " [" << signature.name() << "]\n";
    if(tooltip.size())
        ss << "  " << tooltip << "\n";
    if(description.size())
        ss << "  " << description << "\n";
    return ss.str();
}
