#pragma once
#include "Aquila/Detail/Export.hpp"
#include <string>
#include <boost/log/trivial.hpp>
//#define LOG(severity) BOOST_LOG(severity) << "[" << __FUNCTION__ << "] " 
namespace aq
{
    void AQUILA_EXPORTS SetupLogging(const std::string& log_dir = "");
    void AQUILA_EXPORTS ShutdownLogging();
}
