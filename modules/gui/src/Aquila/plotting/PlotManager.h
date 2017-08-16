#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "Plotter.h"
#include <RuntimeObjectSystem/shared_ptr.hpp>

namespace aq
{
    class Plotter;
    class AQUILA_EXPORTS PlotManager
    {
    public:
        static PlotManager* Instance();
        rcc::shared_ptr<Plotter> getPlot(const std::string& plotName);
        std::vector<std::string> getAvailablePlots();
        std::vector<std::string> getAcceptablePlotters(mo::IParam* param);
        bool canPlotParameter(mo::IParam* param);
    };
}