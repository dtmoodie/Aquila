#include "Aquila/plotting/PlotManager.h"
#include "Aquila/plotting/PlotInfo.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include "RuntimeObjectSystem/IObjectState.hpp"
#include <MetaObject/logging/logging.hpp>
using namespace aq;

PlotManager* PlotManager::Instance()
{
    static PlotManager instance;
    return &instance;
}

rcc::shared_ptr<Plotter> PlotManager::getPlot(const std::string& plotName)
{
    auto pConstructor = mo::MetaObjectFactory::instance()->getConstructor(plotName.c_str());
    // IObjectConstructor* pConstructor =
    // ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(plotName.c_str());
    if (pConstructor && pConstructor->GetInterfaceId() == Plotter::getHash())
    {
        IObject* obj = pConstructor->Construct();
        if (obj)
        {
            obj = static_cast<IObject*>(obj->GetInterface(Plotter::getHash()));
            if (obj)
            {
                Plotter* plotter = dynamic_cast<Plotter*>(obj);
                if (plotter)
                {
                    plotter->Init(true);
                    MO_LOG(info, "[ PlotManager ] successfully generating plot {}", plotName);
                    return rcc::shared_ptr<Plotter>(*plotter);
                }
                else
                {
                    MO_LOG(warn, "[ PlotManager ] failed to cast to plotter object {}", plotName);
                }
            }
            else
            {
                MO_LOG(warn, "[ PlotManager ] incorrect interface {}", plotName);
            }
        }
        else
        {
            MO_LOG(warn, "[ PlotManager ] failed to construct plot {}", plotName);
        }
    }
    else
    {
        MO_LOG(warn, "[ PlotManager ] failed to get constructor {}", plotName);
    }
    return rcc::shared_ptr<Plotter>();
}

std::vector<std::string> PlotManager::getAvailablePlots()
{
    std::vector<std::string> output;
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(Plotter::getHash());

    for (size_t i = 0; i < constructors.size(); ++i)
    {
        output.push_back(constructors[i]->GetName());
    }
    return output;
}

std::vector<std::string> PlotManager::getAcceptablePlotters(mo::IParam* param)
{
    std::vector<std::string> output;
    std::vector<IObjectConstructor*> constructors =
        mo::MetaObjectFactory::instance()->getConstructors(Plotter::getHash());

    for (IObjectConstructor* constructor : constructors)
    {
        const IObjectInfo* object_info = constructor->GetObjectInfo();
        if (object_info)
        {
            const PlotterInfo* plot_info = dynamic_cast<const PlotterInfo*>(object_info);
            if (plot_info)
            {
                if (plot_info->acceptsParameter(param))
                {
                    output.push_back(constructor->GetName());
                }
            }
        }
    }
    return output;
}

bool PlotManager::canPlotParameter(mo::IParam* param)
{
    std::vector<IObjectConstructor*> constructors =
        mo::MetaObjectFactory::instance()->getConstructors(Plotter::getHash());
    // auto constructors = ObjectManager::Instance().GetConstructorsForInterface(IID_Plotter);
    for (IObjectConstructor* constructor : constructors)
    {
        const IObjectInfo* object_info = constructor->GetObjectInfo();
        if (object_info)
        {
            const PlotterInfo* plot_info = dynamic_cast<const PlotterInfo*>(object_info);
            if (plot_info)
            {
                if (plot_info->acceptsParameter(param))
                {
                    return true;
                }
            }
        }
    }
    return false;
}
