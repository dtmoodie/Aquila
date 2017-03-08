#include "MetaObject/Parameters/UI/WidgetFactory.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
using namespace mo;
using namespace mo::UI;

#ifdef HAVE_WT
#include "MetaObject/Parameters/UI/WT.hpp"
#include <boost/thread.hpp>
#include <Wt/WApplication>
#include <Wt/WServer>
#include <Wt/WBreak>
#include <Wt/WContainerWidget>
#include <Wt/WLineEdit>
#include <Wt/WPushButton>
#include <Wt/WText>
using namespace Wt;

mo::UI::wt::MainApplication::MainApplication(const WEnvironment& env)
    : WApplication(env)
{
    setTitle("EagleEye Web");                               // application title
    enableUpdates();
}
void mo::UI::wt::MainApplication::requestUpdate()
{
    _dirty = true;
    auto current_time = boost::posix_time::microsec_clock::universal_time();
    if ((current_time - _last_update_time).total_milliseconds() > 15)
    {
        this->triggerUpdate();
    }
}
#endif

struct wt::WidgetFactory::impl
{
    std::map<mo::TypeInfo, wt::WidgetFactory::WidgetConstructor_f> _widget_constructors;
    std::map<mo::TypeInfo, wt::WidgetFactory::PlotConstructor_f> _plot_constructors;
};

wt::WidgetFactory::WidgetFactory()
{
    _pimpl = new impl();
}

wt::WidgetFactory* wt::WidgetFactory::Instance()
{
    static WidgetFactory* g_inst = nullptr;
    if (g_inst == nullptr)
        g_inst = new WidgetFactory();
    return g_inst;
}

wt::IParameterProxy* wt::WidgetFactory::CreateWidget(mo::IParameter* param, MainApplication* app,
                                                     Wt::WContainerWidget* container)
{
    if (param->CheckFlags(mo::Input_e))
        return nullptr;
    if (param->CheckFlags(mo::Output_e))
        return nullptr;
    auto itr = _pimpl->_widget_constructors.find(param->GetTypeInfo());
    if (itr != _pimpl->_widget_constructors.end())
    {
        return itr->second(param, app, container);
    }
    return nullptr;
}

void wt::WidgetFactory::RegisterConstructor(const mo::TypeInfo& dtype,
                                            const WidgetConstructor_f& constructor)
{
    if(_pimpl->_widget_constructors.count(dtype) == 0)
        _pimpl->_widget_constructors[dtype]= constructor;
}

void wt::WidgetFactory::RegisterConstructor(const mo::TypeInfo& type,
                         const PlotConstructor_f& constructor)
{
    if(_pimpl->_plot_constructors.count(type) == 0)
        _pimpl->_plot_constructors[type] = constructor;
}

wt::IPlotProxy* wt::WidgetFactory::CreatePlot(mo::IParameter* param, MainApplication* app,
                              Wt::WContainerWidget* container)
{
    if (param->CheckFlags(mo::Input_e))
        return nullptr;
    if (param->CheckFlags(mo::Output_e))
        return nullptr;
    auto itr = _pimpl->_plot_constructors.find(param->GetTypeInfo());
    if (itr != _pimpl->_plot_constructors.end())
    {
        return itr->second(param, app, container);
    }
    return nullptr;
}
bool wt::WidgetFactory::CanPlot(mo::IParameter* param)
{
    auto itr = _pimpl->_plot_constructors.find(param->GetTypeInfo());
    return itr != _pimpl->_plot_constructors.end();
}
