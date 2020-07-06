#include "Aquila/plotting/Plotter.h"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <MetaObject/params/IParam.hpp>
#include <RuntimeObjectSystem/ISimpleSerializer.h>
#include <map>
using namespace aq;

void Plotter::Init(bool firstInit)
{
    plotInit(firstInit);
    MetaObject::Init(firstInit);
    if (!firstInit)
    {
        setInput(m_parameter);
        if (m_parameter)
        {
            auto update_slot = this->getSlot_param_updated();
            m_parameter->registerUpdateNotifier(*update_slot);
            // param->RegisterUpdateNotifier(GetSlot<void(mo::IAsyncStream*, mo::IParameter*)>("on_parameter_update"));
        }
    }
}

void Plotter::plotInit(bool firstInit)
{
}

void Plotter::setInput(mo::IParam* param_)
{
    m_parameter = param_;
    if (m_parameter)
    {
        //_connections[&param->update_signal] = param->RegisterNotifier(std::bind(&Plotter::OnParameterUpdate, this,
        // std::placeholders::_1));
        //_connections[&param->delete_signal] = param->RegisterDeleteNotifier(std::bind(&Plotter::on_param_delete, this,
        // std::placeholders::_1));
    }
}

void Plotter::on_parameter_modified(const mo::IParam& param, mo::Header, mo::UpdateFlags, mo::IAsyncStream& ctx)
{
}

void Plotter::on_parameter_delete(mo::IParam const* param)
{
    param = nullptr;
}

void QtPlotter::AddPlot(QWidget* plot_)
{
    plot_widgets.push_back(plot_);
}

#ifdef QT_GUI_LIB
#define Qt5_FOUND
#define HAVE_OPENCV

#include "qgridlayout.h"
#include "qwidget.h"
#include <MetaObject/Parameters/UI/Qt/IParameterProxy.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>

class QtPlotter::impl
{
  public:
    std::map<std::string, std::shared_ptr<mo::UI::qt::IParameterProxy>> parameter_proxies;
    QGridLayout* control_layout;
    QWidget* parent;
};

QWidget* QtPlotter::GetControlWidget(QWidget* parent)
{
    auto parameters = GetParameters();
    if (parameters.size())
    {
        if (_pimpl == nullptr)
        {
            _pimpl.reset(new impl());
        }
        QWidget* control_widget = new QWidget(parent);
        QGridLayout* layout = new QGridLayout();
        control_widget->setLayout(layout);
        for (auto param : parameters)
        {
            auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
            auto param_widget = proxy->GetParameterWidget(parent);
            layout->addWidget(param_widget);
            _pimpl->parameter_proxies[param->GetTreeName()] = proxy;
        }
        _pimpl->control_layout = layout;
        _pimpl->parent = parent;
        return control_widget;
    }
    else
    {
        return nullptr;
    }
}

mo::IParameter* QtPlotter::addParameter(mo::IParameter* param)
{
    // auto ret = Plotter::addParameter(param);
    if (_pimpl)
    {
        auto itr = _pimpl->parameter_proxies.find(param->GetTreeName());
        if (itr == _pimpl->parameter_proxies.end())
        {
            auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
            auto param_widget = proxy->GetParameterWidget(_pimpl->parent);
            _pimpl->control_layout->addWidget(param_widget);
            _pimpl->parameter_proxies[param->GetTreeName()] = proxy;
        }
        else
        {
            itr->second->SetParameter(param);
        }
    }
    // return ret;
    return param;
}

mo::IParameter* QtPlotter::addParameter(std::shared_ptr<mo::IParameter> param)
{
    // auto ret = Plotter::addParameter(param);
    if (_pimpl)
    {
        auto itr = _pimpl->parameter_proxies.find(param->GetTreeName());
        if (itr == _pimpl->parameter_proxies.end())
        {
            auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param.get());
            auto param_widget = proxy->GetParameterWidget(_pimpl->parent);
            _pimpl->control_layout->addWidget(param_widget);
            _pimpl->parameter_proxies[param->GetTreeName()] = proxy;
        }
        else
        {
            itr->second->SetParameter(param.get());
        }
    }
    // return ret;
    return param.get();
}
#else
QWidget* QtPlotter::getControlWidget(QWidget* parent)
{
    return nullptr;
}

void QtPlotter::addParam(std::shared_ptr<mo::IParam> param)
{
}

void QtPlotter::addParam(mo::IParam* param)
{
}
#endif
