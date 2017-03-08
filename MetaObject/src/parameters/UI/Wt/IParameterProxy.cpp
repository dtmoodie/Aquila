#ifdef HAVE_WT
#include <MetaObject/Parameters/UI/Wt/IParameterProxy.hpp>
#include <MetaObject/Parameters/IParameter.hpp>

using namespace mo::UI::wt;
using namespace mo;

IParameterProxy::IParameterProxy(IParameter* param_, MainApplication* app_,
    WContainerWidget *parent_) :
    Wt::WContainerWidget(parent_),
    _app(app_),
    _onUpdateSlot(std::bind(&IParameterProxy::onParameterUpdate, this,
        std::placeholders::_1, std::placeholders::_2))
{
    _onUpdateConnection = param_->RegisterUpdateNotifier(&_onUpdateSlot);
    auto text = new Wt::WText(param_->GetTreeName(), this);
    text->setToolTip(Demangle::TypeToName(param_->GetTypeInfo()));
    this->addWidget(text);
}

IParameterProxy::~IParameterProxy()
{

}

IPlotProxy::IPlotProxy(IParameter *param_, MainApplication *app_, WContainerWidget *parent_):
    Wt::WContainerWidget(parent_),
    _app(app_),
    _onUpdateSlot(std::bind(&IPlotProxy::onParameterUpdate, this,
        std::placeholders::_1, std::placeholders::_2))
{
    _onUpdateConnection = param_->RegisterUpdateNotifier(&_onUpdateSlot);
}
IPlotProxy::~IPlotProxy()
{

}

#endif
