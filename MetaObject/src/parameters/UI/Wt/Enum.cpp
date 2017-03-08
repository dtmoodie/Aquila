#ifdef HAVE_WT
#include <MetaObject/Parameters/UI/Wt/Enum.hpp>
using namespace mo::UI::wt;
using namespace mo;

TParameterProxy<EnumParameter, void>::TParameterProxy(ITypedParameter<EnumParameter>* param_,
    MainApplication* app_,
    WContainerWidget* parent_) :
    IParameterProxy(param_, app_, parent_),
    _param(param_)
{
    _combo_box = new Wt::WComboBox(this);
    for (auto& name : param_->GetDataPtr()->enumerations)
    {
        _combo_box->addItem(name);
    }
    _combo_box->changed().connect(std::bind(&TParameterProxy<EnumParameter, void>::onUiChanged, this));
}



void TParameterProxy<EnumParameter, void>::SetTooltip(const std::string& tip)
{
    auto lock = _app->getUpdateLock();
    _combo_box->setToolTip(tip);
    _app->requestUpdate();
}

void TParameterProxy<EnumParameter, void>::onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
{
    auto lock = _app->getUpdateLock();
    _combo_box->clear();
    boost::recursive_mutex::scoped_lock param_lock(param->mtx());
    for (auto& name : _param->GetDataPtr()->enumerations)
    {
        _combo_box->addItem(name);
    }
    _app->requestUpdate();
}

void TParameterProxy<EnumParameter, void>::onUiChanged()
{
    boost::recursive_mutex::scoped_lock lock(_param->mtx());
    std::vector<std::string>& enums = _param->GetDataPtr()->enumerations;
    for (int i = 0; i < enums.size(); ++i)
    {
        if (enums[i] == _combo_box->currentText())
        {
            _param->GetDataPtr()->currentSelection = i;
            return;
        }
    }
}

#endif

