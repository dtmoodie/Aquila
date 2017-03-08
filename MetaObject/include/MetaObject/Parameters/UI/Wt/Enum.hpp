#pragma once

#include "IParameterProxy.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>
#include <MetaObject/Parameters/Types.hpp>

#include <Wt/WComboBox>

namespace mo
{
namespace UI
{
namespace wt
{

    template<>
    class TParameterProxy<EnumParameter, void> : public IParameterProxy
    {
    public:
        TParameterProxy(ITypedParameter<EnumParameter>* param_,
            MainApplication* app_,
            WContainerWidget* parent_ = 0);
    protected:
        void SetTooltip(const std::string& tip);
        void onParameterUpdate(mo::Context* ctx, mo::IParameter* param);
        void onUiChanged();

        ITypedParameter<EnumParameter>* _param;
        Wt::WComboBox* _combo_box;
    };
}
}
}
