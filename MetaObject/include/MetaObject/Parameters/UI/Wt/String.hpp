#pragma once

#include "IParameterProxy.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>
#include <Wt/WLineEdit>

namespace mo
{
namespace UI
{
namespace wt
{
    template<>
    class MO_EXPORTS TDataProxy<std::string, void>
    {
    public:
        static const int IS_DEFAULT = false;
        TDataProxy();
        void CreateUi(IParameterProxy* proxy, std::string* data, bool read_only);
        void UpdateUi(const std::string& data);
        void onUiUpdate(std::string& data);
        void SetTooltip(const std::string& tp);
    protected:
        Wt::WLineEdit* _line_edit = nullptr;
    };

} // namespace wt
} // namespace UI
} // namespace mo
