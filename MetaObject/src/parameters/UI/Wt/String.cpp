#ifdef HAVE_WT
#include <MetaObject/Parameters/UI/Wt/String.hpp>
using namespace mo::UI::wt;
using namespace mo;

TDataProxy<std::string, void>::TDataProxy()
{

}

void TDataProxy<std::string, void>::SetTooltip(const std::string& tp)
{

}

void TDataProxy<std::string, void>::CreateUi(IParameterProxy* proxy, std::string* data, bool read_only)
{
    if(_line_edit)
    {
        delete _line_edit;
        _line_edit = nullptr;
    }
    _line_edit = new Wt::WLineEdit(proxy);
    _line_edit->setReadOnly(read_only);
    if(data)
    {
        _line_edit->setText(*data);
    }
    _line_edit->enterPressed().connect(proxy, &IParameterProxy::onUiUpdate);
}

void TDataProxy<std::string, void>::UpdateUi(const std::string& data)
{
    _line_edit->setText(data);
}

void TDataProxy<std::string, void>::onUiUpdate(std::string& data)
{
    data = _line_edit->text().toUTF8();
}

#endif

