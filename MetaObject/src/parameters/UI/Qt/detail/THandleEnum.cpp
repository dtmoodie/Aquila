#ifdef HAVE_QT5
#include "MetaObject/Parameters/UI/Qt/POD.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<EnumParameter, void>::THandler() : enumCombo(nullptr), _updating(false){}
THandler<EnumParameter, void>::~THandler()
{

}

void THandler<EnumParameter, void>::UpdateUi( EnumParameter* data)
{
    if(_updating)
        return;
    if(data)
    {
        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
        _updating = true;
        enumCombo->clear();
        for (int i = 0; i < data->enumerations.size(); ++i)
        {
            enumCombo->addItem(QString::fromStdString(data->enumerations[i]));
        }
        enumCombo->setCurrentIndex(data->currentSelection);
        _updating = false;
    }                    
}

void THandler<EnumParameter, void>::OnUiUpdate(QObject* sender, int idx)
{
    if(_updating || !IHandler::GetParamMtx())
        return;
    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
    if (idx != -1 && sender == enumCombo && enumData)
    {
        if(enumData->currentSelection == idx)
            return;
        _updating = true;
        enumData->currentSelection = idx;
        if (onUpdate)
            onUpdate();
        if(_listener)
            _listener->OnUpdate(this);
        _updating = false;
    }
}

void THandler<EnumParameter, void>::SetData(EnumParameter* data_)
{
    enumData = data_;
    if (enumCombo)
        UpdateUi(enumData);
}

EnumParameter*  THandler<EnumParameter, void>::GetData()
{
    return enumData;
}

std::vector<QWidget*> THandler<EnumParameter, void>::GetUiWidgets(QWidget* parent)
{

    std::vector<QWidget*> output;
    if (enumCombo == nullptr)
        enumCombo = new QComboBox(parent);
    enumCombo->connect(enumCombo, SIGNAL(currentIndexChanged(int)), proxy, SLOT(on_update(int)));
    output.push_back(enumCombo);
    return output;
}
#endif