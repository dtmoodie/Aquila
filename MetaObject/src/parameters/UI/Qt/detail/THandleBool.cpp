#ifdef HAVE_QT5
#include "MetaObject/Parameters/UI/Qt/POD.hpp"
#include "MetaObject/Parameters/UI/Qt/SignalProxy.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qcheckbox.h"

using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<bool,void>::THandler():
    chkBox(nullptr),
    boolData(nullptr),
    _currently_updating(false)
{
}

void THandler<bool,void>::UpdateUi( bool* data)
{
    if(data)
    {
        _currently_updating = true;
        chkBox->setChecked(*data);
        _currently_updating = false;
    }
}

void THandler<bool,void>::OnUiUpdate(QObject* sender, int val)
{
    if(_currently_updating)
        return;
    if (sender == chkBox && IHandler::GetParamMtx())
    {
        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
        if (boolData)
        {
            *boolData = chkBox->isChecked();
            if (onUpdate)
                onUpdate();
            if(_listener)
                _listener->OnUpdate(this);
        }                            
    }
}

void THandler<bool,void>::SetData(bool* data_)
{    
    if(IHandler::GetParamMtx())
    {
        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
        boolData = data_;
        if (chkBox)
            UpdateUi(data_);
    }
}

bool* THandler<bool,void>::GetData()
{
    return boolData;
}

std::vector < QWidget*> THandler<bool,void>::GetUiWidgets(QWidget* parent_)
{
    std::vector<QWidget*> output;
    if (chkBox == nullptr)
        chkBox = new QCheckBox(parent_);
    chkBox->connect(chkBox, SIGNAL(stateChanged(int)), IHandler::proxy, SLOT(on_update(int)));
    output.push_back(chkBox);
    return output;
}

bool THandler<bool,void>::UiUpdateRequired()
{
    return false;
}
#endif