#ifdef HAVE_QT5
#include "MetaObject/Parameters/UI/Qt/POD.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include <boost/thread/recursive_mutex.hpp>
#include "qpushbutton.h"
using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

THandler<std::function<void(void)>, void>::THandler() : 
    funcData(nullptr), 
    btn(nullptr) 
{}

void THandler<std::function<void(void)>, void>::UpdateUi(std::function<void(void)>* data)
{
    funcData = data;
}

void THandler<std::function<void(void)>, void>::OnUiUpdate(QObject* sender)
{
    if (sender == btn && IHandler::GetParamMtx())
    {
        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
        if (funcData)
        {
            (*funcData)();
            if (onUpdate)
            {
                onUpdate();
                if(_listener)
                    _listener->OnUpdate(this);
            }
        }
    }
}

void THandler<std::function<void(void)>, void>::SetData(std::function<void(void)>* data_)
{
    funcData = data_;
}

std::function<void(void)>* THandler<std::function<void(void)>, void>::GetData()
{
    return funcData;
}

std::vector<QWidget*> THandler<std::function<void(void)>, void>::GetUiWidgets(QWidget* parent)
{
    std::vector<QWidget*> output;
    if (btn == nullptr)
    {
        btn = new QPushButton(parent);
    }

    btn->connect(btn, SIGNAL(clicked()), proxy, SLOT(on_update()));
    output.push_back(btn);
    return output;
}

#endif