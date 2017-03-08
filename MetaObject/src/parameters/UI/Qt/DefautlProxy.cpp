
#include "MetaObject/Parameters/UI/Qt/DefaultProxy.hpp"
#include "MetaObject/Parameters/IParameter.hpp"

using namespace mo;
using namespace mo::UI::qt;

DefaultProxy::DefaultProxy(IParameter* param)
{
    delete_slot = std::bind(&DefaultProxy::onParamDelete, this, std::placeholders::_1);
    update_slot = std::bind(&DefaultProxy::onParamUpdate, this, std::placeholders::_1, std::placeholders::_2);
    parameter = param;
    param->RegisterUpdateNotifier(&update_slot);
    param->RegisterDeleteNotifier(&delete_slot);
}
bool DefaultProxy::SetParameter(IParameter* param)
{
    parameter = param;
    param->RegisterUpdateNotifier(&update_slot);
    param->RegisterDeleteNotifier(&delete_slot);
    return true;
}
bool DefaultProxy::CheckParameter(IParameter* param)
{    
    return param == parameter;
}
#ifdef HAVE_QT5
#include <qgridlayout.h>
#include <qlabel.h>
#include <qstring.h>
#endif
QWidget* DefaultProxy::GetParameterWidget(QWidget* parent)
{    
#ifdef HAVE_QT5
    QWidget* output = new QWidget(parent);

    QGridLayout* layout = new QGridLayout(output);
    QLabel* nameLbl = new QLabel(QString::fromStdString(parameter->GetName()), output);
    nameLbl->setToolTip(QString::fromStdString(parameter->GetTypeInfo().name()));
    layout->addWidget(nameLbl, 0, 0);
    output->setLayout(layout);
    return output;
#endif
	return nullptr;
}

void DefaultProxy::onUiUpdate()
{
}

void DefaultProxy::onParamUpdate(Context* ctx, IParameter* param)
{

}

void DefaultProxy::onParamDelete(IParameter const* param)
{
    parameter = nullptr;
}