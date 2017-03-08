#pragma once
#ifdef HAVE_QT5
#include <MetaObject/Parameters/IParameter.hpp>
#include "qwidget.h"
#include "qgridlayout.h"
#include "qpushbutton.h"
#include "qlabel.h"

namespace mo
{
    template<class T> class ITypedParameter;
    template<class T> class ITypedRangedParameter;
    namespace UI
    {
        namespace qt
        {
            template <typename T>
            class has_minmax
            {
                typedef char one;
                typedef long two;

                template <typename C> static one test(decltype(&C::SetMinMax));
                template <typename C> static two test(...);

            public:
                enum { value = sizeof(test<T>(0)) == sizeof(char) };
            };

            template<typename T> 
            void SetMinMax(typename std::enable_if<has_minmax<THandler<T>>::value, THandler<T>>::type& handler, ITypedParameter<T>* param)
            {
                /*auto rangedParam = dynamic_cast<ITypedRangedParameter<T>*>(param);
                if (rangedParam)
                {
                    typename Handler<T>::min_max_type min, max;
                    rangedParam->GetRange(min, max);
                    handler.SetMinMax(min, max);
                }*/
            }

            template<typename T> 
            void SetMinMax(typename std::enable_if<!has_minmax<THandler<T>>::value, THandler<T>>::type& handler, ITypedParameter<T>* param)
            {

            }
            
            template<typename T> 
            ParameterProxy<T>::~ParameterProxy()
            {
                //InvalidCallbacks::invalidate((void*)&paramHandler);
            }

            template<typename T> 
            void ParameterProxy<T>::onUiUpdate()
            {
                //TODO Notify parameter of update on the processing thread.
                parameter->modified = true;
                parameter->OnUpdate(nullptr);
            }
            
            // Guaranteed to be called on the GUI thread thanks to the signal connection configuration
            template<typename T> 
            void ParameterProxy<T>::onParamUpdate(Context* ctx, IParameter* param)
            {
                auto dataPtr = parameter->Data();    
                if (dataPtr)
                {
                    if (THandler<T>::UiUpdateRequired())
                    {
                        paramHandler.UpdateUi(dataPtr);
                    }
                }
            }
            
            template<typename T> 
            void ParameterProxy<T>::onParamDelete(IParameter const* param)
            {
                if(param == parameter)
                {
                    parameter = nullptr;
                    paramHandler.SetParamMtx(nullptr);
                }
            }

            template<typename T> 
            ParameterProxy<T>::ParameterProxy(IParameter* param)
            {
                SetParameter(param);
            }
            
            template<typename T> 
            bool ParameterProxy<T>::CheckParameter(IParameter* param)
            {
                return param == parameter;
            }
            
            template<typename T> 
            QWidget* ParameterProxy<T>::GetParameterWidget(QWidget* parent)
            {
                QWidget* output = new QWidget(parent);
                auto widgets = paramHandler.GetUiWidgets(output);
                SetMinMax<T>(paramHandler, parameter);
                QGridLayout* layout = new QGridLayout(output);
                if (parameter->GetTypeInfo() == TypeInfo(typeid(std::function<void(void)>)))
                {
                    dynamic_cast<QPushButton*>(widgets[0])->setText(QString::fromStdString(parameter->GetName()));
                    layout->addWidget(widgets[0], 0, 0);
                }
                else
                {
                    QLabel* nameLbl = new QLabel(QString::fromStdString(parameter->GetName()), output);
                    nameLbl->setToolTip(QString::fromStdString(parameter->GetTypeInfo().name()));
                    layout->addWidget(nameLbl, 0, 0);
                    int count = 1;
                    output->setLayout(layout);
                    for (auto itr = widgets.rbegin(); itr != widgets.rend(); ++itr, ++count)
                    {
                        layout->addWidget(*itr, 0, count);
                    }
                    // Correct the tab order of the widgets
                    for(size_t i = widgets.size() - 1; i > 0; --i)
                    {
                        QWidget::setTabOrder(widgets[i], widgets[i - 1]);
                    }
                    paramHandler.UpdateUi(parameter->GetDataPtr());
                }
                return output;
            }

            template<typename T> 
            bool ParameterProxy<T>::SetParameter(IParameter* param)
            {
                if(param->GetTypeInfo() != TypeInfo(typeid(T)))
                    return false;
                auto typedParam = dynamic_cast<ITypedParameter<T>*>(param);
                if (typedParam)
                {
                    parameter = typedParam;
                    parameter->mtx();
                    paramHandler.SetParamMtx(&parameter->_mtx);
                    paramHandler.SetData(parameter->GetDataPtr());
                    paramHandler.IHandler::GetOnUpdate() = std::bind(&ParameterProxy<T>::onUiUpdate, this);
                    //connection = parameter->update_signal.connect(std::bind(&ParameterProxy<T>::onParamUpdate, this, std::placeholders::_1), Signals::GUI, true, this);
                    //delete_connection = parameter->delete_signal.connect(std::bind(&ParameterProxy<T>::onParamDelete, this));
                    return true;
                }
                return false;
            }
        }
    }
}
#endif // HAVE_QT5