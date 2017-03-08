#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Parameters/UI/WidgetFactory.hpp"
#include "IParameterProxy.hpp"
#include "THandler.hpp"
class QWidget;

namespace mo
{
    class IParameter;
    class Context;
    template<typename T> class ITypedParameter;
    template<typename T> class ITypedRangedParameter;
    template<typename T> class THandler;
    template<typename T, int N, typename Enable> struct MetaParameter;
    namespace UI
    {
        namespace qt
        {
            // **********************************************************************************
            // *************************** ParameterProxy ***************************************
            // **********************************************************************************  
            template<typename T> class ParameterProxy : public IParameterProxy
            {
            public:
                static const bool IS_DEFAULT = THandler<T>::IS_DEFAULT;

                ParameterProxy(IParameter* param);
                ~ParameterProxy();
                
                QWidget* GetParameterWidget(QWidget* parent);
                bool CheckParameter(IParameter* param);
                bool SetParameter(IParameter* param);
            protected:
                void onParamUpdate(Context* ctx, IParameter* param);
                void onParamDelete(IParameter const* param);
                void onUiUpdate();
                THandler<T> paramHandler;
                ITypedParameter<T>* parameter;
            };
            // **********************************************************************************
            // *************************** Constructor ******************************************
            // **********************************************************************************

            template<typename T> class Constructor
            {
            public:
                Constructor()
                {
                    if(!ParameterProxy<T>::IS_DEFAULT)
                        WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)), std::bind(&Constructor<T>::Create, std::placeholders::_1));
                }
                static std::shared_ptr<IParameterProxy> Create(IParameter* param)
                {
                    return std::shared_ptr<IParameterProxy>(new ParameterProxy<T>(param));
                }
            };
        }
    }
#define MO_UI_QT_PARAMTERPROXY_METAPARAMETER(N) \
            template<class T> struct MetaParameter<T, N, typename std::enable_if<!UI::qt::ParameterProxy<T>::IS_DEFAULT, void>::type>: public MetaParameter<T, N-1, void> \
            { \
                static UI::qt::Constructor<T> _parameter_proxy_constructor; \
                MetaParameter(const char* name): \
                    MetaParameter<T, N-1, void>(name) \
                { \
                    (void)&_parameter_proxy_constructor; \
                } \
            }; \
            template<class T> UI::qt::Constructor<T> MetaParameter<T,N, typename std::enable_if<!UI::qt::ParameterProxy<T>::IS_DEFAULT, void>::type>::_parameter_proxy_constructor;

    MO_UI_QT_PARAMTERPROXY_METAPARAMETER(__COUNTER__)
}
#include "detail/TParameterProxyImpl.hpp"
#endif // HAVE_QT5