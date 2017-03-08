#pragma once
#include <MetaObject/Parameters/MetaParameter.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>
#include <MetaObject/Parameters/UI/WT.hpp>
#include <MetaObject/Signals/TypedSlot.hpp>
#include <MetaObject/Parameters/Demangle.hpp>
#include <MetaObject/Parameters/IParameter.hpp>

#include <Wt/WContainerWidget>
#include <Wt/WText>

#include <boost/thread/recursive_mutex.hpp>
namespace Wt
{
namespace Chart
{
class WAbstractChart;
}
}
namespace mo
{
class IParameter;
namespace UI
{
namespace wt
{
    class MainApplication;
    class MO_EXPORTS IParameterProxy : public Wt::WContainerWidget
    {
    public:
        IParameterProxy(IParameter* param_, MainApplication* app_,
            WContainerWidget *parent_ = 0);
        virtual ~IParameterProxy();
        virtual void SetTooltip(const std::string& tip) = 0;
    protected:
        template<class T, class E> friend class TDataProxy;
        virtual void onParameterUpdate(mo::Context* ctx, mo::IParameter* param) = 0;
        virtual void onUiUpdate() = 0;
        mo::TypedSlot<void(mo::Context*, mo::IParameter*)> _onUpdateSlot;
        std::shared_ptr<mo::Connection>  _onUpdateConnection;
        MainApplication* _app;
    };

    class MO_EXPORTS IPlotProxy: public Wt::WContainerWidget
    {
    public:
        IPlotProxy(IParameter* param_, MainApplication* app_,
            WContainerWidget *parent_ = 0);

        virtual ~IPlotProxy();
        virtual Wt::Chart::WAbstractChart* GetPlot(){return nullptr;}
    protected:
        template<class T, class E> friend class TDataProxy;
        virtual void onParameterUpdate(mo::Context* ctx, mo::IParameter* param) = 0;
        virtual void onUiUpdate() = 0;
        mo::TypedSlot<void(mo::Context*, mo::IParameter*)> _onUpdateSlot;
        std::shared_ptr<mo::Connection>  _onUpdateConnection;
        MainApplication* _app;
    };

    template<class T, class Enable = void>
    class TDataProxy
    {
    public:
        static const bool IS_DEFAULT = true;
        TDataProxy(){}
        void CreateUi(IParameterProxy* proxy, T* data, bool read_only){}
        void UpdateUi(const T& data){}
        void onUiUpdate(T& data){}
        void SetTooltip(const std::string& tooltip){}
    };



    template<class T, typename Enable = void>
    class TParameterProxy : public IParameterProxy
    {
    public:
        static const bool IS_DEFAULT = TDataProxy<T, void>::IS_DEFAULT;

        TParameterProxy(ITypedParameter<T>* param_, MainApplication* app_,
                        WContainerWidget *parent_ = 0):
            IParameterProxy(param_, app_, parent_),
            _param(param_),
            _data_proxy()
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = param_->GetDataPtr();
            if(ptr)
            {
                _data_proxy.CreateUi(this, ptr, param_->CheckFlags(State_e));
            }
        }
        void SetTooltip(const std::string& tip)
        {
            _data_proxy.SetTooltip(tip);
        }
    protected:
        void onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = _param->GetDataPtr();
            if(ptr)
            {
                _app->getUpdateLock();
                _data_proxy.UpdateUi(*ptr);
                _app->requestUpdate();
            }
        }
        void onUiUpdate()
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = _param->GetDataPtr();
            if(ptr)
            {
                _data_proxy.onUiUpdate(*ptr);
                _param->Commit();
            }
        }
        mo::ITypedParameter<T>* _param;
        TDataProxy<T, void> _data_proxy;
    };

    template<class T, typename Enable = void>
    class TPlotDataProxy
    {
    public:
        static const bool IS_DEFAULT = true;
        TPlotDataProxy(){}
        void CreateUi(Wt::WContainerWidget* container, T* data, bool read_only, const std::string& name = ""){}
        void UpdateUi(const T& data, long long ts){}
        void onUiUpdate(T& data){}
    };

    template<class T, typename Enable = void>
    class TPlotProxy : public IPlotProxy
    {
    public:
        static const bool IS_DEFAULT = TPlotDataProxy<T, void>::IS_DEFAULT;

        TPlotProxy(ITypedParameter<T>* param_, MainApplication* app_,
                        WContainerWidget *parent_ = 0):
            IPlotProxy(param_, app_, parent_),
            _param(param_)
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = param_->GetDataPtr();
            if(ptr)
            {

                if(IPlotProxy* parent = dynamic_cast<IPlotProxy*>(parent_))
                {
                    _data_proxy.CreateUi(parent, ptr, param_->CheckFlags(State_e), param_->GetTreeName());
                }else
                {
                    _data_proxy.CreateUi(this, ptr, param_->CheckFlags(State_e), param_->GetTreeName());
                }
            }
        }
    protected:
        void onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = _param->GetDataPtr();
            if(ptr)
            {
                _app->getUpdateLock();
                _data_proxy.UpdateUi(*ptr, _param->GetTimestamp());
                _app->requestUpdate();
            }
        }
        void onUiUpdate()
        {
            boost::recursive_mutex::scoped_lock param_lock(_param->mtx());
            T* ptr = _param->GetDataPtr();
            if(ptr)
            {
                _data_proxy.onUiUpdate(*ptr);
                _param->Commit();
            }
        }
        mo::ITypedParameter<T>* _param;
        TPlotDataProxy<T, void> _data_proxy;
    };

    template<class T> struct WidgetConstructor
    {

        WidgetConstructor()
        {
            if(!TParameterProxy<T, void>::IS_DEFAULT)
                WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)),
                        std::bind(WidgetConstructor<T>::CreateWidget, std::placeholders::_1,
                                  std::placeholders::_2, std::placeholders::_3));
        }
        static IParameterProxy* CreateWidget(IParameter* param, MainApplication* app, Wt::WContainerWidget* container)
        {
            if (param->GetTypeInfo() == TypeInfo(typeid(T)))
            {
                auto typed = dynamic_cast<ITypedParameter<T>*>(param);
                if (typed)
                {
                     return new TParameterProxy<T, void>(typed, app, container);
                }
            }
            return nullptr;
        }
    };
    template<class T> struct PlotConstructor
    {
        PlotConstructor()
        {
            if(!TParameterProxy<T, void>::IS_DEFAULT)
                WidgetFactory::Instance()->RegisterConstructor(TypeInfo(typeid(T)),
                        std::bind(&PlotConstructor<T>::CreatePlot, std::placeholders::_1,
                                  std::placeholders::_2, std::placeholders::_3));
        }
        static IPlotProxy* CreatePlot(IParameter* param, MainApplication* app, Wt::WContainerWidget* container)
        {
            if (param->GetTypeInfo() == TypeInfo(typeid(T)))
            {
                auto typed = dynamic_cast<ITypedParameter<T>*>(param);
                if (typed)
                {
                        return new TPlotProxy<T, void>(typed, app, container);
                }
            }
            return nullptr;
        }
    };

}
}
#define MO_UI_WT_PARAMTERPROXY_METAPARAMETER(N) \
template<class T> \
struct MetaParameter<T, N, typename std::enable_if<!mo::UI::wt::TParameterProxy<T>::IS_DEFAULT>::type> : public MetaParameter<T, N - 1, void> \
{ \
    static UI::wt::WidgetConstructor<T> _parameter_proxy_constructor; \
    MetaParameter(const char* name): \
        MetaParameter<T, N-1, void>(name) \
    { \
        (void)&_parameter_proxy_constructor; \
    } \
}; \
template<class T> UI::wt::WidgetConstructor<T> MetaParameter<T,N, typename std::enable_if<!mo::UI::wt::TParameterProxy<T>::IS_DEFAULT>::type>::_parameter_proxy_constructor;

MO_UI_WT_PARAMTERPROXY_METAPARAMETER(__COUNTER__)

#define MO_UI_WT_PLOTPROXY_METAPARAMETER(N) \
template<class T> \
struct MetaParameter<T, N, typename std::enable_if<!mo::UI::wt::TPlotProxy<T>::IS_DEFAULT>::type> : public MetaParameter<T, N - 1, void> \
{ \
    static UI::wt::PlotConstructor<T> _parameter_plot_constructor; \
    MetaParameter(const char* name): \
        MetaParameter<T, N-1, void>(name) \
    { \
        (void)&_parameter_plot_constructor; \
    } \
}; \
template<class T> UI::wt::PlotConstructor<T> MetaParameter<T,N, typename std::enable_if<!mo::UI::wt::TPlotProxy<T>::IS_DEFAULT>::type>::_parameter_plot_constructor;

MO_UI_WT_PLOTPROXY_METAPARAMETER(__COUNTER__)
}
