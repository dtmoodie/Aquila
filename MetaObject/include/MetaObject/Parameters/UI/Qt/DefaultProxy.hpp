#pragma once
#include "IParameterProxy.hpp"
#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"
namespace mo
{
    class Context;
    class IParameter;
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                DefaultProxy
            // *****************************************************************************
            class MO_EXPORTS DefaultProxy: public IParameterProxy
            {
            public:
                DefaultProxy(IParameter* param);
                virtual bool CheckParameter(IParameter* param);
                bool SetParameter(IParameter* param);
                QWidget* GetParameterWidget(QWidget* parent);
            protected:
                IParameter* parameter;
                TypedSlot<void(IParameter const*)> delete_slot;
                TypedSlot<void(Context*, IParameter*)> update_slot;
                virtual void onUiUpdate();
                virtual void onParamUpdate(Context*, IParameter*);
                virtual void onParamDelete(IParameter const*);
            };
        }
    }
}