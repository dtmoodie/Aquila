#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/Export.hpp"

#include <qstring.h>

#include <functional>
#include <vector>

namespace boost
{
    class recursive_mutex;
}
class QObject;
class QWidget;

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            class SignalProxy;
            class IHandler;
            struct MO_EXPORTS UiUpdateListener
            {
                virtual void OnUpdate(IHandler* handler) = 0;
            };
            // *****************************************************************************
            //                                IHandler
            // *****************************************************************************
            class MO_EXPORTS IHandler
            {
            private:
                boost::recursive_mutex** paramMtx;
            protected:
                UiUpdateListener* _listener;
                SignalProxy* proxy;
                std::function<void(void)> onUpdate;
            public:
                IHandler();
                virtual ~IHandler();
                virtual void OnUiUpdate(QObject* sender);
                virtual void OnUiUpdate(QObject* sender, double val);
                virtual void OnUiUpdate(QObject* sender, int val);
                virtual void OnUiUpdate(QObject* sender, bool val);
                virtual void OnUiUpdate(QObject* sender, QString val);
                virtual void OnUiUpdate(QObject* sender, int row, int col);
                virtual void SetUpdateListener(UiUpdateListener* listener);
                
                virtual std::function<void(void)>& GetOnUpdate();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent);
                virtual void SetParamMtx(boost::recursive_mutex** mtx);
                virtual boost::recursive_mutex* GetParamMtx();
                static bool UiUpdateRequired();
            };
        }
    }
}
#endif