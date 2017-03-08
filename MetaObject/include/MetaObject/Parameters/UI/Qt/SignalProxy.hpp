#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/Export.hpp"
#include <qobject.h>
#include <qdatetime.h>

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            class IHandler;
            // *****************************************************************************
            //                                SignalProxy
            // *****************************************************************************
            class MO_EXPORTS SignalProxy : public QObject
            {
                Q_OBJECT
                IHandler* handler;
                QTime lastCallTime;
            public:
                SignalProxy(IHandler* handler_);

            public slots:
                void on_update();
                void on_update(int);
                void on_update(double);
                void on_update(bool);
                void on_update(QString);
                void on_update(int row, int col);
            };
        }
    }
}
#endif