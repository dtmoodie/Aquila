#pragma once

#include "MetaObject/Detail/Export.hpp"
#include <memory>
class QWidget;

namespace mo
{
    class IParameter;
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                IParameterProxy
            // *****************************************************************************
            class MO_EXPORTS IParameterProxy
            {
            protected:
            public:
                typedef std::shared_ptr<IParameterProxy> Ptr;
                IParameterProxy();
                virtual ~IParameterProxy();
                
                virtual QWidget* GetParameterWidget(QWidget* parent) = 0;
                virtual bool CheckParameter(IParameter* param) = 0;
                virtual bool SetParameter(IParameter* param) = 0;
            };
        }
    }
}