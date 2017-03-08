#pragma once
#include <MetaObject/Detail/Export.hpp>
#include <MetaObject/Detail/TypeInfo.h>
#include <Wt/WApplication>

namespace mo
{
    template<class T>
    class ITypedParameter;
    class IParameter;
    namespace UI
    {
        namespace wt
        {
            class IParameterProxy;
            class IParameterInputProxy;
            class IParameterOutputProxy;
            class MO_EXPORTS MainApplication : public Wt::WApplication
            {
            public:
                MainApplication(const Wt::WEnvironment& env);
                void requestUpdate();
            private:
                void greet();
                bool _dirty;
                boost::posix_time::ptime _last_update_time;
            };
        }
    }
}
