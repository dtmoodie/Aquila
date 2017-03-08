/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <memory>
#include <functional>
class QWidget;
namespace Wt
{
    class WContainerWidget;
}
namespace mo
{
class TypeInfo;
class IParameter;
namespace UI
{
namespace qt
{
    class IHandler;
    class IParameterProxy;
    // *****************************************************************************
    //                                WidgetFactory
    // *****************************************************************************

    class MO_EXPORTS WidgetFactory
    {
    public:
        typedef std::function<std::shared_ptr<IParameterProxy>(IParameter*)> HandlerConstructor_f;

        static WidgetFactory* Instance();

        void RegisterConstructor(const TypeInfo& type, HandlerConstructor_f f);
        std::shared_ptr<IParameterProxy> CreateProxy(IParameter* param);
    private:
        WidgetFactory();
        struct impl;
        impl* _pimpl;
    };
} /* namespace qt */
namespace wt
{
class MainApplication;
class IParameterProxy;
class IPlotProxy;
class MO_EXPORTS WidgetFactory
{
public:
    enum WidgetType
    {
        Control,
        Display
    };

    typedef std::function<IParameterProxy*(mo::IParameter*, MainApplication*,
                                           Wt::WContainerWidget*)> WidgetConstructor_f;
    typedef std::function<IPlotProxy*(mo::IParameter*, MainApplication*,
                                           Wt::WContainerWidget*)> PlotConstructor_f;

    static WidgetFactory* Instance();
    IParameterProxy* CreateWidget(mo::IParameter* param, MainApplication* app,
                                  Wt::WContainerWidget* container = nullptr);
    bool CanPlot(mo::IParameter* param);
    IPlotProxy* CreatePlot(mo::IParameter* param, MainApplication* app,
                                  Wt::WContainerWidget* container = nullptr);

    void RegisterConstructor(const mo::TypeInfo& type,
                             const WidgetConstructor_f& constructor);

    void RegisterConstructor(const mo::TypeInfo& type,
                             const PlotConstructor_f& constructor);
private:
    WidgetFactory();
    struct impl;
    impl* _pimpl;
};

} /* namespace wt */

} /* namespace UI */
} /* namespace Perameters */
