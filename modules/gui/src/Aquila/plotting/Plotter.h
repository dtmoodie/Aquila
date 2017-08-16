#pragma once
#include <Aquila/core/detail/Export.hpp>
#include <MetaObject/object/IMetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <IObjectInfo.h>
#include <list>

// Aquila only contains the interface for the plotting mechanisms, actual implementations will be handled inside of
// the plotting plugin

class QCustomPlot;
class QWidget;

namespace mo
{
    class IParam;
    class Context;
}
/*
    A plotter object is a proxy that handles plotting a single parameter.  This object is created from
    the factory for a given parameter and it is installed into a plot.  A plot object handles rendering
    all plotters that have been installed in the plot.


*/

namespace aq
{
    class PlotterInfo;
    class AQUILA_EXPORTS Plotter : public TInterface<Plotter, mo::IMetaObject>
    {
    public:
        typedef PlotterInfo InterfaceInfo;
        typedef rcc::shared_ptr<Plotter> Ptr;
        typedef rcc::weak_ptr<Plotter>   WeakPtr;

        virtual void Init(bool firstInit);
        virtual void plotInit(bool firstInit);
        virtual void setInput(mo::IParam* param_ = nullptr);
        virtual bool acceptsParameter(mo::IParam* param) = 0;

        MO_BEGIN(Plotter)
            MO_SLOT(void, on_parameter_update, mo::Context*, mo::IParam*);
            MO_SLOT(void, on_parameter_delete, mo::IParam const*);
            PROPERTY(mo::IParam*, parameter, nullptr);
        MO_END;
    protected:
    };

    class AQUILA_EXPORTS QtPlotter : public Plotter
    {
    public:
        virtual mo::IParam* addParam(mo::IParam* param);
        virtual mo::IParam* addParam(std::shared_ptr<mo::IParam> param);
        virtual void AddPlot(QWidget* plot_) = 0;

        virtual QWidget* createPlot(QWidget* parent) = 0;
        virtual QWidget* getControlWidget(QWidget* parent);
    protected:
        std::list<QWidget*> plot_widgets;
    public:
        class impl;
        std::shared_ptr<impl> _pimpl;
    };
    class AQUILA_EXPORTS vtkPlotter: public Plotter
    {
    public:

    protected:

    private:
    };

}
