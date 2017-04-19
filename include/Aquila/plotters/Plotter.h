#pragma once
#include <Aquila/Detail/Export.hpp>
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <IObjectInfo.h>
#include <list>

// Aquila only contains the interface for the plotting mechanisms, actual implementations will be handled inside of
// the plotting plugin

class QCustomPlot;
class QWidget;

namespace mo
{
    class IParameter;
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
    class AQUILA_EXPORTS Plotter : public TInterface<IID_Plotter, mo::IMetaObject>
    {
    public:
        typedef PlotterInfo InterfaceInfo;
        typedef rcc::shared_ptr<Plotter> Ptr;
        typedef rcc::weak_ptr<Plotter>   WeakPtr;

        virtual void Init(bool firstInit);
        virtual void PlotInit(bool firstInit);
        virtual void SetInput(mo::IParameter* param_ = nullptr);
        virtual bool AcceptsParameter(mo::IParameter* param) = 0;

        MO_BEGIN(Plotter)
            MO_SLOT(void, on_parameter_update, mo::Context*, mo::IParameter*);
            MO_SLOT(void, on_parameter_delete, mo::IParameter const*);
            PROPERTY(mo::IParameter*, parameter, nullptr);
        MO_END;
    protected:
    };

    class AQUILA_EXPORTS QtPlotter : public Plotter
    {
    public:
        virtual mo::IParameter* addParameter(mo::IParameter* param);
        virtual mo::IParameter* addParameter(std::shared_ptr<mo::IParameter> param);
        virtual void AddPlot(QWidget* plot_) = 0;

        virtual QWidget* CreatePlot(QWidget* parent) = 0;
        virtual QWidget* GetControlWidget(QWidget* parent);
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
