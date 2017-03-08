#pragma once

#include "IParameterProxy.hpp"
#include <MetaObject/Parameters/ITypedParameter.hpp>

#include <Wt/WSpinBox>
#include <Wt/WDoubleSpinBox>
#include <Wt/Chart/WCartesianChart>
#include <Wt/Chart/WDataSeries>
#include <Wt/WStandardItemModel>

#include <boost/circular_buffer.hpp>
namespace mo
{
namespace UI
{
namespace wt
{
    template<class T>
    class TDataProxy<T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type>
    {
    public:
        static const bool IS_DEFAULT = false;
        TDataProxy():
            _spin_box(nullptr)
        {

        }
        void CreateUi(IParameterProxy* proxy, T* data, bool read_only)
        {
            _spin_box = new Wt::WSpinBox(proxy);
            _spin_box->setReadOnly(read_only);
            if(data)
            {
                _spin_box->setValue(*data);
                _spin_box->changed().connect(proxy, &IParameterProxy::onUiUpdate);
            }
        }
        void UpdateUi(const T& data)
        {
            _spin_box->setValue(data);
        }
        void onUiUpdate(T& data)
        {
            data = _spin_box->value();
        }
        void SetTooltip(const std::string& tooltip){}
    protected:
        Wt::WSpinBox* _spin_box;
    };
    template<class T>
    class TDataProxy<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
    {
    public:
        static const bool IS_DEFAULT = false;
        TDataProxy():
            _spin_box(nullptr)
        {

        }
        void CreateUi(IParameterProxy* proxy, T* data, bool read_only)
        {
            _spin_box = new Wt::WDoubleSpinBox(proxy);
            _spin_box->setReadOnly(read_only);
            if(data)
            {
                _spin_box->setValue(*data);
                _spin_box->changed().connect(proxy, &IParameterProxy::onUiUpdate);
            }
        }
        void UpdateUi(const T& data)
        {
            _spin_box->setValue(data);
        }
        void onUiUpdate(T& data)
        {
            data = _spin_box->value();
        }
        void SetTooltip(const std::string& tooltip){}
    protected:
        Wt::WDoubleSpinBox* _spin_box;
    };


    template<class T>
    class TPlotDataProxy<T, typename std::enable_if<std::is_floating_point<T>::value ||
            std::is_integral<T>::value>::type>
    {
    public:
        static const bool IS_DEFAULT = false;
        TPlotDataProxy():
            _chart(nullptr),
            _model(nullptr),
            _model_column(0)
        {
            _data_history.set_capacity(1000);

        }
        void CreateUi(Wt::WContainerWidget* container, T* data, bool read_only, const std::string& name = "")
        {
            IPlotProxy* proxy = dynamic_cast<IPlotProxy*>(container);

            if(proxy)
            {
                auto other_chart = dynamic_cast<Wt::Chart::WCartesianChart*>(proxy->GetPlot());
                if(other_chart)
                {
                    _chart = other_chart;
                    _model = dynamic_cast<Wt::WStandardItemModel*>(_chart->model());
                    _model_column = _model->columnCount();
                    _model->appendColumn(std::vector<Wt::WStandardItem*>());
                }
            }
            if(!_chart)
            {
                _chart = new Wt::Chart::WCartesianChart(container);
                _model = new Wt::WStandardItemModel(1000, 2, container);
                _chart->setModel(_model);
                _model_column = 1;
            }
            _model->setHeaderData(_model_column, name);
            _model->setHeaderData(0, Wt::WString("Time"));
            _chart->setLegendEnabled(true);
            _chart->setXSeriesColumn(0);
            _chart->setType(Wt::Chart::ScatterPlot);
            _chart->resize(800, 300);
            _chart->setMargin(10, Wt::Top | Wt::Bottom);
            _chart->setMargin(Wt::WLength::Auto, Wt::Left | Wt::Right);
        }

        void UpdateUi(const T& data, long long ts)
        {
            _data_history.push_back(std::make_pair(data, ts));
            for(int i = 0; i < _data_history.size(); ++i)
            {
                _model->setData(i, 0, _data_history[i].second);
                _model->setData(i, _model_column, _data_history[i].first);
            }
        }
        void onUiUpdate(T& data)
        {

        }

        Wt::Chart::WCartesianChart* _chart;
        boost::circular_buffer<std::pair<T, long long>> _data_history;
        Wt::WStandardItemModel* _model;
        int _model_column;
    };

    template<>
    class MO_EXPORTS TDataProxy<bool, void>
    {
    public:
        static const bool IS_DEFAULT = false;
        TDataProxy();
        void CreateUi(IParameterProxy* proxy, bool* data, bool read_only);
        void UpdateUi(const bool& data);
        void onUiUpdate(bool& data);
        void SetTooltip(const std::string& tooltip);
    protected:
        Wt::WCheckBox* _check_box;
    };


} // namespace wt
} // namespace UI
} // namespace mo
