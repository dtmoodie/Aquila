#pragma once
#if defined( HAVE_QT5 ) && defined(HAVE_OPENCV)
#include "POD.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>

#include <qtablewidget.h>
#include "qheaderview.h"

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            template<typename T, typename Enable> class THandler;

            // **********************************************************************************
            // *************************** cv::Rect *********************************************
            // **********************************************************************************
            template<typename T> class THandler<typename ::cv::Rect_<T>, void> : public UiUpdateHandler
            {
                bool _currently_updating;
                THandler<T> _x_handler;
                THandler<T> _y_handler;
                THandler<T> _width_handler;
                THandler<T> _height_handler;
                cv::Rect_<T>* _data;
            public:
                static const bool IS_DEFAULT = false;
                THandler(): _currently_updating(false), _data(nullptr)
                {}

                virtual void UpdateUi( ::cv::Rect_<T>* data)
                {
                    _currently_updating = true;
                    if(data)
                    {
                        _x_handler.UpdateUi(&data->x);
                        _y_handler.UpdateUi(&data->y);
                        _width_handler.UpdateUi(&data->width);
                        _height_handler.UpdateUi(&data->height);
                    }else
                    {
                        _x_handler.UpdateUi(nullptr);
                        _y_handler.UpdateUi(nullptr);
                        _width_handler.UpdateUi(nullptr);
                        _height_handler.UpdateUi(nullptr);
                    }
                    _currently_updating = false;
                }
                virtual void OnUiUpdate(QObject* sender)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    _x_handler.OnUiUpdate(sender);
                    _y_handler.OnUiUpdate(sender);
                    _width_handler.OnUiUpdate(sender);
                    _height_handler.OnUiUpdate(sender);
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(::cv::Rect_<T>* data_)
                {
                    _data = data_;
                    if(data_)
                    {
                        _x_handler.SetData(&data_->x);
                        _y_handler.SetData(&data_->y);
                        _width_handler.SetData(&data_->width);
                        _height_handler.SetData(&data_->height);
                    }
                }
                cv::Rect_<T>* GetData()
                {
                    return _data;;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    std::vector<QWidget*> output;
                    auto out1 = _x_handler.GetUiWidgets(parent);
                    auto out2 = _y_handler.GetUiWidgets(parent);
                    auto out3 = _width_handler.GetUiWidgets(parent);
                    auto out4 = _height_handler.GetUiWidgets(parent);
                    output.insert(output.end(), out1.begin(), out1.end());
                    output.insert(output.end(), out2.begin(), out2.end());
                    output.insert(output.end(), out3.begin(), out3.end());
                    output.insert(output.end(), out4.begin(), out4.end());

                    return output;
                }
                virtual void SetParamMtx(boost::recursive_mutex** mtx)
                {
                    IHandler::SetParamMtx(mtx);
                    _x_handler.SetParamMtx(mtx);
                    _y_handler.SetParamMtx(mtx);
                    _width_handler.SetParamMtx(mtx);
                    _height_handler.SetParamMtx(mtx);
                }
                virtual void SetUpdateListener(UiUpdateListener* listener)
                {
                    _x_handler.SetUpdateListener(listener);
                    _y_handler.SetUpdateListener(listener);
                    _width_handler.SetUpdateListener(listener);
                    _height_handler.SetUpdateListener(listener);
                }
            };
            
            // **********************************************************************************
            // *************************** cv::Range *********************************************
            // **********************************************************************************

            template<> class THandler<cv::Range, void> : public UiUpdateHandler
            {
                THandler<int> _start_handler;
                THandler<int> _end_handler;
                cv::Range* _data;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler() : 
                    _data(nullptr), 
                    _currently_updating(false) 
                {}

                virtual void UpdateUi( cv::Range* data)
                {
                    if(data)
                    {
                        _currently_updating = true;
                        _start_handler.UpdateUi(&data->start);
                        _end_handler.UpdateUi(&data->end);
                        _currently_updating = false;
                    }
                }
                virtual void OnUiUpdate(QObject* sender, int val)
                {
                    if(_currently_updating)
                        return;
                    _start_handler.OnUiUpdate(sender);
                    _end_handler.OnUiUpdate(sender);
                }
                virtual void SetData(cv::Range* data_)
                {    
                    if(IHandler::GetParamMtx())
                    {
                        if(data_)
                        {
                            _data = data_;
                            _start_handler.SetData(&_data->start);
                            _end_handler.SetData(&_data->end);
                        }
                    }
                }
                cv::Range* GetData()
                {
                    return _data;
                }
                virtual std::vector < QWidget*> GetUiWidgets(QWidget* parent_)
                {
                    std::vector<QWidget*> output;
                    auto out1 = _start_handler.GetUiWidgets(parent_);
                    auto out2 = _end_handler.GetUiWidgets(parent_);
                    output.insert(output.end(), out1.begin(), out1.end());
                    output.insert(output.end(), out2.begin(), out2.end());
                    return output;
                }
                static bool UiUpdateRequired()
                {
                    return true;
                }
            };

            // **********************************************************************************
            // *************************** cv::Matx *********************************************
            // **********************************************************************************
            template<typename T, int ROW, int COL> class THandler<typename ::cv::Matx<T, ROW, COL>, void> : public UiUpdateHandler
            {
                QTableWidget* table;
                std::vector<QTableWidgetItem*> items;
                ::cv::Matx<T, ROW, COL>* matData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler() : table(nullptr), matData(nullptr), _currently_updating(false), UiUpdateHandler()
                {
                    
                    table = new QTableWidget();
                    table->horizontalHeader()->hide();
                    table->verticalHeader()->hide();
                    items.reserve(ROW*COL);
                    if(COL == 1)
                    {
                        table->setColumnCount(ROW);
                        table->setRowCount(1);
                    }else
                    {
                        table->setColumnCount(COL);
                        table->setRowCount(ROW);
                    }
                    for (int i = 0; i < ROW; ++i)
                    {
                        for (int j = 0; j < COL; ++j)
                        {
                            if(COL != 1)
                                table->setColumnWidth(j, 40);
                            else
                                table->setColumnWidth(i, 40);
                            QTableWidgetItem* item = new QTableWidgetItem();
                            items.push_back(item);
                            if(COL == 1)
                                table->setItem(0, i, item);
                            else
                                table->setItem(i, j, item);
                        }
                    }
                    
                    proxy->connect(table, SIGNAL(cellChanged(int, int)), proxy, SLOT(on_update(int, int)));
                }
                virtual void SetData(::cv::Matx<T, ROW, COL>* data)
                {
                    matData = data;
                    UpdateUi(data);
                }
                ::cv::Matx<T, ROW, COL>* GetData()
                {
                    return matData;
                }
                virtual void UpdateUi( ::cv::Matx<T, ROW, COL>* data)
                {
                    if(data)
                    {
                        _currently_updating = true;
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        for (int i = 0; i < ROW; ++i)
                        {
                            for (int j = 0; j < COL; ++j)
                            {
                                items[i*COL + j]->setToolTip(QString::number((*data)(i,j)));
                                items[i*COL + j]->setData(Qt::EditRole, (*data)(i, j));
                            }
                        }
                        _currently_updating = false;
                    }                    
                }
                virtual void OnUiUpdate(QObject* sender, int row = -1, int col = -1)
                {
                    if(_currently_updating)
                        return;
                    if (sender == table && IHandler::GetParamMtx())
                    {
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        if (matData)
                        {
                            if (typeid(T) == typeid(float))
                                (*matData)(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toFloat();
                            if(typeid(T) == typeid(double))
                                (*matData)(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toDouble();
                            if(typeid(T) == typeid(int))
                                (*matData)(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toInt();
                            if(typeid(T) == typeid(unsigned int))
                                (*matData)(row, col) = (T)items[row* COL + col]->data(Qt::EditRole).toUInt();
                            if(_listener)
                                _listener->OnUpdate(this);
                        }
                    }
                    else if (row == -1 && col == -1)
                    {
                        if (matData)
                            UpdateUi(matData);
                        if(_listener)
                                _listener->OnUpdate(this);
                    }

                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    std::vector<QWidget*> output;
                    output.push_back(table);
                    return output;
                }
            };
            template<typename T, int ROW> class THandler<typename ::cv::Vec<T, ROW>, void> : public THandler<::cv::Matx<T,ROW,1>>
            {
            };

            template<typename T> class THandler<typename ::cv::Scalar_<T>, void> : public THandler<::cv::Vec<T, 4>>
            {
            };
            // **********************************************************************************
            // *************************** cv::Point_ *********************************************
            // **********************************************************************************
            template<typename T> class THandler<typename ::cv::Point_<T>, void> : public UiUpdateHandler
            {
                QTableWidget* table;
                QTableWidgetItem* first;
                QTableWidgetItem* second;
                ::cv::Point_<T>* ptData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler():ptData(nullptr), _currently_updating(false)
                {
                    
                    table = new QTableWidget();
                    first = new QTableWidgetItem();
                    second = new QTableWidgetItem();
                    table->horizontalHeader()->hide();
                    table->verticalHeader()->hide();
                    table->setItem(0, 0, first);
                    table->setItem(0, 1, second);
                    table->setRowCount(1);
                    table->setColumnCount(2);
                    table->setColumnWidth(0, 40);
                    table->setColumnWidth(1, 40);
                    proxy->connect(table, SIGNAL(cellChanged(int, int)), proxy, SLOT(on_update(int, int)));
                }
                virtual void UpdateUi(::cv::Point_<T>* data)
                {
                    if(data)
                    {
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        if (table)
                        {
                            _currently_updating = true;
                            first = new QTableWidgetItem();
                            second = new QTableWidgetItem();
                            first->setData(Qt::EditRole, ptData->x);
                            second->setData(Qt::EditRole, ptData->y);
                            first->setToolTip(QString::number(ptData->x));
                            second->setToolTip(QString::number(ptData->y));
                            table->setItem(0, 0, first);
                            table->setItem(0, 1, second);
                            table->update();
                            _currently_updating = false;
                        }
                    }                    
                }
                virtual void OnUiUpdate(QObject* sender, int row = -1, int col = -1)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    if (ptData == nullptr)
                        return;
                    if (typeid(T) == typeid(double))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toDouble();
                        ptData->y = (T)second->data(Qt::EditRole).toDouble();
                    }
                    if (typeid(T) == typeid(float))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toFloat();
                        ptData->y = (T)second->data(Qt::EditRole).toFloat();
                    }
                    if (typeid(T) == typeid(int))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toInt();
                        ptData->y = (T)second->data(Qt::EditRole).toInt();
                    }
                    if (typeid(T) == typeid(unsigned int))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toUInt();
                        ptData->y = (T)second->data(Qt::EditRole).toUInt();
                    }
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(::cv::Point_<T>* data_)
                {
                    ptData = data_;
                    UpdateUi(ptData);
                }
                ::cv::Point_<T>* GetData()
                {
                    return ptData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    
                    std::vector<QWidget*> output;
                    output.push_back(table);
                    return output;
                }
            };
            // **********************************************************************************
            // *************************** cv::Point3_ *********************************************
            // **********************************************************************************
            template<typename T> class THandler<typename ::cv::Point3_<T>, void> : public UiUpdateHandler
            {
                QTableWidget* table;
                QTableWidgetItem* first;
                QTableWidgetItem* second;
                QTableWidgetItem* third;
                ::cv::Point3_<T>* ptData;
                bool _updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler():ptData(nullptr)
                {
                    
                    table = new QTableWidget();
                    first = new QTableWidgetItem();
                    second = new QTableWidgetItem();
                    third = new QTableWidgetItem();
                    table->horizontalHeader()->hide();
                    table->verticalHeader()->hide();
                    table->setItem(0, 0, first);
                    table->setItem(0, 1, second);
                    table->setItem(0, 2, third);
                    table->setRowCount(1);
                    table->setColumnCount(3);
                    table->setColumnWidth(0, 40);
                    table->setColumnWidth(1, 40);
                    table->setColumnWidth(2, 40);
                    proxy->connect(table, SIGNAL(cellChanged(int, int)), proxy, SLOT(on_update(int, int)));
                }
                virtual void UpdateUi( ::cv::Point3_<T>* data)
                {
                    if(data)
                    {
                        _updating = true;
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        first = new QTableWidgetItem();
                        second = new QTableWidgetItem();
                        third = new QTableWidgetItem();
                        first->setData(Qt::EditRole, ptData->x);
                        second->setData(Qt::EditRole, ptData->y);
                        third->setData(Qt::EditRole, ptData->z);
                        first->setToolTip(QString::number(ptData->x));
                        second->setToolTip(QString::number(ptData->y));
                        third->setToolTip(QString::number(ptData->z));
                        table->setItem(0, 0, first);
                        table->setItem(0, 1, second);
                        table->setItem(0, 2, third);
                        _updating = false;
                    }                    
                }
                virtual void OnUiUpdate(QObject* sender, int row = -1, int col = -1)
                {
                    if(_updating || !IHandler::GetParamMtx())
                        return;
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    if (ptData == nullptr)
                        return;
                    if (typeid(T) == typeid(double))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toDouble();
                        ptData->y = (T)second->data(Qt::EditRole).toDouble();
                        ptData->z = (T)third->data(Qt::EditRole).toDouble();
                    }
                    if (typeid(T) == typeid(float))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toFloat();
                        ptData->y = (T)second->data(Qt::EditRole).toFloat();
                        ptData->z = (T)third->data(Qt::EditRole).toFloat();
                    }
                    if (typeid(T) == typeid(int))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toInt();
                        ptData->y = (T)second->data(Qt::EditRole).toInt();
                        ptData->z = (T)third->data(Qt::EditRole).toInt();
                    }
                    if (typeid(T) == typeid(unsigned int))
                    {
                        ptData->x = (T)first->data(Qt::EditRole).toUInt();
                        ptData->y = (T)second->data(Qt::EditRole).toUInt();
                        ptData->z = (T)third->data(Qt::EditRole).toUInt();
                    }
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(::cv::Point3_<T>* data_)
                {
                    
                    ptData = data_;
                    UpdateUi(ptData);
                }
                ::cv::Point3_<T>* GetData()
                {
                    return ptData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    
                    std::vector<QWidget*> output;
                    output.push_back(table);
                    return output;
                }
            };
        }
    }
}
#endif
