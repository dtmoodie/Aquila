#pragma once
#ifdef HAVE_QT5
#include "POD.hpp"
#include "IHandler.hpp"
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{
    namespace UI
    {
        namespace qt
        {
            struct UiUpdateListener;
            template<class T, typename Enable> class THandler;
            // **********************************************************************************
            // *************************** std::pair ********************************************
            // **********************************************************************************

            template<typename T1> class THandler<std::pair<T1, T1>>  : public IHandler
            {
                std::pair<T1,T1>* pairData;
                THandler<T1> _handler1;
                THandler<T1> _handler2;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler() : 
                    pairData(nullptr), 
                    _currently_updating(false) 
                {
                }

                void UpdateUi( std::pair<T1, T1>* data)
                {
                    _currently_updating = true;
                    if(data)
                    {
                        _handler1.UpdateUi(&data->first);
                        _handler2.UpdateUi(&data->second);
                    }else
                    {
                        _handler1.UpdateUi(nullptr);
                        _handler2.UpdateUi(nullptr);
                    }
                    _currently_updating = false;
                }
                void OnUiUpdate(QObject* sender)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    _handler1.OnUiUpdate(sender);
                    _handler2.OnUiUpdate(sender);
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(std::pair<T1, T1>* data_)
                {
                    pairData = data_;
                    if(data_)
                    {
                        _handler1.SetData(&data_->first);
                        _handler2.SetData(&data_->second);
                    }
                }
                std::pair<T1, T1>* GetData()
                {
                    return pairData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    auto out1 = _handler1.GetUiWidgets(parent);
                    auto out2 = _handler2.GetUiWidgets(parent);
                    out2.insert(out2.end(), out1.begin(), out1.end());
                    return out2;
                }
                virtual void SetParamMtx(boost::recursive_mutex* mtx)
                {
                    IHandler::SetParamMtx(mtx);
                    _handler1.SetParamMtx(mtx);
                    _handler2.SetParamMtx(mtx);
                }
                virtual void SetUpdateListener(UiUpdateListener* listener)
                {
                    _handler1.SetUpdateListener(listener);
                    _handler2.SetUpdateListener(listener);
                }
            };

            template<typename T1, typename T2> class THandler<std::pair<T1, T2>>: public THandler<T1>, public THandler<T2>
            {
                std::pair<T1,T2>* pairData;
            public:
                static const bool IS_DEFAULT = false;
                THandler() : pairData(nullptr) {}

                virtual void UpdateUi( std::pair<T1, T2>* data)
                {
                    
                }
                virtual void OnUiUpdate(QObject* sender)
                {
                    if(IHandler::GetParamMtx())
                    {
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        THandler<T1>::OnUiUpdate(sender);
                        THandler<T2>::OnUiUpdate(sender);
                    }
                }
                virtual void SetData(std::pair<T1, T2>* data_)
                {
                    pairData = data_;
                    THandler<T1>::SetData(&data_->first);
                    THandler<T2>::SetData(&data_->second);
                }
                std::pair<T1, T2>* GetData()
                {
                    return pairData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    
                    auto output = THandler<T1>::GetUiWidgets(parent);
                    auto out2 = THandler<T2>::GetUiWidgets(parent);
                    output.insert(output.end(), out2.begin(), out2.end());
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** std::vector ******************************************
            // **********************************************************************************
            template<typename T> class THandler<std::vector<T>, void> : public THandler < T, void >, public UiUpdateListener
            {
                std::vector<T>* vectorData;
                T _appendData;
                QSpinBox* index;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler(): 
                    index(new QSpinBox()), 
                    vectorData(nullptr), 
                    _currently_updating(false) 
                {
                    THandler<T>::SetUpdateListener(this);
                }
                void UpdateUi( std::vector<T>* data)
                {
                    if (data && data->size())
                    {
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        _currently_updating = true;
                        index->setMaximum(data->size());
                        if(index->value() < data->size())
                            THandler<T>::UpdateUi(&(*data)[index->value()]);
                        else
                            THandler<T>::UpdateUi(&_appendData);
                        _currently_updating = false;
                    }
                }

                void OnUiUpdate(QObject* sender, int idx = 0)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    if (sender == index && vectorData )
                    {
                        if(vectorData->size() && idx < vectorData->size())
                        {
                            boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                            THandler<T>::SetData(&(*vectorData)[idx]);
                            THandler<T>::OnUiUpdate(sender);
                        }else
                        {
                            THandler<T>::SetData(&_appendData);
                            THandler<T>::OnUiUpdate(sender);
                        }   
                    }
                }
                
                void SetData(std::vector<T>* data_)
                {
                    vectorData = data_;
                    if (vectorData)
                    {
                        if (data_->size())
                        {
                            if (index && index->value() < vectorData->size())
                                THandler<T>::SetData(&(*vectorData)[index->value()]);
                        }
                    }
                }
                
                std::vector<T>* GetData()
                {
                    return vectorData;
                }
                
                std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    auto output = THandler<T>::GetUiWidgets(parent);
                    index->setParent(parent);
                    index->setMinimum(0);
                    IHandler::proxy->connect(index, SIGNAL(valueChanged(int)), IHandler::proxy, SLOT(on_update(int)));
                    output.push_back(index);
                    return output;
                }
                
                void OnUpdate(IHandler* handler)
                {
                    if(THandler<T>::GetData() == &_appendData && vectorData)
                    {
                        vectorData->push_back(_appendData);
                        THandler<T>::SetData(&vectorData->back());
                        index->setMaximum(vectorData->size());
                    }
                }
            };
        }
    }
}
#endif
