#pragma once
#ifdef HAVE_QT5
#include "MetaObject/Detail/Export.hpp"
#include "THandler.hpp"
#include "UiUpdateHandler.hpp"
#include "MetaObject/Parameters/UI/Qt/SignalProxy.hpp"

#include <qcombobox.h>
#include <qspinbox.h>

#include <boost/thread/recursive_mutex.hpp>

class QLineEdit;
class QCheckBox;
class QPushButton;
namespace mo
{
    class EnumParameter;
    struct WriteDirectory;
    struct ReadDirectory;
    struct WriteFile;
    struct ReadFile;
    namespace UI
    {
        namespace qt
        {
            

            // **********************************************************************************
            // *************************** Bool ************************************************
            // **********************************************************************************
            template<> class MO_EXPORTS THandler<bool, void> : public UiUpdateHandler
            {
                QCheckBox* chkBox;
                bool* boolData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                virtual void UpdateUi( bool* data);
                virtual void OnUiUpdate(QObject* sender, int val);
                virtual void SetData(bool* data_);
                bool* GetData();
                virtual std::vector < QWidget*> GetUiWidgets(QWidget* parent_);
                static bool UiUpdateRequired();
            };

            // **********************************************************************************
            // *************************** std::string ******************************************
            // **********************************************************************************

            template<> class MO_EXPORTS THandler<std::string, void> : public UiUpdateHandler
            {
                std::string* strData;
                QLineEdit* lineEdit;
                bool _currently_updating = false;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                virtual void UpdateUi( std::string* data);
                virtual void OnUiUpdate(QObject* sender);
                virtual void SetData(std::string* data_);
                std::string* GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent);
            };

            // **********************************************************************************
            // *************************** std::function<void(void)> **************************
            // **********************************************************************************

            template<> class MO_EXPORTS THandler<std::function<void(void)>, void> : public UiUpdateHandler
            {
                std::function<void(void)>* funcData;
                QPushButton* btn;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                void UpdateUi(std::function<void(void)>* data);
                virtual void OnUiUpdate(QObject* sender);
                virtual void SetData(std::function<void(void)>* data_);
                std::function<void(void)>* GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent);
            };

            // **********************************************************************************
            // *************************** floating point data **********************************
            // **********************************************************************************

            template<typename T>
            class THandler<T, typename std::enable_if<std::is_floating_point<T>::value, void>::type> : public UiUpdateHandler
            {
                T* floatData;
                QDoubleSpinBox* box;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                typedef T min_max_type;
                THandler() : box(nullptr), floatData(nullptr), _currently_updating(false) {}
                virtual void UpdateUi( T* data)
                {
                    if(data)
                    {
                        _currently_updating = true;
                        box->setValue(*data);
                        _currently_updating = false;
                    }                    
                }
                virtual void OnUiUpdate(QObject* sender, double val = 0)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    if (sender == box && floatData)
                        *floatData = box->value();
                    if (onUpdate)
                        onUpdate();
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(T* data_)
                {
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    floatData = data_;
                    if (box)
                    {
                        _currently_updating = true;
                        box->setValue(*floatData);
                        _currently_updating = false;
                    }
                        
                }
                T* GetData()
                {
                    return floatData;
                }
                void SetMinMax(T min, T max)
                {
                    box->setMinimum(min);
                    box->setMaximum(max);
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {    
                    std::vector<QWidget*> output;
                    if (box == nullptr)
                    {
                        box = new QDoubleSpinBox(parent);
                        box->setMaximumWidth(100);
                        box->setMinimum(std::numeric_limits<T>::min());
                        box->setMaximum(std::numeric_limits<T>::max());
                    }

                    box->connect(box, SIGNAL(valueChanged(double)), proxy, SLOT(on_update(double)));
                    output.push_back(box);
                    return output;
                }
            };

            // **********************************************************************************
            // *************************** integers *********************************************
            // **********************************************************************************

            template<typename T>
            class THandler<T, typename std::enable_if<std::is_integral<T>::value, void>::type> : public UiUpdateHandler
            {
                T* intData;
                QSpinBox* box;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                typedef T min_max_type;
                THandler() : box(nullptr), intData(nullptr), _currently_updating(false){}
                virtual void UpdateUi( T* data)
                {
                    if(data)
                    {
                        _currently_updating = true;
                        boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                        box->setValue(*data);
                        _currently_updating = false;
                    }                    
                }
                virtual void OnUiUpdate(QObject* sender, int val = -1)
                {
                    if(_currently_updating || !IHandler::GetParamMtx())
                        return;
                    boost::recursive_mutex::scoped_lock lock(*IHandler::GetParamMtx());
                    if (sender == box && intData)
                    {
                        if(val == -1)
                        {
                            *intData = box->value();
                        }else
                        {
                            *intData = val;
                        }
                    }
                    if (onUpdate)
                        onUpdate();
                    if(_listener)
                        _listener->OnUpdate(this);
                }
                virtual void SetData(T* data_)
                {
                    intData = data_;
                    if (box)
                    {
                        _currently_updating = true;
                        box->setValue(*intData);
                        _currently_updating = false;
                    }                        
                }
                T* GetData()
                {
                    return intData;
                }
                void SetMinMax(T min_, T max_)
                {
                    box->setMinimum(min_);
                    box->setMaximum(max_);
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    std::vector<QWidget*> output;
                    if (box == nullptr)
                    {
                        box = new QSpinBox(parent);
                        box->setMaximumWidth(100);
                        if (std::numeric_limits<T>::max() > std::numeric_limits<int>::max())
                            box->setMinimum(std::numeric_limits<int>::max());
                        else
                            box->setMinimum(std::numeric_limits<T>::max());

                        box->setMinimum(std::numeric_limits<T>::min());

                        if (intData)
                            box->setValue(*intData);
                        else
                            box->setValue(0);
                    }

                    box->connect(box, SIGNAL(valueChanged(int)), proxy, SLOT(on_update(int)));
                    box->connect(box, SIGNAL(editingFinished()), proxy, SLOT(on_update()));
                    output.push_back(box);
                    return output;
                }
            };
            // **********************************************************************************
            // *************************** Enums ************************************************
            // **********************************************************************************
            template<> class MO_EXPORTS THandler<EnumParameter, void> : public UiUpdateHandler
            {
                QComboBox* enumCombo;
                EnumParameter* enumData;
                bool _updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                ~THandler();
                virtual void UpdateUi( EnumParameter* data);
                virtual void OnUiUpdate(QObject* sender, int idx);
                virtual void SetData(EnumParameter* data_);
                EnumParameter*  GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent);
            };

            // **********************************************************************************
            // *************************** Files ************************************************
            // **********************************************************************************

            template<> class MO_EXPORTS THandler<WriteDirectory, void> : public UiUpdateHandler
            {
                QPushButton* btn;
                QWidget* parent;
                WriteDirectory* fileData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                virtual void UpdateUi( WriteDirectory* data);
                virtual void OnUiUpdate(QObject* sender);
                virtual void SetData(WriteDirectory* data_);
                WriteDirectory* GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent_);
            };
            template<> class MO_EXPORTS THandler<ReadDirectory, void> : public UiUpdateHandler
            {
                QPushButton* btn;
                QWidget* parent;
                ReadDirectory* fileData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                virtual void UpdateUi( ReadDirectory* data);
                virtual void OnUiUpdate(QObject* sender);
                virtual void SetData(ReadDirectory* data_);
                ReadDirectory* GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent_);
            };
            template<> class MO_EXPORTS THandler<WriteFile, void> : public UiUpdateHandler
            {
                QPushButton* btn;
                QWidget* parent;
                WriteFile* fileData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                virtual void UpdateUi( WriteFile* data);
                virtual void OnUiUpdate(QObject* sender);
                virtual void SetData(WriteFile* data_);
                WriteFile* GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent_);
            };
            template<> class MO_EXPORTS THandler<ReadFile, void> : public UiUpdateHandler
            {
                QPushButton* btn;
                QWidget* parent;
                ReadFile* fileData;
                bool _currently_updating;
            public:
                static const bool IS_DEFAULT = false;
                THandler();
                virtual void UpdateUi( ReadFile* data);
                virtual void OnUiUpdate(QObject* sender);
                virtual void SetData(ReadFile* data_);
                ReadFile* GetData();
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent_);
            };
        }
    }
}
#endif
