#pragma once
#ifdef HAVE_QT5
#include "IHandler.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Parameters/Demangle.hpp"
#include "MetaObject/Parameters/UI/Qt/IHandler.hpp"
class QWidget;
class QObject;
namespace mo
{
    namespace UI
    {
        namespace qt
        {
            // *****************************************************************************
            //                                Unspecialized handler
            // *****************************************************************************
            template<typename T, typename Enable = void>
            class THandler : public IHandler
            {
                T* currentData;
                
            public:
                THandler():
                    currentData(nullptr)
                {
                    LOG(debug) << "Creating handler for default unspecialized parameter " << Demangle::TypeToName(typeid(T));
                }
                virtual void UpdateUi( T* data){}
                virtual void OnUiUpdate(QObject* sender){}
                virtual void SetData(T* data)
                {
                    currentData = data;
                }
                T* GetData()
                {
                    return currentData;
                }
                virtual std::vector<QWidget*> GetUiWidgets(QWidget* parent)
                {
                    LOG(debug) << "Creating widget for default unspecialized parameter " << Demangle::TypeToName(typeid(T));
                    return std::vector<QWidget*>();
                }
                static bool UiUpdateRequired()
                {
                    return false;
                }
                static const bool IS_DEFAULT = true;
            };
        }
    }
}
#endif // HAVE_QT5
