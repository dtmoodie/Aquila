#pragma once
#include "IHandler.hpp"

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            class UiUpdateHandler: public IHandler
            {
            public:
                //static const bool UiUpdateRequired = true;
                static bool UiUpdateRequired() 
                {
                    return true;
                }
            };
        }
    }
}