#pragma once
#include "Aquila/detail/export.hpp"
#include <Aquila/gui/UiCallbackHandlers.h>

#include <boost/thread.hpp>
namespace mo
{
    class MetaObjectFactory;
}
namespace aq
{
    namespace gui
    {
        AQUILA_EXPORTS void initModule(mo::MetaObjectFactory* factory);
        void AQUILA_EXPORTS guiThreadFunc();
        boost::thread AQUILA_EXPORTS createGuiThread();
    } // namespace gui
} // namespace aq
