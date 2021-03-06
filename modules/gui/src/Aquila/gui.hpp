#pragma once
#include "Aquila/core/detail/Export.hpp"
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
}
}
