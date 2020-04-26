#include "core.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObjectFactory.cpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

namespace aq
{
namespace core
{

void initModule(mo::MetaObjectFactory* factory, const std::string& log_dir)
{
    // TODO setup logging to a log dir
    factory->setupObjectConstructors(PerModuleInterface::GetInstance());
}
}
}
