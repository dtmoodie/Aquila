#include "Aquila/core/Aquila.hpp"
#include <Aquila/core/Logging.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>

void aq::Init(const std::string& log_dir)
{
    aq::SetupLogging(log_dir);
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
}
