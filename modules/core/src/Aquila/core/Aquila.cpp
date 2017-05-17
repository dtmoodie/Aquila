#include "Aquila/core/Aquila.hpp"
#include "Aquila/Logging.h"
#include <MetaObject/MetaObjectFactory.hpp>

void aq::Init(const std::string& log_dir)
{
    aq::SetupLogging(log_dir);
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
}
