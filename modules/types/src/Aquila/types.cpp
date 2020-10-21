#include "types.hpp"

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

namespace aq
{
    namespace types
    {
        void initModule(SystemTable* table)
        {
            PerModuleInterface::GetInstance()->SetSystemTable(table);
        }
    } // namespace types
} // namespace aq