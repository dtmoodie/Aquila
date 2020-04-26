#pragma once

#include <Aquila/detail/export.hpp>

struct SystemTable;
namespace aq
{
    namespace serialization
    {
        AQUILA_EXPORTS void initModule(SystemTable* table);
    }
} // namespace aq
