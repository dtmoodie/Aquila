#ifndef AQUILA_TYPES_HPP
#define AQUILA_TYPES_HPP

#define AQUILA_MODULE "types"
#include <Aquila/detail/export.hpp>
#undef AQUILA_MODULE

#include <string>

struct SystemTable;
namespace aq
{
    namespace types
    {
        AQUILA_EXPORTS void initModule(SystemTable* table);
    }
} // namespace aq

#endif // AQUILA_TYPES_HPP
