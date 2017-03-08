#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <string>
#include <ostream>

namespace mo
{
    class MO_EXPORTS Demangle
    {
    public:
        static std::string TypeToName(TypeInfo type);
        static void RegisterName(TypeInfo type, const char* name);
        static void RegisterType(TypeInfo type);
        // This returns a type map that is stored in cereal binary format that
        // can be used to reconstruct this database
        static void GetTypeMapBinary(std::ostream& stream);
        static void SaveTypeMap(const std::string& filename);
    };
}