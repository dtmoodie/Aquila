#ifndef AQUILA_CORE_HPP
#define AQUILA_CORE_HPP

#include <Aquila/core/export.hpp>

#include <string>

namespace mo
{
    class MetaObjectFactory;
}

namespace aq
{
    namespace core
    {
        AQUILA_EXPORTS void initModule(mo::MetaObjectFactory* factory, const std::string& log_dir = "");
    } // namespace aq::core
} // namespace aq

#endif // AQUILA_CORE_HPP
