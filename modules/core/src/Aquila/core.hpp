#ifndef AQUILA_CORE_HPP
#define AQUILA_CORE_HPP

#define AQUILA_MODULE "core"
#include <Aquila/detail/export.hpp>
#undef AQUILA_MODULE

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
