#ifndef AQUILA_TYPES_COMPONENT_FACTORY_HPP
#define AQUILA_TYPES_COMPONENT_FACTORY_HPP
#include <Aquila/core/detail/Export.hpp>

#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/detail/TypeInfo.hpp>

#include <ce/shared_ptr.hpp>

namespace aq
{
    struct IComponentProvider;
    struct AQUILA_EXPORTS ComponentFactory
    {
        using FactoryFunc_t = std::function<ce::shared_ptr<IComponentProvider>()>;

        MO_INLINE static std::shared_ptr<ComponentFactory> instance();
        static std::shared_ptr<ComponentFactory> instance(SystemTable*);

        virtual ce::shared_ptr<IComponentProvider> createComponent(mo::TypeInfo type) const = 0;
        virtual void registerConstructor(mo::TypeInfo, FactoryFunc_t) = 0;
    };

    MO_INLINE std::shared_ptr<ComponentFactory> ComponentFactory::instance()
    {
        return mo::singleton<ComponentFactory>();
    }
} // namespace aq

#endif // AQUILA_TYPES_COMPONENT_FACTORY_HPP