#include "ComponentFactory.hpp"

#include <MetaObject/core/SystemTable.hpp>

#include <unordered_map>

namespace aq
{
    struct ComponentFactoryImpl : ComponentFactory
    {
        std::unordered_map<mo::TypeInfo, FactoryFunc_t> m_ctr_map;

        ce::shared_ptr<IComonentProvider> createComponent(mo::TypeInfo type) const override
        {
            auto itr = m_ctr_map.find(type);
            if (itr != m_ctr_map.end())
            {
                return itr->second();
            }
            return {};
        }

        void registerConstructor(mo::TypeInfo type, FactoryFunc_t func) override
        {
            auto itr = m_ctr_map.find(type);
            if (itr == m_ctr_map.end())
            {
                m_ctr_map[type] = std::move(func);
            }
        }
    };

    std::shared_ptr<ComponentFactory> ComponentFactory::instance(SystemTable* table)
    {
        return table->getSingleton<ComponentFactory, ComponentFactoryImpl>();
    }

} // namespace aq