#include "EntityComponentSystem.hpp"
#include <MetaObject/logging/logging.hpp>

namespace aq
{
    uint32_t EntityComponentSystem::getNumComponents() const
    {
        return m_component_providers.size();
    }

    uint32_t EntityComponentSystem::getNumEntities() const
    {
        if (m_component_providers.empty())
        {
            return 0;
        }
        boost::optional<uint32_t> num_entities;
        for (const auto& provider : m_component_providers)
        {
            const auto entities = provider->getNumEntities();
            if (num_entities == boost::none)
            {
                num_entities = entities;
            }
            else
            {
                MO_ASSERT(*num_entities == entities);
            }
        }

        return *num_entities;
    }

    const std::type_info* EntityComponentSystem::getComponentType(uint32_t idx) const
    {
        if (idx < m_component_providers.size())
        {
            return m_component_providers[idx]->getComponentType(0);
        }
        return nullptr;
    }

    ct::ext::IComponentProvider* EntityComponentSystem::getProvider(const std::type_info& type)
    {
        for (auto& provider : m_component_providers)
        {
            if (provider->providesComponent(type))
            {
                return provider->getProvider(type);
            }
        }
        return nullptr;
    }

    const ct::ext::IComponentProvider* EntityComponentSystem::getProvider(const std::type_info& type) const
    {
        for (auto& provider : m_component_providers)
        {
            if (provider->providesComponent(type))
            {
                return provider->getProvider(type);
            }
        }
        return nullptr;
    }

    void EntityComponentSystem::addProvider(ce::shared_ptr<IDynamicProvider> provider)
    {
        m_component_providers.push_back(std::move(provider));
    }

    uint32_t EntityComponentSystem::append()
    {
        uint32_t num_entities = 0;
        for (auto& provider : m_component_providers)
        {
            const uint32_t num_entities_ = provider->getNumEntities();
            if (num_entities == 0)
            {
                num_entities = num_entities_;
            }
            else
            {
                MO_ASSERT_EQ(num_entities_, num_entities);
            }
            provider->resize(num_entities + 1);
        }
        return num_entities;
    }
} // namespace aq