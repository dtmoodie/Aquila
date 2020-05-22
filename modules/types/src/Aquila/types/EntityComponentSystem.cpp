#include "EntityComponentSystem.hpp"
#include <MetaObject/logging/logging.hpp>

namespace aq
{

    EntityComponentSystem::EntityComponentSystem(const EntityComponentSystem& other)
    {
        auto providers = other.m_component_providers;
        for (auto& provider : providers)
        {
            provider.setConst();
        }
        m_component_providers = std::move(providers);
    }

    EntityComponentSystem& EntityComponentSystem::operator=(const EntityComponentSystem& other)
    {
        auto providers = other.m_component_providers;
        for (auto& provider : providers)
        {
            provider.setConst();
        }
        m_component_providers = std::move(providers);
        return *this;
    }

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

    mo::TypeInfo EntityComponentSystem::getComponentType(uint32_t idx) const
    {
        if (idx < m_component_providers.size())
        {
            return m_component_providers[idx]->getComponentType();
        }
        return mo::TypeInfo::Void();
    }

    IComponentProvider* EntityComponentSystem::getProvider(const mo::TypeInfo type)
    {
        for (auto& provider : m_component_providers)
        {
            if (provider->providesComponent(type))
            {
                return provider.get();
            }
        }
        return nullptr;
    }

    const IComponentProvider* EntityComponentSystem::getProvider(mo::TypeInfo type) const
    {
        for (auto& provider : m_component_providers)
        {
            if (provider->providesComponent(type))
            {
                return provider.get();
            }
        }
        return nullptr;
    }

    void EntityComponentSystem::addProvider(ce::shared_ptr<IComponentProvider> provider)
    {
        m_component_providers.push_back(std::move(provider));
    }

    void EntityComponentSystem::erase(uint32_t entity_id)
    {
        for (auto& provider : m_component_providers)
        {
            provider->erase(entity_id);
        }
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

    void EntityComponentSystem::setProviders(std::vector<ce::shared_ptr<IComponentProvider>> providers)
    {
        m_component_providers = providers;
    }

    std::vector<ce::shared_ptr<IComponentProvider>> EntityComponentSystem::getProviders() const
    {
        std::vector<ce::shared_ptr<IComponentProvider>> out;
        for (const auto& provider : m_component_providers)
        {
            auto tmp = provider;
            tmp.setConst();
            out.push_back(tmp);
        }
        return out;
    }

    void EntityComponentSystem::clear()
    {
        for (auto& provider : m_component_providers)
        {
            provider->clear();
        }
    }
} // namespace aq
