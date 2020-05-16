#ifndef AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP
#define AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP
#include <Aquila/core/detail/Export.hpp>
#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/logging/logging.hpp>

#include <ct/extensions/DataTable.hpp>
#include <ct/static_asserts.hpp>

#include <ce/shared_ptr.hpp>
#include <memory>
namespace aq
{
    struct AQUILA_EXPORTS IComonentProvider
    {
        virtual ~IComonentProvider() = default;
        virtual void resize(uint32_t) = 0;
        virtual void erase(uint32_t) = 0;
        virtual void clear() = 0;

        virtual size_t getNumEntities() const = 0;

        virtual bool providesComponent(mo::TypeInfo info) const = 0;
        virtual mo::TypeInfo getComponentType() const = 0;
    };

    template <class T>
    struct TComponentProvider : IComonentProvider
    {
        void resize(uint32_t) override;
        void erase(uint32_t) override;
        void clear() override;

        bool providesComponent(mo::TypeInfo info) const override;

        ct::TArrayView<T> getComponentMutable();
        ct::TArrayView<const T> getComponent() const;

        mo::TypeInfo getComponentType() const override;

        size_t getNumEntities() const override;

      private:
        ct::ext::DataTableStorage<T> m_data;
    };
    template <class T, class E = void>
    struct TEntityComponentSystem;
    struct AQUILA_EXPORTS EntityComponentSystem
    {

        EntityComponentSystem() = default;
        EntityComponentSystem(const EntityComponentSystem& other);
        EntityComponentSystem(EntityComponentSystem&& other) = default;
        EntityComponentSystem& operator=(EntityComponentSystem&& other) = default;
        EntityComponentSystem& operator=(const EntityComponentSystem& other);

        uint32_t getNumComponents() const;
        mo::TypeInfo getComponentType(uint32_t idx) const;

        IComonentProvider* getProvider(mo::TypeInfo type);
        const IComonentProvider* getProvider(mo::TypeInfo type) const;

        void addProvider(ce::shared_ptr<IComonentProvider>);
        void setProviders(std::vector<ce::shared_ptr<IComonentProvider>> providers);

        std::vector<ce::shared_ptr<IComonentProvider>>
        getProviders(std::vector<ce::shared_ptr<IComonentProvider>> providers) const;

        uint32_t getNumEntities() const;

        template <class T>
        void push_back(const T& obj)
        {
            using Components_t = typename ct::ext::SelectComponents<typename ct::GlobMemberObjects<T>::types>::type;
            using Objects_t = typename ct::GlobMemberObjects<T>::types;
            ct::StaticEqualTypes<Components_t, Objects_t>{};
            const uint32_t new_id = append();
            const auto idx = ct::Reflect<T>::end();
            pushRecurse(obj, new_id, idx);
        }

        template <class T>
        T get(uint32_t id) const
        {
            using Components_t = typename ct::ext::SelectComponents<typename ct::GlobMemberObjects<T>::types>::type;
            using Objects_t = typename ct::GlobMemberObjects<T>::types;
            ct::StaticEqualTypes<Components_t, Objects_t>{};
            // Assemble an object by copying component data out
            T out;
            const auto idx = ct::Reflect<T>::end();
            getRecurse(out, id, idx);
            return out;
        }

        template <class T>
        ct::TArrayView<T> getComponentMutable()
        {
            ct::TArrayView<T> view;
            auto provider = getProvider(mo::TypeInfo::create<T>());
            if (provider)
            {
                auto typed = static_cast<TComponentProvider<T>*>(provider);
                view = typed->getComponentMutable();
            }

            return view;
        }

        template <class T>
        ct::TArrayView<const T> getComponent() const
        {
            ct::TArrayView<const T> view;
            auto provider = getProvider(mo::TypeInfo::create<T>());
            if (provider)
            {
                auto typed = static_cast<const TComponentProvider<T>*>(provider);
                view = typed->getComponent();
            }
            return view;
        }

        void erase(uint32_t entity_id);
        void clear();

      private:
        std::vector<ce::shared_ptr<IComonentProvider>> m_component_providers;

        uint32_t append();

        template <class T, ct::index_t I>
        void pushImpl(const T& obj, const uint32_t id, ct::Indexer<I> field_index)
        {
            auto ptr = ct::Reflect<T>::getPtr(field_index);
            using Component_t = typename decltype(ptr)::Data_t;
            auto provider = getProvider(mo::TypeInfo::create<Component_t>());
            if (!provider)
            {
                MO_ASSERT_EQ(id, 0);
                // create a provider
                auto new_provider = ce::shared_ptr<TComponentProvider<Component_t>>::create();
                new_provider->resize(1);
                addProvider(ce::shared_ptr<IComonentProvider>(std::move(new_provider)));
                provider = getProvider(mo::TypeInfo::create<Component_t>());
            }
            MO_ASSERT(provider);
            auto typed = static_cast<TComponentProvider<Component_t>*>(provider);
            MO_ASSERT(typed);
            ct::TArrayView<Component_t> view = typed->getComponentMutable();

            MO_ASSERT(view.size());
            MO_ASSERT(id < view.size());
            view[id] = ptr.get(obj);
        }

        template <class T>
        void pushRecurse(const T& obj, const uint32_t entity_id, ct::Indexer<0> field_index)
        {
            pushImpl(obj, entity_id, field_index);
        }

        template <class T, ct::index_t I>
        void pushRecurse(const T& obj, const uint32_t entity_id, ct::Indexer<I> field_index)
        {
            pushImpl(obj, entity_id, field_index);
            const auto next_field = --field_index;
            pushRecurse(obj, entity_id, next_field);
        }

        template <class T, ct::index_t I>
        void getImpl(T& obj, const uint32_t entity_id, ct::Indexer<I> field_index) const
        {
            auto ptr = ct::Reflect<T>::getPtr(field_index);
            using Component_t = typename decltype(ptr)::Data_t;
            auto provider = getProvider(mo::TypeInfo::create<Component_t>());
            MO_ASSERT(provider);
            auto typed = static_cast<const TComponentProvider<Component_t>*>(provider);
            MO_ASSERT(typed);
            ct::TArrayView<const Component_t> view = typed->getComponent();

            MO_ASSERT(entity_id < view.size());
            ptr.set(obj, view[entity_id]);
        }

        template <class T>
        void getRecurse(T& obj, const uint32_t entity_id, ct::Indexer<0> field_index) const
        {
            getImpl(obj, entity_id, field_index);
        }

        template <class T, ct::index_t I>
        void getRecurse(T& obj, const uint32_t entity_id, ct::Indexer<I> field_index) const
        {
            getImpl(obj, entity_id, field_index);
            const auto next_field = --field_index;
            getRecurse(obj, entity_id, next_field);
        }
    };

    template <class T>
    void addComponents(EntityComponentSystem& ecs, ct::VariadicTypedef<T>)
    {
        auto ptr = ce::make_shared<TComponentProvider<T>>();
        MO_ASSERT(ptr);
        ecs.addProvider(std::move(ptr));
    }

    template <class T, class... Ts>
    void addComponents(EntityComponentSystem& ecs,
                       ct::VariadicTypedef<T, Ts...>,
                       ct::EnableIf<sizeof...(Ts) != 0, int32_t> = 0)
    {
        auto ptr = ce::make_shared<TComponentProvider<T>>();
        MO_ASSERT(ptr);
        ecs.addProvider(std::move(ptr));
        addComponents(ecs, ct::VariadicTypedef<Ts...>());
    }

    template <class T, class E>
    struct TEntityComponentSystem : EntityComponentSystem
    {
        template <class U>
        TEntityComponentSystem(const TEntityComponentSystem<U, void>& other)
            : EntityComponentSystem(other)
        {
        }

        TEntityComponentSystem()
        {
            addComponents(*this, ct::VariadicTypedef<T>{});
        }
    };

    template <class T>
    struct TEntityComponentSystem<T, ct::EnableIfReflected<T>> : EntityComponentSystem
    {
        template <class U>
        TEntityComponentSystem(const TEntityComponentSystem<U, void>& other)
            : EntityComponentSystem(other)
        {
        }

        TEntityComponentSystem()
        {
            using MemberObjects_t = typename ct::GlobMemberObjects<T>::types;
            using Components_t = typename ct::ext::SelectComponents<MemberObjects_t>::type;
            addComponents(*this, Components_t{});
        }
    };

    template <class... T>
    struct TEntityComponentSystem<ct::VariadicTypedef<T...>, void> : EntityComponentSystem
    {
        template <class U>
        TEntityComponentSystem(const TEntityComponentSystem<U, void>& other)
            : EntityComponentSystem(other)
        {
        }

        TEntityComponentSystem()
        {
            addComponents(*this, ct::VariadicTypedef<T...>{});
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Implementation
    ////////////////////////////////////////////////////////////////////////////////
    template <class T>
    void TComponentProvider<T>::resize(uint32_t size)
    {
        m_data.resize(size);
    }

    template <class T>
    bool TComponentProvider<T>::providesComponent(mo::TypeInfo info) const
    {
        return info.template isType<T>();
    }

    template <class T>
    ct::TArrayView<T> TComponentProvider<T>::getComponentMutable()
    {
        return ct::TArrayView<T>(m_data.data(0).begin, m_data.size());
    }

    template <class T>
    ct::TArrayView<const T> TComponentProvider<T>::getComponent() const
    {
        return ct::TArrayView<const T>(m_data.data(0).begin, m_data.size());
    }

    template <class T>
    mo::TypeInfo TComponentProvider<T>::getComponentType() const
    {
        return mo::TypeInfo::create<T>();
    }
    template <class T>
    size_t TComponentProvider<T>::getNumEntities() const
    {
        return m_data.size();
    }

    template <class T>
    void TComponentProvider<T>::erase(uint32_t id)
    {
        m_data.erase(id);
    }

    template <class T>
    void TComponentProvider<T>::clear()
    {
        m_data.clear();
    }

} // namespace aq

namespace ct
{
    REFLECT_BEGIN(aq::EntityComponentSystem)
        PROPERTY(providers, &DataType::getProviders, &DataType::setProviders)
        MEMBER_FUNCTION(getNumComponents)
        MEMBER_FUNCTION(getNumEntities)
        MEMBER_FUNCTION(getComponentType)
        MEMBER_FUNCTION(erase)
    REFLECT_END;

    REFLECT_TEMPLATED_BEGIN(aq::TEntityComponentSystem)
    REFLECT_END;
} // namespace ct

#endif // AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP