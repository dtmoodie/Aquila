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
    struct AQUILA_EXPORTS IDynamicProvider : virtual ct::ext::IComponentProvider
    {
        virtual void resize(uint32_t) = 0;
        virtual void erase(uint32_t) = 0;
    };

    template <class T>
    struct TComponentProvider : ct::ext::TComponentProvider<T>, virtual IDynamicProvider
    {
        void resize(uint32_t) override;
        bool providesComponent(const std::type_info& info) const override;
        void getComponentMutable(ct::TArrayView<T>&) override;
        void getComponent(ct::TArrayView<const T>&) const override;

        uint32_t getNumComponents() const override;
        const std::type_info* getComponentType(uint32_t idx) const override;

        ct::ext::IComponentProvider* getProvider(const std::type_info&) override;
        const ct::ext::IComponentProvider* getProvider(const std::type_info&) const override;
        size_t getNumEntities() const override;

        void erase(uint32_t) override;

      private:
        ct::ext::DataTableStorage<T> m_data;
    };

    struct AQUILA_EXPORTS EntityComponentSystem
    {
        uint32_t getNumComponents() const;
        const std::type_info* getComponentType(uint32_t idx) const;

        ct::ext::IComponentProvider* getProvider(const std::type_info& type);
        const ct::ext::IComponentProvider* getProvider(const std::type_info& type) const;
        void addProvider(ce::shared_ptr<IDynamicProvider>);

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
            auto provider = getProvider(typeid(T));
            if (provider)
            {
                auto typed = static_cast<ct::ext::TComponentProvider<T>*>(provider);
                typed->getComponentMutable(view);
            }

            return view;
        }

        template <class T>
        ct::TArrayView<const T> getComponent() const
        {
            ct::TArrayView<const T> view;
            auto provider = getProvider(typeid(T));
            if (provider)
            {
                auto typed = static_cast<const ct::ext::TComponentProvider<T>*>(provider);
                typed->getComponent(view);
            }
            return view;
        }

        void erase(uint32_t entity_id);

      private:
        std::vector<ce::shared_ptr<IDynamicProvider>> m_component_providers;

        uint32_t append();

        template <class T, ct::index_t I>
        void pushImpl(const T& obj, const uint32_t id, ct::Indexer<I> field_index)
        {
            auto ptr = ct::Reflect<T>::getPtr(field_index);
            using Component_t = typename decltype(ptr)::Data_t;
            auto provider = getProvider(typeid(Component_t));
            if (!provider)
            {
                MO_ASSERT_EQ(id, 0);
                // create a provider
                auto new_provider = ce::shared_ptr<TComponentProvider<Component_t>>::create();
                new_provider->resize(1);
                addProvider(ce::shared_ptr<IDynamicProvider>(std::move(new_provider)));
                provider = getProvider(typeid(Component_t));
            }
            MO_ASSERT(provider);
            auto typed = static_cast<ct::ext::TComponentProvider<Component_t>*>(provider);
            MO_ASSERT(typed);
            ct::TArrayView<Component_t> view;
            MO_ASSERT(provider->getComponentMutable(view));
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
            auto provider = getProvider(typeid(Component_t));
            MO_ASSERT(provider);
            auto typed = static_cast<const ct::ext::TComponentProvider<Component_t>*>(provider);
            MO_ASSERT(typed);
            ct::TArrayView<const Component_t> view;
            typed->getComponent(view);
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
    ////////////////////////////////////////////////////////////////////////////////
    // Implementation
    ////////////////////////////////////////////////////////////////////////////////
    template <class T>
    void TComponentProvider<T>::resize(uint32_t size)
    {
        m_data.resize(size);
    }

    template <class T>
    bool TComponentProvider<T>::providesComponent(const std::type_info& info) const
    {
        return &info == &typeid(T);
    }

    template <class T>
    void TComponentProvider<T>::getComponentMutable(ct::TArrayView<T>& out)
    {
        out = ct::TArrayView<T>(m_data.data(0).begin, m_data.size());
    }

    template <class T>
    void TComponentProvider<T>::getComponent(ct::TArrayView<const T>& out) const
    {
        out = ct::TArrayView<const T>(m_data.data(0).begin, m_data.size());
    }

    template <class T>
    uint32_t TComponentProvider<T>::getNumComponents() const
    {
        return 1;
    }

    template <class T>
    const std::type_info* TComponentProvider<T>::getComponentType(uint32_t idx) const
    {
        if (idx == 0)
        {
            return &typeid(T);
        }
        return nullptr;
    }

    template <class T>
    ct::ext::IComponentProvider* TComponentProvider<T>::getProvider(const std::type_info& type)
    {
        if (&type == &typeid(T))
        {
            return static_cast<ct::ext::TComponentProvider<T>*>(this);
        }
        return nullptr;
    }

    template <class T>
    const ct::ext::IComponentProvider* TComponentProvider<T>::getProvider(const std::type_info& type) const
    {
        if (&type == &typeid(T))
        {
            return static_cast<const ct::ext::TComponentProvider<T>*>(this);
        }
        return nullptr;
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

} // namespace aq

#endif // AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP