#ifndef AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP
#define AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP
#include "ComponentFactory.hpp"

#include <Aquila/core/detail/Export.hpp>

#include <MetaObject/core/IAsyncStream.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/logging/logging.hpp>

#include <MetaObject/runtime_reflection/visitor_traits/TypeInfo.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/array_adapter.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/string.hpp>

#include <MetaObject/params/TPublisher.hpp>
#include <MetaObject/params/TSubscriber.hpp>

#include <ct/extensions/DataTable.hpp>
#include <ct/static_asserts.hpp>

#include <ce/shared_ptr.hpp>
#include <memory>
namespace aq
{
    struct ComponentFactory;

    struct AQUILA_EXPORTS IComponentProvider
    {
        virtual ~IComponentProvider() = default;
        virtual void resize(uint32_t) = 0;
        virtual void erase(uint32_t) = 0;
        virtual void clear() = 0;

        virtual size_t getNumEntities() const = 0;

        virtual bool providesComponent(mo::TypeInfo info) const = 0;
        virtual mo::TypeInfo getComponentType() const = 0;

        virtual std::shared_ptr<IComponentProvider> clone() const = 0;

        virtual void save(mo::ISaveVisitor& visitor, const std::string& name) const = 0;
        virtual void load(mo::ILoadVisitor& visitor, const std::string& name) = 0;
    };

    template <class T>
    struct TComponentProvider : IComponentProvider
    {
        TComponentProvider();
        void resize(uint32_t) override;
        void erase(uint32_t) override;
        void clear() override;

        bool providesComponent(mo::TypeInfo info) const override;

        typename ct::ext::DataDimensionality<T>::TensorView getComponentMutable();
        typename ct::ext::DataDimensionality<T>::ConstTensorView getComponent() const;

        mo::TypeInfo getComponentType() const override;

        size_t getNumEntities() const override;

        std::shared_ptr<IComponentProvider> clone() const override;

        template <class U>
        void assign(uint32_t idx, U&& val);

        void save(mo::ISaveVisitor& visitor, const std::string& name) const override;
        void load(mo::ILoadVisitor& visitor, const std::string& name) override;

      private:
        ct::ext::DataTableStorage<T> m_data;
    };

    template <class T, class E = void>
    struct TEntityComponentSystem;
    struct ComponentAssigner;
    struct AQUILA_EXPORTS EntityComponentSystem
    {

        EntityComponentSystem() = default;
        EntityComponentSystem(const EntityComponentSystem& other);
        EntityComponentSystem(EntityComponentSystem&& other) = default;
        EntityComponentSystem& operator=(EntityComponentSystem&& other) = default;
        EntityComponentSystem& operator=(const EntityComponentSystem& other);

        virtual ~EntityComponentSystem() = default;

        uint32_t getNumComponents() const;
        mo::TypeInfo getComponentType(uint32_t idx) const;

        IComponentProvider* getProviderMutable(mo::TypeInfo type);
        const IComponentProvider* getProvider(mo::TypeInfo type) const;

        template <class T>
        TComponentProvider<T>* getProviderMutable();
        template <class T>
        const TComponentProvider<T>* getProvider() const;

        void addProvider(ce::shared_ptr<IComponentProvider>);
        template <class T>
        void addProvider();
        void setProviders(std::vector<ce::shared_ptr<IComponentProvider>> providers);

        std::vector<ce::shared_ptr<IComponentProvider>> getProviders() const;

        uint32_t getNumEntities() const;

        template <class T>
        void assignObject(T&& obj, const uint32_t id)
        {
            const auto idx = ct::Reflect<ct::decay_t<T>>::end();
            assignRecurse(obj, id, idx);
        }

        template <class T>
        void pushObject(T&& obj)
        {
            const uint32_t new_id = append();
            assignObject(std::forward<T>(obj), new_id);
        }

        template <class T>
        void push_back(const T& obj)
        {
            pushObject(obj);
        }

        template <class... Us>
        void pushComponents(const Us&... data)
        {
            uint32_t id = this->append();
            pushComponentsRecursive(id, data...);
        }

        template <class T>
        T at(uint32_t id) const
        {
            // Assemble an object by copying component data out
            T out;
            const auto idx = ct::Reflect<T>::end();
            getRecurse(out, id, idx);
            return out;
        }

        template <class T>
        typename ct::ext::DataDimensionality<T>::TensorView getComponentMutable()
        {
            typename ct::ext::DataDimensionality<T>::TensorView view;
            auto provider = getProviderMutable<T>();
            if (provider)
            {
                view = provider->getComponentMutable();
            }

            return view;
        }

        template <class T>
        typename ct::ext::DataDimensionality<T>::ConstTensorView getComponent() const
        {
            typename ct::ext::DataDimensionality<T>::ConstTensorView view;
            auto provider = getProvider<T>();
            if (provider)
            {
                return provider->getComponent();
            }
            return view;
        }

        void erase(uint32_t entity_id);
        void clear();
        void resize(uint32_t size);

        ComponentAssigner operator[](uint32_t idx);

        template <class Component_t>
        void assignComponent(const Component_t& obj, const uint32_t id)
        {
            auto provider = getProviderMutable<Component_t>();
            if (!provider)
            {
                MO_ASSERT_EQ(id, 0);
                // create a provider
                auto new_provider = ce::shared_ptr<TComponentProvider<Component_t>>::create();
                new_provider->resize(1);
                addProvider(ce::shared_ptr<IComponentProvider>(std::move(new_provider)));
                provider = getProviderMutable<Component_t>();
            }
            MO_ASSERT(provider);
            provider->assign(id, obj);
        }

        template <class T>
        void assignComponent(const ct::TArrayView<T>& obj, const uint32_t id)
        {
            auto provider = getProviderMutable<T>();
            if (!provider)
            {
                MO_ASSERT_EQ(id, 0);
                // create a provider
                auto new_provider = ce::shared_ptr<TComponentProvider<ct::TArrayView<T>>>::create();
                new_provider->resize(1);
                addProvider(ce::shared_ptr<IComponentProvider>(std::move(new_provider)));
                provider = getProvider(mo::TypeInfo::create<ct::TArrayView<T>>());
            }
            MO_ASSERT(provider);
            provider->assign(id, obj);
        }

      protected:
        uint32_t append();

        template <class U>
        void pushComponentsRecursive(uint32_t id, const U& head)
        {
            assignComponent(head, id);
        }

        template <class U, class... Us>
        ct::EnableIf<sizeof...(Us) != 0> pushComponentsRecursive(uint32_t id, const U& head, const Us&... tail)
        {
            assignComponent(head, id);
            pushComponentsRecursive(id, tail...);
        }

        template <class T, ct::index_t I>
        void assignImpl(const T& obj, const uint32_t id, ct::Indexer<I> field_index)
        {
            auto ptr = ct::Reflect<T>::getPtr(field_index);
            using Component_t = typename decltype(ptr)::Data_t;
            assignComponent<Component_t>(ptr.get(obj), id);
        }

        template <class T>
        void assignRecurse(const T& obj, const uint32_t entity_id, ct::Indexer<0> field_index)
        {
            assignImpl(obj, entity_id, field_index);
        }

        template <class T, ct::index_t I>
        void assignRecurse(const T& obj, const uint32_t entity_id, ct::Indexer<I> field_index)
        {
            assignImpl(obj, entity_id, field_index);
            const auto next_field = --field_index;
            assignRecurse(obj, entity_id, next_field);
        }

        template <class T, ct::index_t I>
        void getImpl(T& obj, const uint32_t entity_id, ct::Indexer<I> field_index) const
        {
            auto ptr = ct::Reflect<T>::getPtr(field_index);
            using Component_t = typename decltype(ptr)::Data_t;
            auto provider = getProvider<Component_t>();
            MO_ASSERT(provider);
            auto view = provider->getComponent();

            MO_ASSERT(entity_id < view.getShape()[0]);
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

      private:
        std::vector<ce::shared_ptr<IComponentProvider>> m_component_providers;
    };

    template <class T>
    TComponentProvider<T>* EntityComponentSystem::getProviderMutable()
    {
        IComponentProvider* provider = this->getProviderMutable(mo::TypeInfo::create<T>());
        if (provider)
        {
            return static_cast<TComponentProvider<T>*>(provider);
        }
        return nullptr;
    }

    template <class T>
    const TComponentProvider<T>* EntityComponentSystem::getProvider() const
    {
        const IComponentProvider* provider = this->getProvider(mo::TypeInfo::create<T>());
        if (provider)
        {
            return static_cast<const TComponentProvider<T>*>(provider);
        }
        return nullptr;
    }

    struct ComponentAssigner
    {
        ComponentAssigner(EntityComponentSystem&, uint32_t);

        template <class OBJECT>
        ComponentAssigner& assignObject(OBJECT&& obj)
        {
            const uint32_t id = m_idx;
            if (ct::IsReflected<OBJECT>::value)
            {
                auto provider = this->m_ecs.getProvider<OBJECT>();
                if (provider)
                {
                    this->m_ecs.assignComponent(std::forward<OBJECT>(obj), id);
                    return *this;
                }
            }
            this->m_ecs.assignObject(std::forward<OBJECT>(obj), id);
            return *this;
        }

        // By default assignment is for components
        template <class OBJ>
        ComponentAssigner& operator=(OBJ&& cmp)
        {
            return this->assignObject(std::forward<OBJ>(cmp));
        }

        /*template <class OBJECT>
        ComponentAssigner& assignObject(const OBJECT& obj)
        {
            this->m_ecs.pushObject(obj);
            return *this;
        }*/

        template <class T>
        ct::EnableIfReflected<T, T> convertTo() const
        {
            return m_ecs.at<T>(m_idx);
        }

        template <class T>
        ct::DisableIfReflected<T, T> convertTo() const
        {
            THROW(warn, "This type cannot be extracted");
            return {};
        }

        template <class COMPONENT>
        operator COMPONENT() const
        {
            const TComponentProvider<COMPONENT>* provider = this->m_ecs.getProvider<COMPONENT>();
            if (provider)
            {
                auto components = provider->getComponent();
                return components[this->m_idx];
            }
            return convertTo<COMPONENT>();
        }

      private:
        EntityComponentSystem& m_ecs;
        uint32_t m_idx;
    };

    template <class T>
    void addComponentsHelper(EntityComponentSystem& ecs, const ct::VariadicTypedef<T>*)
    {
        auto ptr = ce::make_shared<TComponentProvider<T>>();
        MO_ASSERT(ptr);
        ecs.addProvider(std::move(ptr));
    }

    template <class T, class... Ts>
    auto addComponentsHelper(EntityComponentSystem& ecs, const ct::VariadicTypedef<T, Ts...>*)
        -> ct::EnableIf<sizeof...(Ts) != 0>
    {
        auto ptr = ce::make_shared<TComponentProvider<T>>();
        MO_ASSERT(ptr);
        ecs.addProvider(std::move(ptr));
        addComponentsHelper(ecs, static_cast<const ct::VariadicTypedef<Ts...>*>(nullptr));
    }

    template <class U>
    ct::EnableIfReflected<U> addComponentsHelper(EntityComponentSystem& ecs, const U*)
    {
        using MemberObjects_t = typename ct::GlobMemberObjects<U>::types;
        addComponentsHelper(ecs, static_cast<const MemberObjects_t*>(nullptr));
    }

    template <class U>
    void addComponents(EntityComponentSystem& ecs)
    {
        addComponentsHelper(ecs, static_cast<const U*>(nullptr));
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
            addComponents<T>(*this);
        }

        T at(uint32_t id) const
        {
            return EntityComponentSystem::template at<T>(id);
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
            // using MemberObjects_t = typename ct::GlobMemberObjects<T>::types;
            // addComponents(*this, MemberObjects_t{});
            addComponents<T>(*this);
        }

        T at(uint32_t id) const
        {
            return EntityComponentSystem::template at<T>(id);
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
            addComponents<ct::VariadicTypedef<T...>>(*this);
        }

        void push_back(const T&... data)
        {
            const uint32_t new_id = append();
            pushComponentsRecursive(new_id, data...);
        }

      private:
        // TODO get returning a std::tuple<T...>
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Implementation
    ////////////////////////////////////////////////////////////////////////////////

    template <class T>
    TComponentProvider<T>::TComponentProvider()
    {
        static bool registered = false;
        if (!registered)
        {
            auto factory_instance = ComponentFactory::instance();
            MO_ASSERT(factory_instance);
            const auto type = mo::TypeInfo::create<T>();
            auto factory_func = []() -> ce::shared_ptr<IComponentProvider> {
                return ce::make_shared<TComponentProvider<T>>();
            };

            factory_instance->registerConstructor(type, std::move(factory_func));
        }
    }
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
    typename ct::ext::DataDimensionality<T>::TensorView TComponentProvider<T>::getComponentMutable()
    {
        return m_data.data(0);
    }

    template <class T>
    typename ct::ext::DataDimensionality<T>::ConstTensorView TComponentProvider<T>::getComponent() const
    {
        return m_data.data(0);
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

    template <class T>
    std::shared_ptr<IComponentProvider> TComponentProvider<T>::clone() const
    {
        return std::make_shared<TComponentProvider<T>>(*this);
    }

    template <class T>
    template <class U>
    void TComponentProvider<T>::assign(uint32_t idx, U&& val)
    {
        m_data.assign(idx, std::forward<U>(val));
    }

    template <class T>
    void TComponentProvider<T>::save(mo::ISaveVisitor& visitor, const std::string& name) const
    {
        visitor(&m_data, name);
    }

    template <class T>
    void TComponentProvider<T>::load(mo::ILoadVisitor& visitor, const std::string& name)
    {
        visitor(&m_data, name);
    }

} // namespace aq

namespace mo
{
    template <>
    struct PolymorphicSerializationHelper<aq::IComponentProvider>
    {
        template <class Ptr_t>
        static void load(ILoadVisitor& visitor, Ptr_t& ptr)
        {
            mo::TypeInfo type;
            visitor(&type, "type");
            auto component_factory = aq::ComponentFactory::instance();
            auto newly_created_component = component_factory->createComponent(type);
            MO_ASSERT(newly_created_component);
            ptr = newly_created_component;
        }
        template <class Ptr_t>
        static void save(ISaveVisitor& visitor, const Ptr_t& val)
        {
            const auto type = val->getComponentType();
            visitor(&type, "type");
        }
    };

    template <class T>
    struct TTraits<ce::shared_ptr<T>, 4> : virtual StructBase<ce::shared_ptr<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t cnt) const override
        {
            ce::shared_ptr<T>* ptr = ct::ptrCast<ce::shared_ptr<T>>(inst);
            for (size_t i = 0; i < cnt; ++i)
            {
                SharedPointerHelper<T>::load(visitor, ptr[i]);
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            const ce::shared_ptr<T>* ptr = ct::ptrCast<ce::shared_ptr<T>>(inst);
            for (size_t i = 0; i < cnt; ++i)
            {
                SharedPointerHelper<T>::save(visitor, ptr[i]);
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("ptr");
        }
    };

    // Overload for publishing and subscribing to an entity component system
    struct IEntityComponentSystemPublisher
    {
        virtual ~IEntityComponentSystemPublisher() = default;
        virtual void getComponents(std::vector<TypeInfo>& types) const = 0;
    };

    template <>
    struct TPublisher<aq::EntityComponentSystem> : TPublisherImpl<aq::EntityComponentSystem>,
                                                   IEntityComponentSystemPublisher
    {
        void getComponents(std::vector<TypeInfo>& types) const override;
    };

    template <class T>
    void populateTypes(std::vector<TypeInfo>& types, ct::VariadicTypedef<T>)
    {
        types.push_back(TypeInfo::create<T>());
    }

    template <class T, class... Ts>
    void populateTypes(std::vector<TypeInfo>& types, ct::VariadicTypedef<T, Ts...>)
    {
        types.push_back(TypeInfo::create<T>());
        populateTypes(types, ct::VariadicTypedef<Ts...>());
    }
    template <class T>
    struct TPublisher<aq::TEntityComponentSystem<T>> : TPublisherImpl<aq::EntityComponentSystem>,
                                                       IEntityComponentSystemPublisher
    {
        void getComponents(std::vector<TypeInfo>& types) const override
        {
            using MemberObjects_t = typename ct::GlobMemberObjects<T>::types;
            using Components_t = typename ct::ext::SelectComponents<MemberObjects_t>::type;
            types.reserve(Components_t::size());
            populateTypes(types, Components_t{});
        }
    };

    template <class... T>
    struct TPublisher<aq::TEntityComponentSystem<ct::VariadicTypedef<T...>>>
        : TPublisherImpl<aq::EntityComponentSystem>, IEntityComponentSystemPublisher
    {
        void getComponents(std::vector<TypeInfo>& types) const override
        {
            types.reserve(sizeof...(T));
            populateTypes(types, ct::VariadicTypedef<T...>{});
        }
    };

    template <class T>
    struct TSubscriber<aq::TEntityComponentSystem<T>> : TSubscriberImpl<aq::EntityComponentSystem>
    {
        using MemberObjects_t = typename ct::GlobMemberObjects<T>::types;
        using Components_t = typename ct::ext::SelectComponents<MemberObjects_t>::type;

        bool setInput(IPublisher* publisher = nullptr) override
        {
            if (!acceptsPublisher(*publisher))
            {
                return false;
            }
            return TSubscriberImpl<aq::EntityComponentSystem>::setInput(publisher);
        }

        bool acceptsPublisher(const IPublisher& param) const override
        {
            if (TSubscriberImpl<aq::EntityComponentSystem>::acceptsPublisher(param))
            {
                const auto& tparam = dynamic_cast<const IEntityComponentSystemPublisher&>(param);
                std::vector<TypeInfo> required_components;
                populateTypes(required_components, Components_t());
                std::vector<TypeInfo> published_components;
                tparam.getComponents(published_components);
                for (const auto& cmp : required_components)
                {
                    if (std::find(published_components.begin(), published_components.end(), cmp) ==
                        published_components.end())
                    {
                        this->getLogger().info(
                            "Unable to find required component {} provided from {} with the following components {}",
                            cmp,
                            param.getName(),
                            published_components);
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
    };

    template <class... T>
    struct TSubscriber<aq::TEntityComponentSystem<ct::VariadicTypedef<T...>>>
        : TSubscriberImpl<aq::EntityComponentSystem>
    {
        bool setInput(IPublisher* publisher = nullptr) override
        {
            if (!acceptsPublisher(*publisher))
            {
                return false;
            }
            return TSubscriberImpl<aq::EntityComponentSystem>::setInput(publisher);
        }

        bool acceptsPublisher(const IPublisher& param) const override
        {
            if (TSubscriberImpl<aq::EntityComponentSystem>::acceptsPublisher(param))
            {
                const auto& tparam = dynamic_cast<const IEntityComponentSystemPublisher&>(param);
                std::vector<TypeInfo> required_components;
                populateTypes(required_components, ct::VariadicTypedef<T...>());
                std::vector<TypeInfo> published_components;
                tparam.getComponents(published_components);
                for (const auto& cmp : required_components)
                {
                    if (std::find(published_components.begin(), published_components.end(), cmp) ==
                        published_components.end())
                    {
                        this->getLogger().info(
                            "Unable to find required component {} provided from {} with the following components {}",
                            cmp,
                            param.getName(),
                            published_components);
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
    };

} // namespace mo

namespace ct
{
    REFLECT_BEGIN(aq::IComponentProvider)
        MEMBER_FUNCTION(getComponentType)
        MEMBER_FUNCTION(getNumEntities)
        MEMBER_FUNCTION(clear)
        MEMBER_FUNCTION(resize)
        MEMBER_FUNCTION(erase)
        MEMBER_FUNCTION(providesComponent)
    REFLECT_END;

    REFLECT_TEMPLATED_DERIVED(aq::TComponentProvider, aq::IComponentProvider)
        PROPERTY(data, &DataType::getComponent, &DataType::getComponentMutable)
    REFLECT_END;

    REFLECT_BEGIN(aq::EntityComponentSystem)
        PROPERTY(providers, &DataType::getProviders, &DataType::setProviders)
        MEMBER_FUNCTION(getNumComponents)
        MEMBER_FUNCTION(getNumEntities)
        MEMBER_FUNCTION(getComponentType)
        MEMBER_FUNCTION(erase)
        MEMBER_FUNCTION(clear)
    REFLECT_END;

    REFLECT_TEMPLATED_DERIVED(aq::TEntityComponentSystem, aq::EntityComponentSystem)
        // MEMBER_FUNCTION(at, &DataType::at)
    REFLECT_END;
} // namespace ct

#endif // AQ_TYPES_ENTITY_COMPONENT_SYSTEM_HPP
