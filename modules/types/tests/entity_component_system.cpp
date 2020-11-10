#include <Aquila/types/EntityComponentSystem.hpp>

#include <ct/reflect/compare.hpp>
#include <ct/reflect/print.hpp>
#include <ct/static_asserts.hpp>

#include <MetaObject/runtime_reflection/TraitInterface.hpp>
#include <MetaObject/runtime_reflection/VisitorTraits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>
#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>

#include <gtest/gtest.h>

struct Velocity : ct::ext::Component
{
    REFLECT_INTERNAL_BEGIN(Velocity)
        REFLECT_INTERNAL_MEMBER(float, x)
        REFLECT_INTERNAL_MEMBER(float, y)
        REFLECT_INTERNAL_MEMBER(float, z)
    REFLECT_INTERNAL_END;
};

struct Position : ct::ext::Component
{
    REFLECT_INTERNAL_BEGIN(Position)
        REFLECT_INTERNAL_MEMBER(float, x)
        REFLECT_INTERNAL_MEMBER(float, y)
        REFLECT_INTERNAL_MEMBER(float, z)
    REFLECT_INTERNAL_END;
};

struct Orientation : ct::ext::Component
{
    REFLECT_INTERNAL_BEGIN(Orientation)
        REFLECT_INTERNAL_MEMBER(float, x)
        REFLECT_INTERNAL_MEMBER(float, y)
        REFLECT_INTERNAL_MEMBER(float, z)
        REFLECT_INTERNAL_MEMBER(float, w)
    REFLECT_INTERNAL_END;
};

struct GameObject
{
    static GameObject init(int val = 0)
    {
        GameObject obj;
        obj.velocity.x = 0;
        obj.velocity.y = 1 * val;
        obj.velocity.z = 2 * val;
        obj.position.x = 3 * val;
        obj.position.y = 3 * val;
        obj.position.z = 3 * val;
        obj.orientation.w = 0;
        obj.orientation.x = 0;
        obj.orientation.y = 0;
        obj.orientation.z = 0;
        return obj;
    }
    REFLECT_INTERNAL_BEGIN(GameObject)
        REFLECT_INTERNAL_MEMBER(Velocity, velocity)
        REFLECT_INTERNAL_MEMBER(Position, position)
        REFLECT_INTERNAL_MEMBER(Orientation, orientation)
    REFLECT_INTERNAL_END;
};

struct Sphere
{
    REFLECT_INTERNAL_BEGIN(Sphere)
        REFLECT_INTERNAL_MEMBER(Velocity, velocity)
        REFLECT_INTERNAL_MEMBER(Position, position)
    REFLECT_INTERNAL_END;
};

TEST(entity_component_system, initialization)
{
    aq::EntityComponentSystem ecs;
    auto obj = GameObject::init();
    for (size_t i = 0; i < 10; ++i)
    {
        ecs.push_back(obj);
    }
    ASSERT_EQ(ecs.getNumComponents(), 3);
    ASSERT_EQ(ecs.getNumEntities(), 10);

    auto new_obj = ecs.at<GameObject>(0);
    ASSERT_EQ(obj, new_obj);

    auto sphere = ecs.at<Sphere>(0);
    ASSERT_EQ(sphere.velocity, new_obj.velocity);
    ASSERT_EQ(sphere.position, new_obj.position);

    auto view = ecs.getComponent<Orientation>();
    ASSERT_EQ(view.getShape()[0], 10);
    ASSERT_EQ(view[0], new_obj.orientation);

    auto sphere_view = ecs.getComponent<Sphere>();
    ASSERT_FALSE(sphere_view.getShape()[0]);
}

TEST(entity_component_system, assert_wrong_copy)
{
    aq::TEntityComponentSystem<GameObject> ecs;

    {
        using Type = aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity, Orientation, Position>>;
        EXPECT_NO_THROW(Type other(ecs));
    }
    {
        using Type = aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity, Orientation, Position, Sphere>>;
        EXPECT_THROW(Type other(ecs), mo::TExceptionWithCallstack<std::runtime_error>);
    }
}

TEST(entity_component_system, unique_components)
{
    struct Tag0;
    struct Tag1;
    using Comp0 = ct::ext::ScalarComponent<float, Tag0>;
    using Comp1 = ct::ext::ScalarComponent<float, Tag1>;
    aq::EntityComponentSystem ecs;
    auto provider0 = ce::shared_ptr<aq::TComponentProvider<Comp0>>::create();
    auto provider1 = ce::shared_ptr<aq::TComponentProvider<Comp1>>::create();
    ecs.addProvider(provider0);
    ecs.addProvider(provider1);
    auto P0 = ecs.getProviderMutable<Comp0>();
    auto P1 = ecs.getProviderMutable<Comp1>();
    ASSERT_TRUE(P0);
    ASSERT_TRUE(P1);

    ct::StaticEqualTypes<ct::decay_t<decltype(*P0->getComponentMutable().data())>, float>{};
    ct::StaticEqualTypes<ct::decay_t<decltype(*P1->getComponentMutable().data())>, float>{};
}

TEST(entity_component_system, assignment)
{
    aq::EntityComponentSystem ecs;
    auto obj = GameObject::init();
    for (size_t i = 0; i < 10; ++i)
    {
        obj = GameObject::init(i);
        ecs.push_back(obj);
    }
    ecs[2].assignObject(GameObject::init(4));
    ASSERT_EQ(ecs.at<GameObject>(2), GameObject::init(4));

    obj = ecs[2];
    ASSERT_EQ(obj, GameObject::init(4));
}

TEST(entity_component_system, erase)
{
    aq::EntityComponentSystem ecs;

    for (size_t i = 0; i < 10; ++i)
    {
        auto obj = GameObject::init(i);
        ecs.push_back(obj);
    }
    ASSERT_EQ(ecs.getNumComponents(), 3);
    ASSERT_EQ(ecs.getNumEntities(), 10);

    auto new_obj = ecs.at<GameObject>(0);
    ASSERT_EQ(GameObject::init(0), new_obj);

    ecs.erase(5);
    ASSERT_EQ(ecs.getNumEntities(), 9);
    new_obj = ecs.at<GameObject>(5);
    ASSERT_EQ(new_obj, GameObject::init(6));
}

TEST(entity_component_system, copy_on_write)
{
    aq::EntityComponentSystem ecs;

    for (size_t i = 0; i < 10; ++i)
    {
        auto obj = GameObject::init(i);
        ecs.push_back(obj);
    }

    aq::EntityComponentSystem copy(ecs);

    EXPECT_EQ(copy.getComponent<Orientation>().data(), ecs.getComponent<Orientation>().data());
    EXPECT_EQ(copy.getComponent<Velocity>().data(), ecs.getComponent<Velocity>().data());
    EXPECT_EQ(copy.getComponent<Position>().data(), ecs.getComponent<Position>().data());

    copy.getComponentMutable<Orientation>();

    EXPECT_NE(copy.getComponent<Orientation>().data(), ecs.getComponent<Orientation>().data());
    EXPECT_EQ(copy.getComponent<Velocity>().data(), ecs.getComponent<Velocity>().data());
    EXPECT_EQ(copy.getComponent<Position>().data(), ecs.getComponent<Position>().data());

    copy.erase(4);

    EXPECT_NE(copy.getComponent<Orientation>().data(), ecs.getComponent<Orientation>().data());
    EXPECT_NE(copy.getComponent<Velocity>().data(), ecs.getComponent<Velocity>().data());
    EXPECT_NE(copy.getComponent<Position>().data(), ecs.getComponent<Position>().data());
}

TEST(entity_component_system, typed_ecs)
{
    aq::TEntityComponentSystem<Sphere> ecs;
    ecs.push_back(Sphere{});

    auto velocity = ecs.getComponent<Velocity>();
    ASSERT_EQ(velocity.getShape()[0], 1);
}

TEST(entity_component_system, type_erase)
{
    aq::TEntityComponentSystem<Sphere> tecs;
    tecs.push_back(Sphere{});

    aq::EntityComponentSystem ecs(tecs);

    auto velocity = ecs.getComponent<Velocity>();
    ASSERT_EQ(velocity.getShape()[0], 1);
}

TEST(entity_component_system, serialization_json)
{

    static_assert(mo::HasMemberLoad<aq::IComponentProvider>::value, "");
    static_assert(mo::HasMemberSave<aq::IComponentProvider>::value, "");

    aq::EntityComponentSystem ecs;

    for (size_t i = 0; i < 10; ++i)
    {
        auto obj = GameObject::init(i);
        ecs.push_back(obj);
    }

    std::stringstream ss;
    {
        mo::JSONSaver saver(ss);
        saver(&ecs, "ecs");
    }
    {
        mo::JSONLoader loader(ss);
        aq::EntityComponentSystem loaded_ecs;
        loader(&loaded_ecs, "ecs");
        ASSERT_EQ(loaded_ecs.getNumComponents(), 3);

        auto providers = loaded_ecs.getProviders();

        ASSERT_TRUE(std::count_if(providers.begin(), providers.end(), [](auto provider) {
            return provider->getComponentType().template isType<Velocity>();
        }));

        ASSERT_TRUE(std::count_if(providers.begin(), providers.end(), [](auto provider) {
            return provider->getComponentType().template isType<Position>();
        }));

        ASSERT_TRUE(std::count_if(providers.begin(), providers.end(), [](auto provider) {
            return provider->getComponentType().template isType<Orientation>();
        }));

        auto velocity = loaded_ecs.getComponent<Velocity>();
        ASSERT_EQ(velocity.getShape()[0], 10);
    }
}

TEST(entity_component_system, serialization_binary)
{

    aq::EntityComponentSystem ecs;

    for (size_t i = 0; i < 10; ++i)
    {
        auto obj = GameObject::init(i);
        ecs.push_back(obj);
    }

    std::stringstream ss;
    {
        mo::BinarySaver saver(ss);
        saver(&ecs, "ecs");
    }
    {
        mo::BinaryLoader loader(ss);
        aq::EntityComponentSystem loaded_ecs;
        loader(&loaded_ecs, "ecs");
        ASSERT_EQ(loaded_ecs.getNumComponents(), 3);

        auto providers = loaded_ecs.getProviders();

        ASSERT_TRUE(std::count_if(providers.begin(), providers.end(), [](auto provider) {
            return provider->getComponentType().template isType<Velocity>();
        }));

        ASSERT_TRUE(std::count_if(providers.begin(), providers.end(), [](auto provider) {
            return provider->getComponentType().template isType<Position>();
        }));

        ASSERT_TRUE(std::count_if(providers.begin(), providers.end(), [](auto provider) {
            return provider->getComponentType().template isType<Orientation>();
        }));

        auto velocity = loaded_ecs.getComponent<Velocity>();
        ASSERT_EQ(velocity.getShape()[0], 10);
    }
}

TEST(entity_component_system, pub_sub_subscription_single_components)
{
    auto stream = mo::IAsyncStream::create();
    mo::TPublisher<aq::TEntityComponentSystem<GameObject>> pub;
    {
        mo::TSubscriber<aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity, Position, Orientation>>> sub;
        ASSERT_TRUE(sub.setInput(&pub));
    }
    {
        mo::TSubscriber<aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity, Position>>> sub;
        ASSERT_TRUE(sub.setInput(&pub));
    }

    {
        mo::TSubscriber<aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity>>> sub;
        ASSERT_TRUE(sub.setInput(&pub));
    }
    {
        mo::TSubscriber<aq::TEntityComponentSystem<ct::VariadicTypedef<float>>> sub;
        ASSERT_FALSE(sub.setInput(&pub));
    }
}

TEST(entity_component_system, pub_sub_composite_object)
{
    auto stream = mo::IAsyncStream::create();
    mo::TPublisher<aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity, Position, Orientation>>> pub;
    {
        mo::TSubscriber<aq::TEntityComponentSystem<GameObject>> sub;
        ASSERT_TRUE(sub.setInput(&pub));
    }
}

TEST(entity_component_system, pub_sub_data)
{
    auto stream = mo::IAsyncStream::create();
    mo::TPublisher<aq::TEntityComponentSystem<GameObject>> pub;
    mo::TSubscriber<aq::TEntityComponentSystem<ct::VariadicTypedef<Velocity, Position, Orientation>>> sub;
    ASSERT_TRUE(sub.setInput(&pub));

    aq::EntityComponentSystem ecs;
    for (size_t i = 0; i < 10; ++i)
    {
        auto obj = GameObject::init(i);
        ecs.push_back(obj);
    }

    pub.publish(std::move(ecs));

    auto data = sub.getTypedData();
    ASSERT_TRUE(data);
    auto velocity = data->data.getComponent<Velocity>();
    ASSERT_EQ(velocity.getShape()[0], 10);

    auto position = data->data.getComponent<Position>();
    ASSERT_EQ(position.getShape()[0], 10);

    auto orientation = data->data.getComponent<Orientation>();
    ASSERT_EQ(orientation.getShape()[0], 10);
}