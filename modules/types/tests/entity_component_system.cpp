#include <Aquila/types/EntityComponentSystem.hpp>
#include <ct/reflect/compare.hpp>
#include <ct/reflect/print.hpp>

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

    auto new_obj = ecs.get<GameObject>(0);
    ASSERT_EQ(obj, new_obj);

    auto sphere = ecs.get<Sphere>(0);
    ASSERT_EQ(sphere.velocity, new_obj.velocity);
    ASSERT_EQ(sphere.position, new_obj.position);

    auto view = ecs.getComponent<Orientation>();
    ASSERT_EQ(view.size(), 10);
    ASSERT_EQ(view[0], new_obj.orientation);

    auto sphere_view = ecs.getComponent<Sphere>();
    ASSERT_FALSE(sphere_view.size());
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

    auto new_obj = ecs.get<GameObject>(0);
    ASSERT_EQ(GameObject::init(0), new_obj);
}
