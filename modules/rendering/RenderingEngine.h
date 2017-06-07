#pragma once
#include <MetaObject/IMetaObject.hpp>
#include <memory>

namespace mo
{
    class IParameter;
}

namespace aq
{
    // Render objects are objects that are rendered inside of a scene
    class IRenderObject : public TInterface<IRenderObject, mo::IMetaObject>
    {
    public:
        virtual void Render() = 0;
    };

    class IRenderObjectConstructor : public TInterface<IRenderObjectConstructor, mo::IMetaObject>
    {
    public:
        virtual std::shared_ptr<IRenderObject> Construct(std::shared_ptr<mo::IParam> param) = 0;
    };

    class IRenderObjectFactory : public TInterface<IRenderObjectFactory, mo::IMetaObject>
    {
    public:
        virtual std::shared_ptr<IRenderObject> Create(std::shared_ptr<mo::IParam> param) = 0;
        static void RegisterConstructorStatic(std::shared_ptr<IRenderObjectConstructor> constructor);

        virtual void RegisterConstructor(std::shared_ptr<IRenderObjectConstructor> constructor) = 0;
    };

    // Render scene holds all objects in a scene, enables / disables specific objects, etc.
    class IRenderScene : public TInterface<IRenderScene, IObject>
    {
    public:
        virtual void Render() = 0;
        virtual void AddRenderObject(std::shared_ptr<IRenderObject> obj) = 0;
    };

    class IRenderInteractor : TInterface<IRenderInteractor, IObject>
    {
    public:

    };

    // Render engine handles actually calling render, update, etc
    class IRenderEngine: public TInterface<IRenderEngine, IObject>
    {
    public:
        virtual void Render() = 0;
        virtual void AddRenderScene(std::shared_ptr<IRenderScene> scene) = 0;
    };
}