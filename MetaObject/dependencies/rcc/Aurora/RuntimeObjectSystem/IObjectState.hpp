#pragma once
#include "ObjectInterface.h"
struct IObject;
struct IObjectConstructor;
template<class T> class TObjectConstructorConcrete;
template<class T> class TActual;
namespace rcc
{
    template<class T> class shared_ptr;
    template<class T> class weak_ptr;
}

struct IObjectSharedState
{
    static IObjectSharedState* Get(IObject* obj);
    IObjectSharedState(IObject* obj, IObjectConstructor* constructor);
    ~IObjectSharedState();
    IObject* GetIObject();
    rcc::shared_ptr<IObject> GetSharedPtr();
    rcc::weak_ptr<IObject> GetWeakPtr();
    void IncrementObject();
    void IncrementState();
    void DecrementObject();
    void DecrementState();
    int ObjectCount() const;
    int StateCount() const;
protected:
    friend struct IObject;
    friend struct IObjectConstructor;
    template<class T> friend class TObjectConstructorConcrete;
    template<class T> friend class TActual;

    void SetObject(IObject* object);
    void SetConstructor(IObjectConstructor* constructor);

    IObject* object;
    IObjectConstructor* constructor;
    int object_ref_count;
    int state_ref_count;
    PerTypeObjectId id;
};
