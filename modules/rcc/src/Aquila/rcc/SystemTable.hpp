#pragma once
#include <Aquila/core/detail/Export.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <map>
#include <memory>
struct AQUILA_EXPORTS ISingleton {
    virtual ~ISingleton() {}
};

template <typename T>
struct Singleton : public ISingleton {
    Singleton(T* ptr_)
        : ptr(ptr_) {}
    virtual ~Singleton() {
        delete ptr;
    }
    T* ptr;
    operator T*() const { return ptr; }
};

template <typename T>
struct IObjectSingleton : public ISingleton {
    rcc::shared_ptr<T> ptr;
    IObjectSingleton(T* ptr_)
        : ptr(ptr_) {}
    IObjectSingleton(const rcc::shared_ptr<T>& ptr_)
        : ptr(ptr_) {}
    operator T*() const { return ptr.get(); }
};

struct AQUILA_EXPORTS SystemTable {
    SystemTable();
    void cleanUp();
    // These are per stream singletons
    template <typename T>
    T* getSingleton() {
        auto g_itr = g_singletons.find(mo::TypeInfo(typeid(T)));
        if (g_itr != g_singletons.end()) {
            return static_cast<Singleton<T>*>(g_itr->second.get())->ptr;
        }
        return nullptr;
    }

    template <typename T>
    typename std::enable_if<!std::is_base_of<IObject, T>::value, T*>::type setSingleton(T* singleton) {
        g_singletons[mo::TypeInfo(typeid(T))] = std::shared_ptr<ISingleton>(new Singleton<T>(singleton));
        return singleton;
    }

    template <typename T>
    typename std::enable_if<std::is_base_of<IObject, T>::value, T*>::type setSingleton(T* singleton) {
        g_singletons[mo::TypeInfo(typeid(T))] = std::shared_ptr<ISingleton>(new IObjectSingleton<T>(singleton));
        return singleton;
    }

    void deleteSingleton(mo::TypeInfo type);

    template <typename T>
    void deleteSingleton() {
        deleteSingleton(mo::TypeInfo(typeid(T)));
    }

private:
    std::map<mo::TypeInfo, std::shared_ptr<ISingleton> > g_singletons;
};
