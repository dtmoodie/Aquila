#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "ObjectInterfacePerModule.h"
#include <functional>
#include <memory>
struct SystemTable;
struct IRuntimeObjectSystem;
struct IObjectInfo;
struct IObjectConstructor;
namespace mo
{
    class IMetaObject;
    template<class Sig> class TypedSlot;
    class Connection;
    class MO_EXPORTS MetaObjectFactory
    {
    public:
        IMetaObject*                       Create(const char* type_name, int interface_id = -1);
        template<class T> T*               Create(const char* type_name);
        IMetaObject*                       Get(ObjectId id, const char* type_name);
        
        static MetaObjectFactory*          Instance(SystemTable* system_table = nullptr);

        std::vector<std::string>           ListConstructableObjects(int interface_id = -1) const;
        std::string                        PrintAllObjectInfo(int interface_id = -1) const;

        std::vector<IObjectConstructor*>   GetConstructors(int interface_id = -1) const;
        IObjectConstructor*                GetConstructor(const char* type_name) const;
        IObjectInfo*                       GetObjectInfo(const char* type_name) const;
        std::vector<IObjectInfo*>          GetAllObjectInfo() const;

        bool                               LoadPlugin(const std::string& filename);
        int                                LoadPlugins(const std::string& path = "./");
        std::vector<std::string>           ListLoadedPlugins() const;

        // This function is inlined to guarantee it exists in the calling translation unit, which 
        // thus makes certain to load the correct PerModuleInterface instance
		inline void                        RegisterTranslationUnit()
		{
			SetupObjectConstructors(PerModuleInterface::GetInstance());
		}
        void                               SetupObjectConstructors(IPerModuleInterface* pPerModuleInterface);
        IRuntimeObjectSystem*              GetObjectSystem();

		// Recompilation stuffs
		bool AbortCompilation();
		bool CheckCompile();
		bool IsCurrentlyCompiling();
		bool IsCompileComplete();
		bool SwapObjects();
        void SetCompileCallback(std::function<void(const std::string, int)>& f);
        std::shared_ptr<Connection> ConnectConstructorAdded(TypedSlot<void(void)>* slot);

    private:
        MetaObjectFactory(SystemTable* system_table);
        ~MetaObjectFactory();
        struct impl;
        impl* _pimpl;
    };
    template<class T> 
    T* MetaObjectFactory::Create(const char* type_name)
    {
        return static_cast<T*>(Create(type_name, T::s_interfaceID));
    }
}
