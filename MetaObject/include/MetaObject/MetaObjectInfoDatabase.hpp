#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <vector>
#include <string>
namespace mo
{
    class IMetaObjectInfo;
    class MO_EXPORTS MetaObjectInfoDatabase
    {
    public:
        static MetaObjectInfoDatabase* Instance();
        
        void RegisterInfo(IMetaObjectInfo* info);
        
        std::vector<IMetaObjectInfo*> GetMetaObjectInfo();
        IMetaObjectInfo* GetMetaObjectInfo(std::string name);
    private:
        MetaObjectInfoDatabase();
        ~MetaObjectInfoDatabase();
        struct impl;
        impl* _pimpl;
    };
}