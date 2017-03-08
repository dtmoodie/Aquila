#include "MetaObject/MetaObjectInfoDatabase.hpp"
#include "MetaObject/MetaObjectInfo.hpp"
#include <map>
using namespace mo;

struct MetaObjectInfoDatabase::impl
{
    std::map<std::string, IMetaObjectInfo*> info;
};

MetaObjectInfoDatabase::MetaObjectInfoDatabase()
{
    _pimpl = new impl();
}
MetaObjectInfoDatabase::~MetaObjectInfoDatabase()
{
    delete _pimpl;
}

MetaObjectInfoDatabase* MetaObjectInfoDatabase::Instance()
{
    static MetaObjectInfoDatabase g_inst;
    return &g_inst;
}
        
void MetaObjectInfoDatabase::RegisterInfo(IMetaObjectInfo* info)
{
    _pimpl->info[info->GetObjectName()] = info;
}
        
std::vector<IMetaObjectInfo*> MetaObjectInfoDatabase::GetMetaObjectInfo()
{
    std::vector<IMetaObjectInfo*> output;
    for(auto& itr : _pimpl->info)
    {
        output.push_back(itr.second);
    }
    return output;
}

IMetaObjectInfo* MetaObjectInfoDatabase::GetMetaObjectInfo(std::string name)
{
    auto itr = _pimpl->info.find(name);
    if(itr != _pimpl->info.end())
    {
        return itr->second;
    }
    return nullptr;
}