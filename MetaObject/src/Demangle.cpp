#include "MetaObject/Parameters/Demangle.hpp"
#include <map>
#ifdef HAVE_CEREAL
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#endif
using namespace mo;
std::map<TypeInfo, std::string>& Registry()
{
    static std::map<TypeInfo, std::string> inst;
    return inst;
}

std::string Demangle::TypeToName(TypeInfo type)
{
    std::map<TypeInfo, std::string>& reg = Registry();
    auto itr = reg.find(type);
    if(itr != reg.end())
    {
        return itr->second;
    }
    return type.name();
}

void Demangle::RegisterName(TypeInfo type, const char* name)
{
    std::map<TypeInfo, std::string>& reg = Registry();
    auto itr = reg.find(type);
    if(itr == reg.end())
    {
        reg[type] = name;
    }else
    {
        if(itr->second.empty())
            itr->second = std::string(name);
    }
}

void Demangle::RegisterType(TypeInfo type)
{
    std::map<TypeInfo, std::string>& reg = Registry();
    auto itr = reg.find(type);
    if(itr == reg.end())
    {
        reg[type] = "";
    }
}

void Demangle::GetTypeMapBinary(std::ostream& stream)
{
#ifdef HAVE_CEREAL
    cereal::BinaryOutputArchive ar(stream);
    std::map<std::string, size_t> lut;
    auto& reg = Registry();
    for(auto& itr : reg)
    {
        if(itr.second.size())
        {
            lut[itr.second] = itr.first.Get().hash_code();
        }else
        {
            lut[itr.first.name()] = itr.first.Get().hash_code();
        }
    }
    ar(lut);
#endif
}

void Demangle::SaveTypeMap(const std::string& filename)
{
    
}