#include <MetaObject/Parameters/Types.hpp>
using namespace mo;

ReadFile::ReadFile(const std::string& str) : 
    boost::filesystem::path(str) 
{}

WriteFile::WriteFile(const std::string& file) : 
    boost::filesystem::path(file) 
{}

ReadDirectory::ReadDirectory(const boost::filesystem::path& path) :
    boost::filesystem::path(path) 
{}

WriteDirectory::WriteDirectory(const std::string& str) : 
    boost::filesystem::path(str) 
{}


EnumParameter::EnumParameter(const std::initializer_list<std::pair<const char*, int>>& values)
{
    enumerations.clear();
    this->values.clear();
    for (auto itr = values.begin(); itr != values.end(); ++itr)
    {
        enumerations.emplace_back(itr->first);
        this->values.emplace_back(itr->second);
    }
}
EnumParameter::EnumParameter()
{
    currentSelection = 0;
}

void EnumParameter::SetValue(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values)
{
    auto iItr = values.begin();
    auto nItr = string.begin();
    enumerations.clear();
    this->values.clear();
    for (; iItr != values.end() && nItr != string.end(); ++iItr, ++nItr)
    {
        enumerations.push_back(*nItr);
        this->values.push_back(*iItr);
    }
}

void EnumParameter::addEnum(int value, const ::std::string& enumeration)
{
    enumerations.push_back(enumeration);
    values.push_back(value);
}
int EnumParameter::getValue()
{
    if (values.empty() || currentSelection >= values.size())
    {
        throw std::range_error("values.empty() || currentSelection >= values.size()");
    }
    return values[currentSelection];
}
std::string EnumParameter::getEnum()
{
    if (currentSelection >= values.size())
    {
        throw std::range_error("currentSelection >= values.size()");
    }
    return enumerations[currentSelection];
}