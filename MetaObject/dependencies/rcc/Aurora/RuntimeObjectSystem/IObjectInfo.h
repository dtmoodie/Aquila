#pragma once
#include <string>
struct IObjectInfo
{
    virtual unsigned int GetInterfaceId() const = 0;
    // This is what actually gets displayed
    virtual std::string GetObjectName() const = 0;
    virtual std::string GetObjectTooltip() const = 0;
    virtual std::string GetObjectHelp() const = 0;
    virtual std::string Print() const = 0;
};
