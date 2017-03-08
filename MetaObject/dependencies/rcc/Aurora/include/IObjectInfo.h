#pragma once
#include <string>
struct IObjectInfo
{
    enum ObjectInfoType
    {
        base = 0,
        node = 1,
        frame_grabber
    };
    virtual int GetObjectInfoType() = 0;
    virtual std::string GetObjectName() = 0;
    virtual std::string GetObjectTooltip() = 0;
    virtual std::string GetObjectHelp() = 0;
};