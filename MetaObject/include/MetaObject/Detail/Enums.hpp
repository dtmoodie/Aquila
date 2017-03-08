#pragma once
#include "Export.hpp"
#include <string>
namespace mo
{
    enum ParameterType
    {
        None_e = 0,
        Input_e = 1,
        Output_e = 2,
        State_e = 4,
        Control_e = 8,
        Buffer_e = 16,
        Optional_e = 32,
        Desynced_e = 64
    };
    MO_EXPORTS std::string ParameteTypeToString(ParameterType type);
    MO_EXPORTS ParameterType StringToParameteType(const std::string& str);
    enum ParameterTypeFlags
    {
        TypedParameter_e = 0,
        cbuffer_e ,
        cmap_e,
        map_e,
        StreamBuffer_e,
        BlockingStreamBuffer_e,
        NNStreamBuffer_e,
        ForceBufferedConnection_e = 1024,
        ForceDirectConnection_e = 2048
    };
    MO_EXPORTS std::string ParameterTypeFlagsToString(ParameterTypeFlags flags);
    MO_EXPORTS ParameterTypeFlags StringToParameterTypeFlags(const std::string& str);

}
