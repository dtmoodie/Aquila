#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "MetaObject/MetaObjectInfo.hpp"
namespace mo
{
    class IParameter;
}
namespace aq
{
    class AQUILA_EXPORTS PlotterInfo: public mo::IMetaObjectInfo
    {
    public:
        virtual bool AcceptsParameter(mo::IParameter* parameter) = 0;
        //std::string Print() const;
    };
}

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template<class Type>
    struct MetaObjectInfoImpl<Type, aq::PlotterInfo> : public aq::PlotterInfo
    {
        bool AcceptsParameter(mo::IParameter* parameter)
        {
            return Type::AcceptsParameter(parameter);
        }
    };
}