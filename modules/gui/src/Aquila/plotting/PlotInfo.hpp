#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"
namespace mo
{
    class IParameter;
}
namespace aq
{
    class AQUILA_EXPORTS PlotterInfo: public mo::IMetaObjectInfo
    {
    public:
        virtual bool AcceptsParameter(mo::IParam* parameter) = 0;
        //std::string Print() const;
    };
}

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template<class Type>
    struct MetaObjectInfoImpl<Type, aq::PlotterInfo> : public aq::PlotterInfo
    {
        bool AcceptsParameter(mo::IParam* parameter)
        {
            return Type::AcceptsParameter(parameter);
        }
    };
}