#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "MetaObject/object/MetaObjectInfo.hpp"

namespace mo
{
    class IParam;
}
namespace aq
{
    class AQUILA_EXPORTS PlotterInfo : public mo::IMetaObjectInfo
    {
      public:
        virtual bool acceptsParameter(mo::IParam* parameter) const = 0;
        // std::string Print() const;
    };
} // namespace aq

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template <class Type>
    struct MetaObjectInfoImpl<Type, aq::PlotterInfo> : public aq::PlotterInfo
    {
        bool acceptsParameter(mo::IParam* parameter) const override
        {
            return Type::AcceptsParameter(parameter);
        }
    };
} // namespace mo