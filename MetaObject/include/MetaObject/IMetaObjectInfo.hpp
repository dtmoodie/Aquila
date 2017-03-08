#pragma once
#include "IObjectInfo.h"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Detail/Export.hpp"

#include <vector>
namespace mo
{
    struct ParameterInfo;
	struct SignalInfo;
	struct SlotInfo;
	class MO_EXPORTS IMetaObjectInfo: public IObjectInfo
	{
    public:
		virtual std::vector<ParameterInfo*> GetParameterInfo() const = 0;
		virtual std::vector<SignalInfo*>    GetSignalInfo() const = 0;
		virtual std::vector<SlotInfo*>      GetSlotInfo() const = 0;
        virtual TypeInfo                    GetTypeInfo() const = 0;
        virtual std::string                 Print() const;
        virtual std::string                 GetDisplayName() const
        {
            return GetObjectName();
        }
	};
}