#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <functional>
#include <memory>
namespace mo
{
	class IParameter;


	namespace Buffer
	{
		class MO_EXPORTS BufferFactory
		{
		public:
			typedef std::function<IParameter*(IParameter*)> create_buffer_f;

			static void RegisterFunction(TypeInfo type, const create_buffer_f& func, ParameterTypeFlags buffer_type_);
			static std::shared_ptr<IParameter> CreateProxy(IParameter* param, ParameterTypeFlags buffer_type_);
		};
	}
}