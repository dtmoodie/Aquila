#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <functional>

namespace mo
{
	class TypeInfo;
	class ISignalRelay;
	class MO_EXPORTS RelayFactory
	{
	public:
		static RelayFactory* Instance();
		void RegisterCreator(std::function<ISignalRelay*(void)> f, const TypeInfo& type);
		ISignalRelay* Create(const TypeInfo& type);
	private:
		RelayFactory();
		~RelayFactory();
		struct impl;
		impl* _pimpl = nullptr;
	};
}