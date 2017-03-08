#include "MetaObject/Signals/RelayFactory.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <map>

using namespace mo;

struct RelayFactory::impl
{
	std::map<TypeInfo, std::function<ISignalRelay*(void)>> factories;
};

RelayFactory::RelayFactory()
{
	_pimpl = new impl();
}

RelayFactory::~RelayFactory()
{
	delete _pimpl;
}

RelayFactory* RelayFactory::Instance()
{
	static RelayFactory inst;
	return &inst;
}

void RelayFactory::RegisterCreator(std::function<ISignalRelay*(void)> f, const TypeInfo& type)
{
	_pimpl->factories[type] = f;
}

ISignalRelay* RelayFactory::Create(const TypeInfo& type)
{
	auto itr = _pimpl->factories.find(type);
	if (itr != _pimpl->factories.end())
	{
		return itr->second();
	}
	return nullptr;
}