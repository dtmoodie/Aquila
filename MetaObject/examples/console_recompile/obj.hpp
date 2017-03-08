#pragma once
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"

struct printable: public mo::IMetaObject
{
	virtual void print();
	MO_BEGIN(printable)
	MO_END;
};