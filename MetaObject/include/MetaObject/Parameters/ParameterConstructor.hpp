#pragma once
#include "ParameterFactory.hpp"
#include "MetaObject/Detail/Enums.hpp"
namespace mo
{
	template<class T> class ParameterConstructor
	{
	public:
		ParameterConstructor()
		{
			ParameterFactory::instance()->RegisterConstructor(TypeInfo(typeid(typename T::ValueType)),
				std::bind(&ParameterConstructor<T>::create), T::Type);

			ParameterFactory::instance()->RegisterConstructor(TypeInfo(typeid(T)),
				std::bind(&ParameterConstructor<T>::create));
		}
		static IParameter* create()
		{
			return new T();
		}
	};
}