#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "BufferFactory.hpp"

namespace mo
{
    class IParameter;
    namespace Buffer
    {
        template<typename T> class Proxy;
        
        template<typename T> struct BufferConstructor
        {
			BufferConstructor()
            {
                BufferFactory::RegisterFunction(TypeInfo(typeid(typename T::ValueType)),
					std::bind(&BufferConstructor<T>::create_buffer, std::placeholders::_1),
					T::BufferType);
            }
            static IParameter* create_buffer(IParameter* input)
            {
                if (auto typed_param = dynamic_cast<ITypedParameter<T>*>(input))
                {
                    return new Proxy<T>(typed_param, new T("map for " + input->GetTreeName()));
                }
                return nullptr;
            }
        };
    }
}
