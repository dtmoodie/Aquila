#pragma once
#include "BufferFactory.hpp"

namespace mo
{
    template<class T> class BufferConstructor
    {
    public:
        BufferConstructor()
        {
            static_assert(T::Type != TypedParameter_e, "Typed parameter not a buffer");
            Buffer::BufferFactory::RegisterFunction(
                TypeInfo(typeid(typename T::ValueType)),
                std::bind(&BufferConstructor<T>::create, std::placeholders::_1), 
                T::Type);
        }
        static IParameter* create(IParameter* input)
        {
            T* ptr = new T();
            if(ptr->SetInput(input))
            {
                return ptr;
            }
            delete ptr;
            return nullptr;
        }
    };
}