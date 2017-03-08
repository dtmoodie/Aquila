#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <functional>
#include <memory>
namespace mo
{
    class IParameter;
    // Only include types that it makes sense to dynamically construct.
    // No reason to create a TypedParameterPtr most of the time because it is used to wrap
    // user owned data


    class MO_EXPORTS ParameterFactory
    {
    public:
        typedef std::function<IParameter*(void)> create_f;
        static ParameterFactory* instance();
        
        // Each specialization of a parameter must have a unique type
        void RegisterConstructor(TypeInfo data_type, create_f function, ParameterTypeFlags parameter_type);
        void RegisterConstructor(TypeInfo parameter_type, create_f function);

		// Give datatype and parameter type enum
        std::shared_ptr<IParameter> create(TypeInfo data_type, ParameterTypeFlags parameter_type); 
		// Must give exact parameter type, such as TypedParameter<int>
		std::shared_ptr<IParameter> create(TypeInfo parameter_type); 
    private:
        struct impl;
        std::shared_ptr<impl> pimpl;
    };
}