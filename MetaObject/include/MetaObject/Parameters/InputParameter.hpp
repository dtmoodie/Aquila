/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#pragma once
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
#include <functional>
#include <string>
#include <memory>
namespace mo
{
    
    class MO_EXPORTS InputParameter: virtual public IParameter
    {
    public:
        typedef std::function<bool(std::weak_ptr<IParameter>)> qualifier_f;
        typedef std::shared_ptr<InputParameter> Ptr;

        InputParameter(){}
        virtual ~InputParameter() {}

        // This loads the value at the requested timestamp into the input
        // parameter such that it can be read
        virtual bool GetInput(long long ts = -1) = 0;

        // This gets a pointer to the variable that feeds into this input
        virtual IParameter* GetInputParam() = 0;

        virtual bool SetInput(std::shared_ptr<IParameter> param) = 0;
        virtual bool SetInput(IParameter* param = nullptr) = 0;
        
        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const = 0;
        virtual bool AcceptsInput(IParameter* param) const = 0;
        virtual bool AcceptsType(TypeInfo type) const = 0;
        
        void SetQualifier(std::function<bool(std::weak_ptr<IParameter>)> f)
        {
            qualifier = f;
        }
    protected:
        InputParameter( const InputParameter& ) = delete;
        InputParameter& operator=(const InputParameter& ) = delete;
        InputParameter& operator=(InputParameter&& ) = delete;
        qualifier_f qualifier;
    };
}